"""
Hand-verified requirement trees for CS-BS, EE-BS and PHYS-BS.

Source: the live 2026-27 bulletin pages (bulletin.stanford.edu/programs/
{CS-BS,EE-BS,PHYS-BS}), read in a real browser (Claude in Chrome) in Nov
2026, plus the School of Engineering's shared Engineering Fundamentals /
Science / Technology in Society lists from the ENGR-BS page's "School
Requirements" section (the CS-BS and EE-BS pages only point at those lists;
they don't repeat them).

Why this exists instead of the generic bulletin_parser.py path:
- The cached samples/*.txt for all three programs were stale. EE-BS's
  Electives list had 4 of its 95 courses. PHYS-BS had 2 of its 8 pathways,
  and the parser made BOTH of them required at once instead of "pick one".
- Even on fresh text, the parser can't model several real structures on
  these pages:
  - "Complete N additional courses from the lists above, the Track
    Electives below, or the General Electives" (CS-BS subplans). The parser
    only saw the one list printed directly under that heading, and often
    used the wrong N.
  - CS-BS's "General Electives" and "Optional Humanities Electives" are
    reference lists, not standalone requirements. The parser made each a
    required "pick 1".
  - Substitutions stated only in notes (EE-BS: MATH52 for MATH51, CME102
    for MATH53, CS109 for EE178).
  - PHYS-BS's open-ended pools ("any MATH course numbered 101+ (not 197)",
    "any PHYSICS 100+ / APPPHYS 200+ course except ..."). These are
    enumerated here from the course snapshot.

Modeling conventions (solver.py has no per-course "used once" exclusivity
within a single program, so these matter):
- An "N additional courses from pools X, Y, Z" requirement is ONE
  at_least_n(N) rule over the de-duplicated union of those pools. In the
  app, any pool with more than 8 options is turned into N generic
  placeholder slots (see placeholderize_large_elective_pools). Those slots
  never overlap with the real courses picked for earlier tracks, so the
  total course count comes out right.
- If an "additional" pool is small enough to stay as real courses AND
  overlaps an earlier track's pool, it's written as a cumulative count over
  both pools instead (e.g. Unspecialized Track B+C: 2 distinct courses from
  the combined 8-course list). That's exact for nested pools.
- A Science-elective item that would re-use a course the program already
  requires (e.g. PHYSICS41+42 when PHYSICS41 is CS-BS's required mechanics
  course) is left out of that elective pool. This is conservative: it can
  never produce a plan the bulletin wouldn't accept.
- "Up to one course from the Humanities Electives list" is not modeled
  (left out of the pools). Again conservative: it never allows an invalid
  plan, it just won't suggest that option.
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

from .models import RequirementRule

_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "courses_snapshot.json"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _and(*keys: str) -> tuple:
    return ("__and__", *keys)


def _keys_of(item) -> set[str]:
    if isinstance(item, tuple):
        return {k for k in item if k != "__and__"}
    return {item}


def _union(*pools, exclude=()) -> list:
    """De-duplicated union of course pools. A bare key that's also a member of
    an OR-tuple elsewhere in the union is dropped (otherwise one course would
    satisfy two "slots" of the same at_least_n). `exclude` removes keys the
    program already requires elsewhere."""
    exclude = set(exclude)
    items: list = []
    for pool in pools:
        for item in pool:
            if isinstance(item, tuple) and item and item[0] == "__and__":
                if not (_keys_of(item) & exclude):
                    items.append(item)
                continue
            if isinstance(item, tuple):
                kept = tuple(k for k in item if k not in exclude)
                if not kept:
                    continue
                items.append(kept if len(kept) > 1 else kept[0])
            elif item not in exclude:
                items.append(item)
    in_or_tuples = {
        k for it in items if isinstance(it, tuple) and it[0] != "__and__" for k in it
    }
    out, seen = [], set()
    for it in items:
        if isinstance(it, str) and it in in_or_tuples:
            continue
        sig = (it[0] == "__and__", frozenset(_keys_of(it))) if isinstance(it, tuple) else it
        if sig in seen:
            continue
        seen.add(sig)
        out.append(it)
    return out


def _without_overlap(pool, required: set[str]) -> list:
    """Drop every item (including whole AND-packages) touching `required`."""
    return [it for it in pool if not (_keys_of(it) & required)]


def _r(label, kind, courses=(), n=None, unit_min=None, sub_rules=(), source=""):
    return RequirementRule(
        label=label, kind=kind, courses=list(courses), n=n, unit_min=unit_min,
        sub_rules=list(sub_rules), source=source,
    )


@functools.lru_cache(maxsize=1)
def _snapshot_courses() -> dict:
    data = json.loads(_SNAPSHOT_PATH.read_text())
    courses = data["courses"]
    if isinstance(courses, dict):
        return courses
    return {f"{c['subject']}{c['code']}": c for c in courses}


_LEADING_NUM_RE = re.compile(r"^(\d+)")


def _numbered_pool(subject: str, min_num: int, *, exclude_nums=(), max_num=None, min_units=3.0) -> list[str]:
    """Every snapshot course in `subject` whose number is >= min_num (and
    < max_num if given), not in exclude_nums, and offered for >= min_units."""
    out = []
    for key, c in _snapshot_courses().items():
        subj = c.get("subject") if isinstance(c, dict) else getattr(c, "subject", None)
        if subj != subject:
            continue
        code = c.get("code") if isinstance(c, dict) else getattr(c, "code", "")
        m = _LEADING_NUM_RE.match(str(code))
        if not m:
            continue
        num = int(m.group(1))
        if num < min_num or num in exclude_nums or (max_num is not None and num >= max_num):
            continue
        units = c.get("units_max") if isinstance(c, dict) else getattr(c, "units_max", 0)
        if (units or 0) < min_units:
            continue
        out.append(key)
    return sorted(out)


# ---------------------------------------------------------------------------
# School of Engineering shared lists (ENGR-BS page, "School Requirements")
# ---------------------------------------------------------------------------

SOE_ENGINEERING_FUNDAMENTALS = [
    "ENGR10", "ENGR14", "ENGR15", "ENGR20", "ENGR21", "ENGR40M", "ENGR42",
    ("ENGR50", "ENGR50E", "ENGR50M"), "ENGR55", "ENGR60", "ENGR76", "EE65",
    ("MS&E111", "MS&E111DS"), ("CS106A", "CS106AX", "CS106B"),
    ("ENGR80", "BIOE80"), ("ENGR90", "CEE70"),
]

SOE_SCIENCE = [
    "BIO81", "BIO82", "BIO83", "BIO84", "BIO85", "BIO86", "BIO45", "BIO47", "BIO150",
    "CEE63", "CEE64", "CEE70",
    _and("CHEM31A", "CHEM31B"), "CHEM31E", "CHEM33", "CHEM121", "CHEM123",
    "EARTHSYS2", "EARTHSYS10", "EARTHSYS11", "EARTHSYS110",
    "HUMBIO2A", "HUMBIO3A", "HUMBIO4A",
    _and("PHYSICS21", "PHYSICS23", "PHYSICS25", "PHYSICS22", "PHYSICS24", "PHYSICS26"),
    _and("PHYSICS41", "PHYSICS42"), _and("PHYSICS41E", "PHYSICS42"),
    _and("PHYSICS43", "PHYSICS44"), _and("PHYSICS45", "PHYSICS46"),
    _and("PHYSICS61", "PHYSICS61L"), "PHYSICS71", "PHYSICS81",
]

SOE_TECHNOLOGY_IN_SOCIETY = [
    "AA252", "ARCHLGY151", "BIOE131", "BIOE177", "CEE102A", "CEE145E", "CLASSICS168",
    "COMM154", "COMM166", "CS120", "CS121", "CS139", "CS152", "CS182", "CS182W",
    "CS222", "CS256", "CS278", "CS329X", "DATASCI154", "EARTHSYS125", "ENGR145",
    "ENGR148", _and("ENERGY177A", "ENERGY177B"), "EPS194", "FEMGEN144", "FEMGEN147",
    "HUMBIO174", "MS&E179", "NBIO101", "POLISCI114S", "PUBLPOL114", "PUBLPOL134",
    "STS1", "STS115",
]

# CME100/CME102 are cross-listed as ENGR154/ENGR155A (stated on the EE-BS page).
_CME100 = ("CME100", "ENGR154")
_CME102 = ("CME102", "ENGR155A")


# ===========================================================================
# CS-BS
# ===========================================================================

_CS_SRC = "CS-BS bulletin 2026-27 (verified)"

CS_BS_GENERAL_ELECTIVES = [
    "CS112", "CS123", "CS124", "CS131", "CS137A", "CS139", "CS140E", "CS143", "CS144",
    "CS145", "CS146J", "CS146S", "CS147", "CS147L", "CS148", "CS149", "CS151", "CS154",
    "CS155", ("CS157", "PHIL151"), "CS166", "CS168", "CS171", "CS173A", "CS177", "CS195",
    "CS197", "CS197C", "CS205L", "CS206", "CS210A", "CS212", "CS217", "CS221", "CS222",
    "CS223A", "CS224G", "CS224N", "CS224R", "CS224S", "CS224V", "CS224W", "CS225A",
    "CS227B", "CS228", "CS229", "CS229B", "CS229M", "CS229S", "CS230", "CS231A", "CS231N",
    "CS232", "CS233", "CS234", "CS235", "CS237A", "CS237B", "CS238", "CS240", "CS240LX",
    "CS242", "CS243", "CS244C", "CS246", "CS247A", "CS247B", "CS247E", "CS247G", "CS247I",
    "CS247S", "CS248A", "CS248B", "CS249I", "CS250", "CS251", "CS254", "CS254B", "CS255",
    "CS256", "CS257", "CS258", "CS259Q", "CS261", "CS263", "CS264", "CS265", "CS269I",
    "CS270", "CS272", "CS273B", "CS273C", "CS274", "CS275", "CS278", "CS279", "CS281",
    "CS293", "CS323", "CS330", "CS336", "CS342", "CS348B", "CS348C", "CS348E", "CS348I",
    "CS348K", "CS348N", "CS355", "CS361", "CS377G", "CS448B", "CEE342", "CME108",
    "DATASCI112", "EE108", "EE180", "EE267", "EE282", "EE364A", "EE374", "ENGR245",
    "MS&E244", "MS&E265", "MS&E296",
]

_CS_PHYSICS_MECH = ("PHYSICS41", "PHYSICS21", "PHYSICS61", "PHYSICS41E")
_CS_PHYSICS_EM = ("PHYSICS43", "PHYSICS23", "PHYSICS81", "OSPPARIS53")
_CS_CAPSTONE = ["CS191", "CS191W", "CS194", "CS194W", "CS210B"]


def _cs_bs_mathematics(with_electives: bool = True) -> RequirementRule:
    required = _r(
        "Required Math Courses", "all",
        ["CS103", "CS109", "MATH19", "MATH20", "MATH21"], source=f"{_CS_SRC}: Mathematics",
    )
    if not with_electives:
        return _r("Mathematics (Computational Biology: 20 units minimum)", "all",
                  sub_rules=[required], source=f"{_CS_SRC}: Computational Biology Subplan")
    electives = _r(
        "Math Electives", "at_least_n",
        [_CME100, _CME102, "CME104", "CS157", "CS205L", "ENGR108", "MATH51", "MATH52",
         "MATH53", "MATH104", "MATH107", "MATH108", "MATH109", "MATH110", "MATH113", "PHIL151"],
        n=2, source=f"{_CS_SRC}: Mathematics",
    )
    return _r("Mathematics (26 units minimum)", "all", sub_rules=[required, electives],
              source=f"{_CS_SRC}: Mathematics")


def _cs_bs_science() -> RequirementRule:
    required_physics = set(_CS_PHYSICS_MECH) | set(_CS_PHYSICS_EM)
    return _r("Science (11 units minimum)", "all", sub_rules=[
        _r("Physics Courses", "all", [_CS_PHYSICS_MECH, _CS_PHYSICS_EM], source=f"{_CS_SRC}: Science"),
        _r("Science Elective (SoE Science list, 3+ units)", "at_least_n",
           _without_overlap(SOE_SCIENCE, required_physics), n=1, source=f"{_CS_SRC}: Science"),
    ], source=f"{_CS_SRC}: Science")


def _cs_bs_tis() -> RequirementRule:
    return _r("Technology in Society (SoE TiS list, 3+ units)", "at_least_n",
              SOE_TECHNOLOGY_IN_SOCIETY, n=1, source=f"{_CS_SRC}: Technology in Society")


def _cs_bs_engineering_fundamentals() -> RequirementRule:
    return _r("Engineering Fundamentals (10 units minimum)", "all",
              ["CS106B", ("ENGR40M", "ENGR76")], source=f"{_CS_SRC}: Engineering Fundamentals")


def _cs_bs_common_tail() -> list[RequirementRule]:
    return [
        _r("Core Program Requirements", "all", [("CS107", "CS107E"), "CS111", "CS161"],
           source=f"{_CS_SRC}: Core Program Requirements"),
        _r("Writing in the Major (WIM)", "at_least_n",
           ["CS121", "CS182W", "CS191W", "CS194W", "CS210B"], n=1,
           source=f"{_CS_SRC}: Writing in the Major"),
        _r("Senior Project (3 units)", "unit_min", _CS_CAPSTONE, unit_min=3.0,
           source=f"{_CS_SRC}: Capstone"),
    ]


# --- subplans ---------------------------------------------------------------

_AI_AREAS = {
    "Area I: AI Methods": ["CS224R", "CS228", "CS229", "CS229M", "CS234", "CS238"],
    "Area II: Natural Language Processing": ["CS124", "CS224N", "CS224S", "CS224V"],
    "Area III: Vision": ["CS131", "CS231A", "CS231N"],
    "Area IV: Robotics": ["CS123", "CS137A", "CS223A", "CS237A"],
}
_AI_TRACK_C_LISTS = [
    "CS157", "CS205L", "CS230", "CS236", "CS257", "STATS315A", "STATS315B",  # AI Methods
    "CS235", "CS279",                                                        # Comp Bio
    "CS224W",                                                                # Information and the Web
    "CS256",                                                                 # Ethics
    "CS225A", "CS327A", "ENGR205",                                           # Robotics and Control
    "CS229S",                                                                # Systems
    "CS222",                                                                 # Simulation
    "CS151", "CS227B",                                                       # Other
]
_AI_TRACK_ELECTIVES = [
    "APPPHYS293", "CS325B", "CS326", "CS342", "EE263", "EE278", "EE364A", "EE364B",
    "MS&E252", "MS&E355", "PHIL152", "PSYCH204A", "PSYCH204B", "PSYCH209", "STATS200",
    "STATS202", "STATS203", "STATS205", "STATS220", "STATS271",
]


def _cs_ai_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Artificial Intelligence Subplan"
    track_b_pool = [c for courses in _AI_AREAS.values() for c in courses]
    return _r("Artificial Intelligence Subplan", "all", sub_rules=[
        _r("Track A Requirement", "all", ["CS221"], source=src),
        _r("Track B Requirement (2 courses, each from a different area)", "at_least_n", n=2,
           sub_rules=[_r(name, "at_least_n", courses, n=1, source=src) for name, courses in _AI_AREAS.items()],
           source=src),
        _r("Track C Requirement (1 more from the Track B lists or the Track C lists)", "at_least_n",
           _union(track_b_pool, _AI_TRACK_C_LISTS), n=1, source=src),
        _r("Track Electives (3 more from the lists above, AI Track Electives, or CS-BS General Electives)",
           "at_least_n",
           _union(track_b_pool, _AI_TRACK_C_LISTS, _AI_TRACK_ELECTIVES, CS_BS_GENERAL_ELECTIVES,
                  exclude={"CS221"}),
           n=3, source=src),
    ], source=src)


_CB_TRACK_D = ["BMDS210", "BMDS214", "BMDS215", "BMDS217", "BMDS219", "BMDS222", "BMDS260",
               "CS273B", "CS279", "IMMUNOL207"]
_CB_TRACK_E = ["CS124", "CS131", "CS224N", "CS224R", "CS224V", "CS224W", "CS228", "CS229",
               "CS229S", "CS231N", "CS234", "CS238"]
_CB_TRACK_F_LIST = [
    "APPPHYS294", "BIO183", "BIO187", "BIOC241", "BMDS210", "BMDS214", "BMDS215", "BMDS217",
    "BMDS219", "BMDS222", "BMDS260", "CHEMENG150", "CS124", "CS131", "CS142", "CS147",
    "CS147L", "CS148", "CS154", "CS166", "CS168", "CS224N", "CS224R", "CS224V", "CS224W",
    "CS228", "CS229", "CS229S", "CS230", "CS231N", "CS234", "CS238", "CS248A", "CS273B",
    "CS279", "DATASCI112", "EE263", "EE364A", "IMMUNOL207", "MS&E152", "MS&E252", "STATS141",
    "STATS202", "STATS203", "STATS205", "STATS206", "STATS211", "STATS315A", "STATS315B",
]
_CB_TRACK_G_LIST = [
    "APPPHYS294", "BIO81", "BIO82", "BIO83", "BIO84", "BIO85", "BIO86", "BIO214", "BIO230",
    "BIOC241", "BIOE220", "BMDS272", "CHEM31A", "CHEM31B", "CHEM31E", "CHEM33", "CHEM141",
    "CHEM143", "CHEM171", "CHEM181", "CHEMENG150", "CHEMENG174", "DBIO210", "GENE211",
    "HUMBIO151R", "ME281",
]


def _cs_compbio_replacements() -> list[RequirementRule]:
    """Replaces the standard Mathematics / Science / Engineering Fundamentals
    rules (the page says Comp Bio's are non-standard and needs no math
    electives)."""
    src = f"{_CS_SRC}: Computational Biology Subplan"
    return [
        _cs_bs_mathematics(with_electives=False),
        _r("Biology or Humbio Sequence (Computational Biology science)", "at_least_n", n=1, sub_rules=[
            _r("Biology Sequence", "at_least_n", ["BIO82", "BIO83", "BIO85", "BIO86"], n=3, source=src),
            _r("Humbio Sequence", "all", ["HUMBIO2A", "HUMBIO3A", "HUMBIO4A"], source=src),
        ], source=src),
        _r("Engineering Fundamentals (Computational Biology)", "all", sub_rules=[
            _r("CS106B", "all", ["CS106B"], source=src),
            _r("SoE Engineering Fundamentals Elective (not another CS106)", "at_least_n",
               _without_overlap(SOE_ENGINEERING_FUNDAMENTALS, {"CS106A", "CS106AX", "CS106B"}),
               n=1, source=src),
        ], source=src),
    ]


def _cs_compbio_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Computational Biology Subplan"
    return _r("Computational Biology Depth (8 courses, 25 units)", "all", sub_rules=[
        # CS173A was previously CS273A -- either counts.
        _r("Track A Requirement", "all", [("CS173A", "CS273A"), "CS221"], source=src),
        # The live page shows "CLASSICS145: Narratives of Enslavement" here --
        # a bad course link on the bulletin; the intended course is CS145
        # (confirmed by the user, Nov 2026).
        _r("Track B Requirement", "at_least_n", ["CS145", "CS246"], n=1, source=src),
        _r("Track C Requirement", "at_least_n", ["CS146J", "CS147L", "CS448B", "DATASCI112"], n=1, source=src),
        _r("Track D Requirement", "at_least_n", _CB_TRACK_D, n=1, source=src),
        _r("Track E Requirement", "at_least_n", _CB_TRACK_E, n=1, source=src),
        _r("Track F Requirement (1 more from Track D/E lists or the Track F list)", "at_least_n",
           _union(_CB_TRACK_D, _CB_TRACK_E, _CB_TRACK_F_LIST), n=1, source=src),
        _r("Track G Requirement (1 more from the Track D list or the Track G list)", "at_least_n",
           _union(_CB_TRACK_D, _CB_TRACK_G_LIST), n=1, source=src),
    ], source=src)


def _cs_compe_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Computer Engineering Subplan"
    return _r("Computer Engineering Subplan", "all", sub_rules=[
        _r("Track A Requirement", "all", ["EE108", "EE180"], source=src),
        _r("Track B Requirement", "at_least_n", ["EE101A", "EE101B", "EE102A", "EE102B"], n=2, source=src),
        _r("Track C Requirement: Pick a Concentration", "at_least_n", n=1, sub_rules=[
            _r("A. Digital Systems Concentration", "all", sub_rules=[
                _r("Required Courses", "all", ["EE109", "EE271"], source=src),
                _r("Required Elective Options", "at_least_n", ["CS112", "CS140E"], n=1, source=src),
            ], source=src),
            _r("B. Robotics and Mechatronics Concentration", "all",
               ["CS205L", "CS223A", ("ME210", "CS225A")], source=src),
            _r("C. Networking Concentration", "all", sub_rules=[
                _r("Required Courses", "all", [("CS112", "CS140E"), "CS144"], source=src),
                _r("Required Elective Options", "at_least_n",
                   ["CS240", "CS240LX", "CS241", "CS244C", "EE179"], n=1, source=src),
            ], source=src),
        ], source=src),
    ], source=src)


_HCI_TRACK_C = ["CS177", "CS278", "CS448B"]
_HCI_TRACK_ELECTIVE_LIST = [
    "CS194H", "CS206", "CS210A", "CS222", "CS224C", "CS247A", "CS247B", "CS247E", "CS247G",
    "CS247I", "CS247S", "CS329X", "CS377E", "CS377G", "CS377J", "CS377P", "CS377Q", "CS377U",
]


def _cs_hci_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Human-Computer Interaction Subplan"
    return _r("Human-Computer Interaction Subplan", "all", sub_rules=[
        _r("Track A Requirement", "all",
           ["CS147", ("CS247A", "CS247B", "CS247E", "CS247G", "CS247I", "CS247S"), "CS347"], source=src),
        _r("Track B Requirement", "at_least_n", ["CS146J", "CS147L"], n=1, source=src),
        _r("Track C Requirement: HCI Mezzanine", "at_least_n", _HCI_TRACK_C, n=1, source=src),
        _r("Track Elective (1 more from the Track C list or the Track Elective list)", "at_least_n",
           _union(_HCI_TRACK_C, _HCI_TRACK_ELECTIVE_LIST), n=1, source=src),
        _r("Additional Elective (1 more from Track C, Track Elective, or CS-BS General Electives)",
           "at_least_n",
           _union(_HCI_TRACK_C, _HCI_TRACK_ELECTIVE_LIST, CS_BS_GENERAL_ELECTIVES, exclude={"CS147", "CS347"}),
           n=1, source=src),
    ], source=src)


_INFO_AREAS = {
    "Area I: Information-based AI Applications": ["CS224N", "CS224S", "CS229", "CS233", "CS234"],
    "Area II: Database and Information Systems": ["CS112", "CS140E", "CS147L", "CS151", "CS246"],
    "Area III: Information Systems in Biology": ["CS235", "CS270", "CS274"],
    "Area IV: Information Systems on the Web": ["CS224W"],
}


def _cs_info_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Information Subplan"
    track_b_pool = [c for courses in _INFO_AREAS.values() for c in courses]
    return _r("Information Subplan", "all", sub_rules=[
        _r("Track A Requirement", "all", ["CS124", "CS145"], source=src),
        _r("Track B Requirement (2 courses from different areas)", "at_least_n", n=2,
           sub_rules=[_r(name, "at_least_n", courses, n=1, source=src) for name, courses in _INFO_AREAS.items()],
           source=src),
        _r("Electives (3 more from the Track B lists or CS-BS General Electives)", "at_least_n",
           _union(track_b_pool, CS_BS_GENERAL_ELECTIVES, exclude={"CS124", "CS145"}), n=3, source=src),
    ], source=src)


_SYS_TRACK_C = ["CS144", "CS145", "CS149", "CS155", "CS217", ("CS240", "CS240LX"), "CS242",
                "CS243", "CS244C", "EE271", "EE282"]
_SYS_SUBPLAN_ELECTIVES = ["CS241", "CS295", "CS340R", "CS343D", "CS349D", "CS349H", "CS448I",
                          "EE108", "EE382A", "EE382C", "EE384S"]


def _cs_systems_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Systems Subplan"
    return _r("Systems Subplan", "all", sub_rules=[
        _r("Track A Requirement", "at_least_n", ["CS112", "CS140E"], n=1, source=src),
        _r("Track B Requirement", "at_least_n", ["CS143", "EE180"], n=1, source=src),
        _r("Track C Requirement", "at_least_n", _SYS_TRACK_C, n=2, source=src),
        _r("Track D: Subplan Electives (3 more from Track C, Systems Subplan Electives, or CS-BS General Electives)",
           "at_least_n", _union(_SYS_TRACK_C, _SYS_SUBPLAN_ELECTIVES, CS_BS_GENERAL_ELECTIVES), n=3, source=src),
    ], source=src)


_THEORY_TRACK_B = ["CS168", "CS255", "CS261", "CS265"]
_THEORY_TRACK_C_LIST = [
    "CS143", "CS151", "CS155", ("CS157", "PHIL151"), "CS166", "CS205L", "CS228", "CS233",
    "CS235", "CS236", "CS242", "CS250", "CS251", "CS254", "CS259Q", "CS263", "CS264",
    "CS269I", "CS355",
]


def _cs_theory_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Theory Subplan"
    return _r("Theory Subplan", "all", sub_rules=[
        _r("Track A Requirement", "all", ["CS154"], source=src),
        _r("Track B Requirement", "at_least_n", _THEORY_TRACK_B, n=1, source=src),
        _r("Track C Requirement (2 more from the Track B list or the Track C list)", "at_least_n",
           _union(_THEORY_TRACK_B, _THEORY_TRACK_C_LIST), n=2, source=src),
        _r("Depth Electives (3 more from Track B/C lists, Theory Subplan Electives, or CS-BS General Electives)",
           "at_least_n",
           _union(_THEORY_TRACK_B, _THEORY_TRACK_C_LIST, ["CME302", "CS254B", "PHIL152"],
                  CS_BS_GENERAL_ELECTIVES, exclude={"CS154"}),
           n=3, source=src),
    ], source=src)


_UNSPEC_TRACK_B = ["CS112", "CS140E", "CS143"]
_UNSPEC_TRACK_C_LIST = ["CS144", "CS155", "CS242", "CS244C", "EE180"]


def _cs_unspecialized_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Unspecialized Subplan"
    return _r("Unspecialized Subplan", "all", sub_rules=[
        _r("Track A Requirement", "all", ["CS154"], source=src),
        _r("Track B Requirement", "at_least_n", _UNSPEC_TRACK_B, n=1, source=src),
        # Track C = "one additional course from the list above, or one of the
        # following". The combined pool is only 8 courses, so the app leaves
        # it as real courses. It's written as a cumulative count over Track
        # B + C so the Track B course can't also count as the Track C one.
        _r("Track B + Track C combined (2 distinct courses from these two lists)", "at_least_n",
           _union(_UNSPEC_TRACK_B, _UNSPEC_TRACK_C_LIST), n=2, source=src),
        _r("Track D Requirement", "at_least_n",
           ["CS221", "CS223A", "CS228", "CS229", "CS231A", "CS238"], n=1, source=src),
        _r("Track E Requirement", "at_least_n",
           ["CS145", "CS147", "CS148", "CS168", "CS173A", "CS248A", "CS248B", "CS279"], n=1, source=src),
        _r("Depth Electives (2 more from CS-BS General Electives)", "at_least_n",
           _union(CS_BS_GENERAL_ELECTIVES, exclude={"CS154"}), n=2, source=src),
    ], source=src)


_VC_DEPTH = ["CS205L", "CS223A", "CS224R", "CS225A", "CS231A", "CS231N", "CS233", "CS248A",
             "CS248B", "CS348C", "CS348I", "CS348K", "CS448I", "EE267"]
_VC_SUBPLAN_ELECTIVES = ["CS123", "CS131", "CS148", "CS149", "CS221", "CS224N", "CS229", "CS230",
                         "CS234", "CS236", "CS448B", "CS448Z", "EE261"]


def _cs_visual_computing_subplan() -> RequirementRule:
    src = f"{_CS_SRC}: Visual Computing Subplan"
    return _r("Visual Computing Subplan", "all", sub_rules=[
        _r("Visual Computing Core", "at_least_n", ["CS225A", "CS248A", "CS248B", "CS231N"], n=2, source=src),
        _r("Visual Computing Depth (3 more)", "at_least_n", _VC_DEPTH, n=3, source=src),
        _r("Track Electives (3 more from VC Depth, VC Subplan Electives, or CS-BS General Electives)",
           "at_least_n", _union(_VC_DEPTH, _VC_SUBPLAN_ELECTIVES, CS_BS_GENERAL_ELECTIVES), n=3, source=src),
    ], source=src)


def _cs_individually_designed_subplan(placeholders: dict[str, str]) -> RequirementRule:
    keys = []
    for i in range(1, 8):
        key = f"PLACEHOLDER:CS_BS_INDIVIDUALLY_DESIGNED_TRACK_{i}"
        placeholders[key] = (
            f"Individually Designed Track course {i} of 7 (needs an approved proposal; "
            "7+ courses / 25+ units, at least 4 CS courses numbered 100+)"
        )
        keys.append(key)
    return _r("Individually Designed Subplan", "all", keys, source=f"{_CS_SRC}: Individually Designed Subplan")


_CS_SUBPLAN_BUILDERS = {
    "Artificial Intelligence Subplan": lambda ph: _cs_ai_subplan(),
    "Computational Biology Subplan": lambda ph: _cs_compbio_subplan(),
    "Computer Engineering Subplan": lambda ph: _cs_compe_subplan(),
    "Human-Computer Interaction Subplan": lambda ph: _cs_hci_subplan(),
    "Individually Designed Subplan": _cs_individually_designed_subplan,
    "Information Subplan": lambda ph: _cs_info_subplan(),
    "Systems Subplan": lambda ph: _cs_systems_subplan(),
    "Theory Subplan": lambda ph: _cs_theory_subplan(),
    "Unspecialized Subplan": lambda ph: _cs_unspecialized_subplan(),
    "Visual Computing Subplan": lambda ph: _cs_visual_computing_subplan(),
}


def cs_bs_verified_rules(include_subplans: list[str] | None = None) -> tuple[list[RequirementRule], dict[str, str]]:
    placeholders: dict[str, str] = {}
    subplans = list(include_subplans or [])
    unknown = [s for s in subplans if s not in _CS_SUBPLAN_BUILDERS]
    if unknown:
        raise KeyError(f"CS-BS: unknown subplan(s) {unknown}; known: {sorted(_CS_SUBPLAN_BUILDERS)}")
    if "Computational Biology Subplan" in subplans:
        head = _cs_compbio_replacements() + [_cs_bs_tis()]
    else:
        head = [_cs_bs_mathematics(), _cs_bs_science(), _cs_bs_tis(), _cs_bs_engineering_fundamentals()]
    rules = head + _cs_bs_common_tail()
    for name in subplans:
        rules.append(_CS_SUBPLAN_BUILDERS[name](placeholders))
    return rules, placeholders


# ===========================================================================
# EE-BS
# ===========================================================================

_EE_SRC = "EE-BS bulletin 2026-27 (verified)"
EE_BS_ELECTIVES_PLACEHOLDER = "PLACEHOLDER:EE_BS_ELECTIVES_ANY_EE_COURSE"

EE_BS_ELECTIVES = [
    "EE101B", "EE102B", "EE104", "EE109", "EE114", "EE116", "EE117", "EE118", "EE124", "EE133",
    "EE134", "EE142", "EE153", "EE160A", "EE156", "EE165", "EE168", "EE179", "EE180", "EE184",
    "EE212", "EE214B", "EE216", "EE222", "EE223", "EE228", "EE235", "EE236A", "EE236B", "EE242",
    "EE247", "EE259", "EE260A", "EE260B", "EE261", "EE263", "EE264", "EE267", "EE269", "EE271",
    "EE272", "EE273", "EE274", "EE276", "EE278", "EE279", "EE282", "EE285",
    "EE372",  # listed on the page as EE372D; the snapshot's key for that course is EE372
    "BIOE131", "CEE107A", "CEE155", "CEE176A", "CEE176B", "CS107", "CS107E", "CS108", "CS111",
    "CS112", "CS131", "CS140", "CS143", "CS144", "CS145", "CS148", "CS149", "CS155", "CS161",
    "CS221", "CS223A", "CS224N", "CS225A", "CS229", "CS231A", "CS231N", "CS241", "CS244",
    "ENGR105", "ENGR205", "ENERGY201B", "MATSCI142", "MATSCI152", "MATSCI156", "MATSCI164",
    "ME220", "MED275B", "MUSIC250A", "MUSIC253", "MUSIC254", "MUSIC256A", "MUSIC257", "MUSIC356",
    "MUSIC422", "RAD206", "RAD220",
]


def ee_bs_verified_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    src = _EE_SRC
    # "CME 100 and CME 102 can be substituted for MATH 51 and MATH 53. MATH 52
    # can be substituted for MATH 51." / "MATH53 may be replaced by CME102."
    # Kept to <= 8 options on purpose: a bigger pool gets collapsed into ONE
    # generic placeholder by placeholderize_large_elective_pools, which would
    # count a 2-course sequence as a single course.
    sequence = [
        _and(a, b) for a in ("MATH51", "MATH52") for b in ("MATH53", "CME102")
    ] + [_and("CME100", "CME102"), _and("ENGR154", "ENGR155A")]
    physics_used = {"PHYSICS41", "PHYSICS61", "PHYSICS71", "EE65", "PHYSICS43", "PHYSICS81"}
    rules = [
        _r("Mathematics", "all", sub_rules=[
            _r("Calculus (AP credit accepted)", "all", ["MATH19", "MATH20", "MATH21"], source=src),
            _r("Mathematics Sequence", "at_least_n", sequence, n=1, source=src),
            _r("EE Math", "at_least_n", ["CS103", "ENGR108", "MATH113"], n=1, source=src),
            # "Students may use CS109 in place of EE178."
            _r("Statistics/Probability", "all", [("EE178", "CS109")], source=src),
        ], source=src),
        _r("Science (3 courses, 12 units minimum)", "all", sub_rules=[
            _r("Physics", "at_least_n", [
                _and("PHYSICS41", "EE65"), _and("PHYSICS61", "EE65"),
                _and("PHYSICS41", "PHYSICS71"), _and("PHYSICS61", "PHYSICS71"),
            ], n=1, source=src),
            # PHYSICS43/81 items are left out because they double as EE Core
            # options (conservative).
            _r("Science Elective (SoE Science list)", "at_least_n",
               _without_overlap(SOE_SCIENCE, physics_used), n=1, source=src),
        ], source=src),
        _r("Technology in Society (SoE TiS list)", "at_least_n", SOE_TECHNOLOGY_IN_SOCIETY, n=1, source=src),
        _r("Engineering Fundamentals", "all", sub_rules=[
            _r("Engineering Fundamentals Required Course", "all", ["CS106B"], source=src),
            # One more from the SoE list, not another CS106 and not EE65 (which
            # the Physics requirement may already be using).
            _r("Engineering Fundamentals Elective (SoE list; ENGR40M or ENGR76 recommended)", "at_least_n",
               _without_overlap(SOE_ENGINEERING_FUNDAMENTALS, {"CS106A", "CS106AX", "CS106B", "EE65"}),
               n=1, source=src),
        ], source=src),
        _r("Core Electrical Engineering Courses", "all",
           [("EE42", "EE142", "PHYSICS43", "PHYSICS81"), "EE100", "EE101A", "EE102A", "EE108"], source=src),
        _r("Required Course (one, 4+ units)", "at_least_n",
           ["EE101B", "EE102B", "EE180", "CS107", "CS107E"], n=1, source=src),
        _r("Electives", "unit_min", EE_BS_ELECTIVES + [EE_BS_ELECTIVES_PLACEHOLDER], unit_min=24.0, source=src),
        _r("WIM", "at_least_n",
           ["CS194W", "EE109", "EE133", "EE134", "EE153", "EE191W", "EE264W", "EE267W"], n=1, source=src),
        _r("Design/Capstone", "at_least_n",
           ["CS194W", "EE109", "EE133", "EE134", "EE153", "EE264", "EE264W", "EE267", "EE267W"],
           n=1, source=src),
    ]
    placeholders = {
        EE_BS_ELECTIVES_PLACEHOLDER: (
            "Any additional letter-graded lecture/lab EE course (the bulletin allows any such "
            "course, plus up to two extra Engineering Fundamentals, beyond the listed electives)"
        ),
    }
    return rules, placeholders


# ===========================================================================
# PHYS-BS
# ===========================================================================

_PH_SRC = "PHYS-BS bulletin 2026-27 (verified)"
_PRACTICUM = ["PHYSICS100", "PHYSICS104", "PHYSICS106", "PHYSICS108", "PHYSICS113", "PHYSICS166"]
_PROBABILITY = ("CS109", "STATS117", "STATS118")

_PHYS_CORE_REQUIRED = {
    "PHYSICS41", "PHYSICS41E", "PHYSICS43", "PHYSICS61", "PHYSICS61L", "PHYSICS71",
    "PHYSICS79L", "PHYSICS89L", "PHYSICS81", "PHYSICS81L", "PHYSICS71L",
    "PHYSICS120", "PHYSICS130", "PHYSICS170", "PHYSICS191",
}


def _math_101_plus_pool() -> list[str]:
    """"Any MATH course numbered 101 and above (not 197), of 3 units or more.\""""
    return _numbered_pool("MATH", 101, exclude_nums={197})


def _phys_elective_pool() -> list[str]:
    """"One Physics (PHYSICS) or Applied Physics (APPPHYS) course of 3 or more
    units, numbered 100 and above for Physics courses or 200 and above for
    Applied Physics courses (not including PHYSICS 190, 198, 199, 201, 205,
    240, 241, 290, 291, 293, 294, APPPHYS 290, APPPHYS 291, APPPHYS 390, or any
    PHYSICS or APPPHYS course numbered 400 or higher).\""""
    phys = _numbered_pool("PHYSICS", 100, max_num=400,
                          exclude_nums={190, 198, 199, 201, 205, 240, 241, 290, 291, 293, 294})
    app = _numbered_pool("APPPHYS", 200, max_num=400, exclude_nums={290, 291, 390})
    return [k for k in phys + app if k not in _PHYS_CORE_REQUIRED]


def _phys_pathways() -> dict[str, RequirementRule]:
    src = f"{_PH_SRC}: Depth in Discipline"
    practicum = lambda extra=(): _r("Practicum Course", "at_least_n", list(extra) + _PRACTICUM, n=1, source=src)
    math_pool = _math_101_plus_pool()
    phys_pool = _phys_elective_pool()
    return {
        "Core Pathway": _r("CORE PATHWAY (23 units)", "all", sub_rules=[
            _r("Core Pathway Required Courses", "all",
               ["PHYSICS110", "PHYSICS121", "PHYSICS131", "PHYSICS171"], source=src),
            practicum(),
            _r("Physics or Math Elective", "at_least_n",
               _union([_PROBABILITY, "EE261", "PHYSICS112"], math_pool, phys_pool,
                      exclude={"PHYSICS110", "PHYSICS121", "PHYSICS131", "PHYSICS171"}),
               n=1, source=src),
        ], source=src),
        "Astrophysics Pathway": _r("ASTROPHYSICS PATHWAY (19 units)", "all", sub_rules=[
            _r("Astrophysics Pathway Required Courses", "all",
               ["PHYSICS100", "PHYSICS160", "PHYSICS161"], source=src),
            _r("Additional Courses (1)", "at_least_n",
               ["PHYSICS113", "PHYSICS166", ("PHYSICS267", "STATS200"), _PROBABILITY], n=1, source=src),
            _r("Additional Courses (2) (EPS119 does not count)", "at_least_n",
               ["EPS219", "PHYSICS110", "PHYSICS121", "PHYSICS131", "PHYSICS171", "PHYSICS262"],
               n=2, source=src),
        ], source=src),
        "Biophysics Pathway": _r("BIOPHYSICS PATHWAY (21 units)", "all", sub_rules=[
            _r("Biophysics Pathway Required Courses", "all",
               [("PHYSICS110", "PHYSICS131"), "PHYSICS171"], source=src),
            _r("Additional Courses", "at_least_n", [
                ("APPPHYS205", "BIO126"), ("APPPHYS223", "BIO223", "BIOE213"),
                ("APPPHYS237", "BIO251"), ("APPPHYS293", "PSYCH242"), ("APPPHYS294", "BIO294"),
                "BIOE101", "BIOE102",
            ], n=3, source=src),
            practicum(["APPPHYS232"]),
        ], source=src),
        "Computational Physics & Data Science Pathway": _r(
            "COMPUTATIONAL PHYSICS & DATA SCIENCE PATHWAY (20 units)", "all", sub_rules=[
                _r("Computational Physics & Data Science Pathway Required Courses", "all",
                   [("PHYSICS110", "PHYSICS131", "PHYSICS171"), "PHYSICS113"], source=src),
                _r("Additional Courses", "at_least_n", [
                    _PROBABILITY, "CS129", "CS154", "CS161", "CS205L", "CS221", "CS229", "CS230",
                    "CS259Q", "PHYSICS166", "STATS200", ("STATS203", "STATS203V"), ("STATS270", "STATS271"),
                ], n=3, source=src),
                _r("Physics Elective (PHYSICS 100+ / APPPHYS 200+, 3+ units, with listed exclusions)",
                   "at_least_n", _union(phys_pool, exclude={"PHYSICS113"}), n=1, source=src),
            ], source=src),
        "Geophysics Pathway": _r("GEOPHYSICS PATHWAY (21 units)", "all", sub_rules=[
            _r("Geophysics Pathway Required Courses", "all",
               ["PHYSICS110", ("PHYSICS121", "PHYSICS131", "PHYSICS171")], source=src),
            _r("Additional Courses", "at_least_n", [
                "EARTHSYS146A", "EARTHSYS146B", "EARTHSYS164", "ESS288", "ESS363F", "GEOPHYS110",
                "GEOPHYS120", "GEOPHYS128", "GEOPHYS130", ("GEOPHYS148", "GEOPHYS248"), "GEOPHYS162",
                "GEOPHYS165", "GEOPHYS184", "GEOPHYS288",
            ], n=3, source=src),
            practicum(),
        ], source=src),
        "Mathematical Physics Pathway": _r("MATHEMATICAL PHYSICS PATHWAY (20 units)", "all", sub_rules=[
            _r("Mathematical Physics Pathway Required Course", "all", ["PHYSICS110"], source=src),
            practicum(),
            _r("Math Electives", "at_least_n",
               _union([_PROBABILITY, "EE261", "PHYSICS112"], math_pool), n=2, source=src),
            _r("Two Additional Electives (MATH 101+ or PHYSICS/APPPHYS electives)", "at_least_n",
               _union([_PROBABILITY, "EE261", "PHYSICS112"], math_pool, phys_pool, exclude={"PHYSICS110"}),
               n=2, source=src),
        ], source=src),
        "Physics Education Pathway": _r("PHYSICS EDUCATION PATHWAY (21 units)", "all", sub_rules=[
            _r("Physics Education Pathway Required Courses", "all",
               ["PHYSICS110", ("PHYSICS121", "PHYSICS131", "PHYSICS171")], source=src),
            _r("Additional Courses", "at_least_n", [
                "EDUC101", "EDUC218", ("PHYSICS295", "EDUC280"), "EDUC391", "EDUC398",
                "EDUC400A", "EDUC486",
            ], n=3, source=src),
            practicum(),
        ], source=src),
        "Quantum Science & Information Pathway": _r(
            "QUANTUM SCIENCE & INFORMATION PATHWAY (22 units)", "all", sub_rules=[
                _r("Quantum Science & Information Pathway Required Courses", "all",
                   [("PHYSICS110", "PHYSICS121", "PHYSICS171"), "PHYSICS131", "PHYSICS134"], source=src),
                _r("Additional Courses", "at_least_n", [
                    "APPPHYS203", "APPPHYS204", "APPPHYS228", "CS154", "CS259Q", "EE224", "EE276",
                    "STATS200", "STATS203",
                ], n=2, source=src),
                practicum(),
            ], source=src),
    }


PHYS_BS_PATHWAYS = [
    "Core Pathway", "Astrophysics Pathway", "Biophysics Pathway",
    "Computational Physics & Data Science Pathway", "Geophysics Pathway",
    "Mathematical Physics Pathway", "Physics Education Pathway",
    "Quantum Science & Information Pathway",
]


def phys_bs_verified_rules(pathway: str | None = None) -> tuple[list[RequirementRule], dict[str, str]]:
    """`pathway=None` leaves the pathway choice to the solver (pick exactly
    one); otherwise only that pathway's requirements are included. The
    Individually Designed Pathway is proposal-only and isn't modeled."""
    src = _PH_SRC
    pathways = _phys_pathways()
    if pathway is not None and pathway not in pathways:
        raise KeyError(f"PHYS-BS: unknown pathway {pathway!r}; known: {sorted(pathways)}")
    depth = (
        pathways[pathway] if pathway is not None
        else _r("Depth in Discipline: one Pathway (6 courses)", "at_least_n", n=1,
                sub_rules=list(pathways.values()), source=f"{src}: Depth in Discipline")
    )
    rules = [
        _r("Introductory Physics Courses (23 units)", "all", [
            ("PHYSICS41", "PHYSICS41E"), "PHYSICS43", "PHYSICS61", "PHYSICS61L", "PHYSICS71",
            ("PHYSICS79L", "PHYSICS89L"), "PHYSICS81", ("PHYSICS81L", "PHYSICS71L"),
        ], source=f"{src}: Core Program Requirements"),
        _r("Introductory Math Courses (18-19 units)", "all", sub_rules=[
            _r("Math Series (MATH 50 or 60CM series)", "at_least_n", n=1, sub_rules=[
                _r("Math 50 Series", "all", ["MATH51", "MATH52", "MATH53"], source=src),
                _r("Math 60CM Series", "all", ["MATH61CM", "MATH62CM", "MATH63CM"], source=src),
            ], source=src),
            _r("Additional Math Course", "at_least_n",
               ["MATH131P", "MATH173", "MATH220A", "PHYSICS111"], n=1, source=src),
        ], source=f"{src}: Core Program Requirements"),
        _r("Intermediate Physics Courses (12 units)", "all",
           ["PHYSICS120", "PHYSICS130", "PHYSICS170"], source=f"{src}: Core Program Requirements"),
        depth,
        _r("WIM Requirement", "all", ["PHYSICS191"], source=f"{src}: Writing in the Major"),
        _r("Capstone (one of three options)", "at_least_n", n=1, sub_rules=[
            _r("1. Senior Thesis (3+ units)", "unit_min", ["PHYSICS205"], unit_min=3.0, source=src),
            # Needs a DUS-approved research project done outside coursework;
            # the paper can be written in PHYSICS191 (the WIM course) or PHYSICS192.
            _r("2. Research Project + Capstone Paper (research done outside coursework)", "at_least_n",
               ["PHYSICS191", "PHYSICS192"], n=1, source=src),
            # The page explicitly allows these to double-count with a pathway.
            _r("3. Course Option", "at_least_n", ["PHYSICS100", "PHYSICS108", "PHYSICS171"], n=1, source=src),
        ], source=f"{src}: Capstone"),
    ]
    return rules, {}
