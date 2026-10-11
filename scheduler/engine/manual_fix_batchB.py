"""
Hand-built requirement trees for four programs that come back INFEASIBLE
under the generic bulletin_parser.py pass, for reasons that are real
bulletin_parser.py/requirements.py logic gaps -- not course-offering gaps.
See engine/requirements.py's build_cs_bs_core_requirements() for the style
this mirrors: read the raw samples/{CODE}.txt text by hand, reason out the
*correct* requirement tree, and write it directly, bypassing the buggy
generic parse for just that one program (rather than risk a wider parser
rewrite for the sake of a handful of programs).

Each function below returns (rules, placeholders) exactly like
requirements.build_program_from_sample() -- pass `placeholders` to
course_data.add_placeholder_courses() before solving, same as any other
parsed program.

Programs covered here, and what was actually wrong (verified directly
against samples/{CODE}.txt and against engine/bulletin_parser.py's own
parse output -- see the ###　comments inline for the specific evidence):

  EE-BS     -- "Electives" is a genuine open-ended pool ("any letter-graded
               lecture/lab EE course"), but the generic parser only ever
               captures the 4 courses the bulletin happens to name as
               *examples* (EE101B, EE102B, EE104, EE109). Those 4 courses
               cap out at 12-16 real, schedulable units (EE104 has zero
               terms_offered in the snapshot), well short of the stated
               24-unit Electives minimum -- so the unit_min is
               mathematically unreachable no matter how the solver
               schedules things. Fixed by adding one placeholder course
               (see models.make_placeholder_course) standing in for "one
               more of the many real EE electives," repeatable up to the
               solver's existing repeat cap.

  INTLR-BA  -- A connector-confusion cascade in "Core Program
               Requirements": the raw text interleaves each of the 6
               category headers with inline "OR" course alternatives
               (e.g. "INTNLREL110C ... OR POLISCI110X ..."), and
               bulletin_parser.py's section-header/connector detector
               desyncs at the first inline OR, silently shifting every
               later category's *course list* into the *previous*
               category's rule while relabeling that rule after
               whatever course line came next. Net effect verified via
               parse dump: "American Foreign Policy" -- which should have
               7 alternative slots -- collapsed to a single mandatory
               course, INTNLREL110C, which has zero terms_offered in the
               snapshot. A solo mandatory course with no valid quarter is
               an unconditional `0 >= 1` constraint, which is exactly what
               was making the whole program INFEASIBLE (confirmed: fixing
               just this one category's course list flips the solve to
               OPTIMAL). Rebuilt all 6 "Core Program Requirements"
               categories by hand from the raw text; left the 11
               Depth-in-Discipline pathway pools, Electives, WIM, Capstone
               and Honors sections as the generic parser already produces
               them (those sections have enough real alternatives per
               pool that the same cascade bug, while present, doesn't
               happen to zero out any of them down to an unschedulable
               solo course).

  ENVSE-BS  -- Two independent bugs, both confirmed against the raw text
               and against parse dumps:
                 1. The "Electives" section's "Focus Breadth Electives"
                    prose reads "at least 1" (Coastal subplan) / "at least
                    2" (other 3 subplans), but a generic "Complete ALL of
                    the following Courses:" widget header on that same
                    section overrode it, making bulletin_parser.py treat
                    all 44 pool courses as mandatory (kind="all") instead
                    of kind="at_least_n". This alone was enough to flip
                    the Sustainable Energy Systems and Sustainable Urban
                    Systems subplans to OPTIMAL.
                 2. Coastal subplan only: the same inline-OR-inside-an-ALL-
                    list connector confusion as INTLR-BA hit the Coastal
                    subplan's Depth section -- "Required Core Courses"
                    dropped OCEANS71 and the CEE101B/CEE101E OR-alternative
                    entirely, and the nested "Focus Electives" 4-area
                    breadth block got relabeled/mis-split so its "Chemical
                    Processes" area collapsed to a solo mandatory CEE170,
                    which doesn't exist in the course snapshot at all --
                    an unconditional `0 >= 1`, exactly like INTLR-BA's bug.
                    Rebuilt Coastal's whole Depth sub-tree by hand from the
                    raw text (restoring OCEANS71, the CEE101B/CEE101E OR,
                    and the true 4-area Chemical-Processes/CEE177/
                    EARTHSYS152 alternatives) and confirmed OPTIMAL.
               Sustainable Freshwater Systems subplan is NOT fixed here --
               see NOTE below, it's a genuine course-data gap, not a rule
               bug.

  SLAV-BA   -- A requirements.py *subplan-scoping* bug, not a
               bulletin_parser.py parse bug: the "Capstone" section
               literally contains three different capstone-experience
               rules, one per subplan ("Slavic Studies Subplan Capstone
               Experience", "Russian Language and Literature Subplan
               Capstone Experience", "Russian and Philosophy Subplan
               Capstone Experience") -- but the section's own *title* is
               just "Capstone", which does not match
               is_subplan_section()'s \\b(subplan|track|path|option)\\b
               regex. So rules_from_parsed_program() never excludes it
               regardless of which subplan is selected, and a student
               choosing e.g. the Slavic Studies subplan ends up also
               hard-required to satisfy the Russian-and-Philosophy
               subplan's capstone (whose own parse is separately garbled
               by the same connector-confusion bug into an impossible
               "at_least_n=9 of a 1-item list" constraint -- 9 separate
               enrollments in a single course). Fixed by dropping every
               rule whose `.source` says it came from the "Capstone"
               section, then appending back only the one correct,
               hand-verified capstone rule for the chosen subplan.

NOTE on Sustainable Freshwater Systems (ENVSE-BS): its "Required Core
Courses" is CEE101B, CEE166A, CEE170, all "ALL"/mandatory -- and this is
*not* a parse bug, the raw bulletin text really does list exactly those
three with no OR/AND embedded. CEE170 is simply absent from
data/courses_snapshot.json entirely (not just missing terms_offered), so
no rule-tree fix can make this subplan schedulable, and no
course_overrides.json entry can fix it either -- course_data.py's
apply_overrides() silently no-ops on a course_key that isn't already in
the snapshot dict (see its `if c is None: print(...); continue`), so this
needs CEE170 actually added to the snapshot (e.g. a fresh
explorecourses fetch that picks it up), not a field-level override. Report
per the batch instructions rather than fixed here.
"""
from __future__ import annotations

from .models import RequirementRule
from .requirements import build_program_from_sample


# ---------------------------------------------------------------------------
# EE-BS
# ---------------------------------------------------------------------------

_EE_BS_ELECTIVES_PLACEHOLDER = "PLACEHOLDER:EE_BS_ELECTIVES_ANY_EE_COURSE"


def ee_bs_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """EE-BS parses correctly everywhere except "Electives": the bulletin
    ("Students may select any combination of electives from the Electives
    list. Electives may include up to two additional Engineering
    Fundamentals and any letter-graded lecture/lab EE courses...") is an
    open-ended pool that bulletin_parser.py can't enumerate, so it only
    captured the 4 courses actually spelled out by name (EE101B, EE102B,
    EE104, EE109) -- and EE104 has zero terms_offered in the snapshot, so
    the real reachable ceiling from named courses alone is 3 courses x 4
    units = 12 units against a stated 24-unit minimum. Adds one
    placeholder standing in for "another real EE elective," which the
    solver may schedule up to its existing repeat cap (see solver.py's
    _MAX_UNIT_MIN_REPEATS) -- confirmed this alone flips the solve from
    INFEASIBLE to OPTIMAL."""
    rules, placeholders = build_program_from_sample("EE-BS")
    placeholders = dict(placeholders)
    placeholders[_EE_BS_ELECTIVES_PLACEHOLDER] = (
        "Any additional letter-graded lecture/lab EE course (bulletin: 'Students may "
        "select any combination of electives from the Electives list...and any "
        "letter-graded lecture/lab EE courses') -- an open-ended pool, not "
        "individually enumerable from the bulletin text."
    )
    for r in rules:
        if r.label == "Electives" and r.kind == "unit_min":
            r.courses = list(r.courses) + [_EE_BS_ELECTIVES_PLACEHOLDER]
    return rules, placeholders


# ---------------------------------------------------------------------------
# INTLR-BA
# ---------------------------------------------------------------------------

# Transcribed directly from samples/INTLR-BA.txt lines ~62-125 ("Applied
# Economics Courses"). Cross-listed pairs (e.g. "ECON131 / OSPBEIJ30") kept
# as OR-alternative tuples so they count as one slot, matching how the rest
# of this program's pools already represent cross-listings.
_INTLR_BA_APPLIED_ECON = [
    "BUSGEN115", "CEE173S", "CEE176B", "EASTASN179", "ECON44", "ECON106",
    "ECON111", "ECON118", "ECON125", "ECON126", "ECON127", "ECON128",
    ("ECON131", "OSPBEIJ30"), "ECON135", ("ECON143", "BUSGEN143"), "ECON149",
    "ECON155", "ECON164", "ECON165", "ECON166", "EDUC306A", "ENERGY110",
    "ENVRES260", "HISTORY103D", "HISTORY200E", "HISTORY207C", "INTLPOL203",
    "INTLPOL207", "INTLPOL229", ("INTNLREL110C", "POLISCI110X"), "INTNLREL123",
    "INTNLREL147", "MS&E145", "OSPBER82", "OSPBER126X", "OSPFLOR26",
    "OSPMADRD40", "OSPMADRD54", "OSPOXFRD11", "OSPPARIS91", "OSPPARIS124X",
    "OSPSANTG119X", "POLISCI110G", "POLISCI141", "POLISCI213R", "POLISCI243",
    "POLISCI244C", "POLISCI247A", "POLISCI247G", "PUBLPOL104", "SIW103",
    ("SOC114", "SOC314"), "SOC175", "URBANST173",
]


def intlr_ba_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """INTLR-BA's "Core Program Requirements" section interleaves each of
    its 6 category headers (International Politics, Comparative
    Governance, American Foreign Policy, Introductory Economics, Applied
    Economics Courses, Skills Class) with inline "OR" course-alternative
    lines. bulletin_parser.py's connector detector desyncs at the first of
    these ("expected AND connector, saw OR after 'American Foreign
    Policy'" -- see its own ParseWarning), which cascades: each later
    category's real course list gets attributed to the *previous*
    category's rule, and that rule gets mislabeled with whatever course
    line came next. Net result verified via a raw parse dump:
    "American Foreign Policy" collapsed from its real 7 alternative slots
    down to a single mandatory course, INTNLREL110C -- which has zero
    terms_offered in the snapshot, so enforce_rule() emits an
    unconditional `0 >= 1` and the whole program reports INFEASIBLE no
    matter what else is schedulable. Rebuilds all 6 categories by hand
    from the raw bulletin text (samples/INTLR-BA.txt lines 23-135);
    everything else (Depth in Discipline's 11 pathway pools, Electives,
    WIM, Capstone, Honors) is left as the generic parser already produces
    it, since the same underlying connector bug touches some of those
    lists too but never happens to zero one down to an unschedulable solo
    course the way it did here."""
    rules, placeholders = build_program_from_sample("INTLR-BA")
    rules = [r for r in rules if r.label != "Degree Requirements Overview"]
    src = (
        "INTLR-BA program page: Core Program Requirements (manual fix -- "
        "connector-confusion cascade dropped/misattributed courses across "
        "categories; rebuilt from raw bulletin text, see manual_fix_batchB.py)"
    )
    core = RequirementRule(
        label="Degree Requirements Overview",
        kind="all",
        sub_rules=[
            RequirementRule(label="International Politics", kind="all",
                             courses=["INTNLREL101"], source=src),
            RequirementRule(label="Comparative Governance", kind="at_least_n", n=1,
                             courses=["INTNLREL102", "INTNLREL114D"], source=src),
            RequirementRule(label="American Foreign Policy", kind="at_least_n", n=1, courses=[
                ("INTNLREL110C", "POLISCI110X"), ("INTNLREL110D", "POLISCI110Y"),
                "INTNLREL154", "INTNLREL168A", ("INTNLREL168", "INTNLREL168W"),
                "INTNLREL173", "INTNLREL174",
            ], source=src),
            RequirementRule(label="Introductory Economics", kind="at_least_n", n=1, courses=[
                "ECON1", ("ECON50", "ECON50Q"), "ECON51", "ECON52",
            ], source=src),
            RequirementRule(label="Applied Economics Courses", kind="at_least_n", n=1,
                             courses=list(_INTLR_BA_APPLIED_ECON), source=src),
            RequirementRule(label="Skills Class", kind="at_least_n", n=1, courses=[
                "CS109", "DATASCI154", "ECON102A", "POLISCI150A", "STATS60", "STATS101",
            ], source=src),
        ],
        source=src,
    )
    rules.insert(0, core)
    return rules, placeholders


# ---------------------------------------------------------------------------
# ENVSE-BS
# ---------------------------------------------------------------------------

# Electives-section "Focus Breadth Electives" count, straight from the
# bulletin's own per-subplan prose (samples/ENVSE-BS.txt lines 40-41).
_ENVSE_BS_ELECTIVES_N = {
    "Sustainable Coastal Systems Subplan": 1,
    "Sustainable Energy Systems Subplan": 2,
    "Sustainable Freshwater Systems Subplan": 2,
    "Sustainable Urban Systems Subplan": 2,
}


def _envse_bs_fix_electives(rules: list[RequirementRule], subplan: str) -> None:
    n = _ENVSE_BS_ELECTIVES_N.get(subplan, 2)
    for r in rules:
        if r.label == "Focus Breadth Electives" and r.kind == "all":
            r.kind, r.n = "at_least_n", n


def envse_bs_manual_rules(
    subplan: str = "Sustainable Coastal Systems Subplan",
) -> tuple[list[RequirementRule], dict[str, str]]:
    """ENVSE-BS's "Electives" section prose says "at least 1" (Coastal) /
    "at least 2" (Energy/Freshwater/Urban) Focus Breadth Electives, but a
    generic "Complete ALL of the following Courses:" widget header on that
    same section overrode it, making bulletin_parser.py require all 44
    pool courses (kind="all") instead of kind="at_least_n". Fixing just
    that flips Sustainable Energy Systems and Sustainable Urban Systems to
    OPTIMAL outright.

    The Coastal subplan needs one more fix: its own Depth section has a
    second, independent connector-confusion bug (same class as
    INTLR-BA's) that dropped OCEANS71 and the CEE101B/CEE101E
    OR-alternative from "Required Core Courses", and relabeled/mis-split
    the "Focus Electives" 4-area breadth block so its "Chemical Processes"
    area collapsed to a solo mandatory CEE170 -- which isn't in the course
    snapshot at all, an unconditional `0 >= 1`. Rebuilt Coastal's Depth
    sub-tree by hand from the raw text; confirmed OPTIMAL afterward.

    Defaults to the Coastal subplan (alphabetically first, matching this
    codebase's own existing default-subplan convention -- see
    requirements.rules_from_parsed_program()'s docstring) since it's now
    fully fixed; pass subplan="Sustainable Energy Systems Subplan" or
    "Sustainable Urban Systems Subplan" for the other two working options.

    NOT fixed here: "Sustainable Freshwater Systems Subplan" stays
    INFEASIBLE even after the Electives fix -- its Depth "Required Core
    Courses" (CEE101B, CEE166A, CEE170, correctly parsed, no bug) hard-
    requires CEE170, which is absent from data/courses_snapshot.json
    entirely. That's a genuine course-data gap, not a rule bug -- see this
    module's docstring NOTE and this batch's final report for what's
    needed (CEE170 has to be added to the snapshot; course_overrides.json
    can't add a wholly-new course key, only correct fields on one that's
    already there)."""
    rules, placeholders = build_program_from_sample("ENVSE-BS", include_subplans=[subplan])
    _envse_bs_fix_electives(rules, subplan)

    if subplan == "Sustainable Coastal Systems Subplan":
        src = (
            "ENVSE-BS program page: Sustainable Coastal Systems Subplan (manual fix -- "
            "connector confusion dropped OCEANS71/the CEE101B-CEE101E OR-alternative and "
            "mislabeled the Focus Electives areas; rebuilt from raw bulletin text)"
        )
        depth = next((r for r in rules if r.label == "Sustainable Coastal Systems Depth"), None)
        if depth is not None:
            depth.sub_rules = [
                RequirementRule(
                    label="Required Core Courses", kind="all",
                    courses=["CEE162F", ("CEE101B", "CEE101E"), "OCEANS71"],
                    source=src,
                ),
                RequirementRule(
                    label="Focus Electives (breadth across 4 areas)", kind="all",
                    sub_rules=[
                        RequirementRule(label="Physical Processes", kind="at_least_n", n=1,
                                         courses=["CEE162D", "CEE162I", "CEE262B", "ESS141"],
                                         source=src),
                        RequirementRule(label="Marine Biology/Ecology", kind="at_least_n", n=1,
                                         courses=["EARTHSYS151", "OCEANS125H", "OSPAUSTL10"],
                                         source=src),
                        RequirementRule(label="Chemical Processes", kind="at_least_n", n=1,
                                         courses=[("CEE170", "CEE177"), "EARTHSYS152"],
                                         source=src),
                        RequirementRule(label="Coastal Issues", kind="at_least_n", n=1,
                                         courses=["CEE162E", "CEE175A", "CEE272", "ESS164",
                                                  "ESS227", "OSPAUSTL32"],
                                         source=src),
                    ],
                    source=src,
                ),
            ]
    return rules, placeholders


# ---------------------------------------------------------------------------
# SLAV-BA
# ---------------------------------------------------------------------------

def _slav_ba_base(subplan: str) -> tuple[list[RequirementRule], dict[str, str]]:
    """Shared, correctly-parsed sections (Core Program Requirements,
    Electives, WIM, plus the chosen subplan's own section) with the leaky
    "Capstone" section (see module docstring) dropped -- every rule whose
    .source records it came from the "Capstone" section is removed,
    regardless of which subplan was requested, since that section
    currently always includes all 3 subplans' capstones no matter what."""
    rules, placeholders = build_program_from_sample("SLAV-BA", include_subplans=[subplan])
    rules = [r for r in rules if not r.source.endswith(": Capstone")]
    return rules, placeholders


def slav_ba_slavic_studies_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """Slavic Studies subplan, capstone correctly scoped to just this
    track: Fulfill ANY of {Independent Study (SLAVIC199), Honors Thesis
    (>=9 units DLCL189H)} -- this is exactly what bulletin_parser.py
    already produces correctly for this subplan's own capstone rule
    (samples/SLAV-BA.txt lines 65-74); the only bug was that the OTHER two
    subplans' capstone rules were leaking in alongside it."""
    rules, placeholders = _slav_ba_base("Slavic Studies Subplan")
    rules.append(RequirementRule(
        label="Slavic Studies Subplan Capstone Experience", kind="at_least_n", n=1,
        sub_rules=[
            RequirementRule(label="Independent Study", kind="all", courses=["SLAVIC199"]),
            RequirementRule(label="Honors Thesis", kind="unit_min", unit_min=9, courses=["DLCL189H"]),
        ],
        source="SLAV-BA program page: Capstone (manual fix, scoped to Slavic Studies Subplan only)",
    ))
    return rules, placeholders


def slav_ba_russian_lang_lit_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """Russian Language and Literature subplan -- same capstone shape as
    Slavic Studies (Independent Study SLAVIC199 OR >=9-unit Honors Thesis
    DLCL189H, samples/SLAV-BA.txt lines 85-94), scoped so the Russian-and-
    Philosophy subplan's capstone doesn't also get pulled in."""
    rules, placeholders = _slav_ba_base("Russian Language and Literature Subplan")
    rules.append(RequirementRule(
        label="Russian Language and Literature Subplan Capstone Experience", kind="at_least_n", n=1,
        sub_rules=[
            RequirementRule(label="Independent Study", kind="all", courses=["SLAVIC199"]),
            RequirementRule(label="Honors Thesis", kind="unit_min", unit_min=9, courses=["DLCL189H"]),
        ],
        source="SLAV-BA program page: Capstone (manual fix, scoped to Russian Language and Literature Subplan only)",
    ))
    return rules, placeholders


_SLAV_BA_RP_CAPSTONE_PLACEHOLDER = "PLACEHOLDER:SLAV_BA_RUSSIAN_PHILOSOPHY_CAPSTONE_SEMINAR_OR_TOPIC_PAPER"


def slav_ba_russian_philosophy_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """Russian and Philosophy subplan. Its own capstone text (samples/
    SLAV-BA.txt lines 105-123) is "Fulfill ANY of: (a) Philosophy and
    Literature Capstone Seminar [no fixed course, chosen with the DUS/
    Faculty Advisor], (b) Honors Thesis [>=9 units DLCL189H], (c) a
    5,000-word Topic Paper via a 3-unit letter-graded independent study in
    the Philosophy and Literature Focal Group [again no fixed course
    code]." bulletin_parser.py's own attempt at this one is separately
    garbled by connector confusion into an impossible "at_least_n=9 of a
    1-item course list" (9 literal enrollments in DLCL189H) -- so this is
    hand-built rather than patched. (a) and (c) name no actual course, so
    they're represented as one placeholder pool standing in for "an
    independent-study/seminar capstone arranged with faculty," alongside
    the one real, enumerable option, (b)."""
    rules, placeholders = _slav_ba_base("Russian and Philosophy Subplan")
    placeholders = dict(placeholders)
    placeholders[_SLAV_BA_RP_CAPSTONE_PLACEHOLDER] = (
        "Russian and Philosophy capstone: either a Philosophy and Literature Capstone "
        "Seminar (chosen in consultation with the DUS/Faculty Advisor for Philosophy, no "
        "fixed course code) or a 5,000-word Topic Paper via a 3-unit letter-graded "
        "independent study in the Philosophy and Literature Focal Group -- neither option "
        "names a specific course in the bulletin."
    )
    rules.append(RequirementRule(
        label="Russian and Philosophy Subplan Capstone Experience", kind="at_least_n", n=1,
        sub_rules=[
            RequirementRule(label="Honors Thesis", kind="unit_min", unit_min=9, courses=["DLCL189H"]),
            RequirementRule(label="Capstone Seminar or Topic Paper", kind="unit_min", unit_min=3,
                             courses=[_SLAV_BA_RP_CAPSTONE_PLACEHOLDER]),
        ],
        source="SLAV-BA program page: Capstone (manual fix, scoped to Russian and Philosophy Subplan only)",
    ))
    return rules, placeholders


def slav_ba_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """Default SLAV-BA entry point (zero-arg, for MANUAL_FIXES) -- picks
    the Slavic Studies subplan, since it shares the major's own name and
    is the most central of the 3 tracks. All 3 subplans are independently
    confirmed OPTIMAL (see this batch's report); the other two are
    available as slav_ba_russian_lang_lit_manual_rules() and
    slav_ba_russian_philosophy_manual_rules() above if the app ever wires
    up per-subplan selection for this program the way CS-BS/DATSC-BS
    already do."""
    return slav_ba_slavic_studies_manual_rules()


MANUAL_FIXES = {
    "EE-BS": ee_bs_manual_rules,
    "INTLR-BA": intlr_ba_manual_rules,
    "ENVSE-BS": envse_bs_manual_rules,
    "SLAV-BA": slav_ba_manual_rules,
}
