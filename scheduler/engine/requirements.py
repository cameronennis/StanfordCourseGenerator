"""
Requirement trees: which courses a person still needs.

Two ways rules get built now:
  1. Hand-encoded, e.g. build_cs_bs_core_requirements() below -- a manually
     transcribed SUBSET, kept as the original proof that RequirementRule
     can express Stanford's real AND/OR/"complete N of M" logic.
  2. Parsed, via bulletin_parser.py -- reads the actual text of a
     bulletin.stanford.edu/programs/{CODE} page and builds the tree
     automatically. This is the scale-up path: validated against CS-BS,
     EE-BS, and CS-MIN (see samples/ and bulletin_parser.py's own
     docstring for what's covered and what's known to be heuristic).

Each program sheet gets its own build_*_requirements() function; a
person's active set is the union of the ones for their declared
major(s)/minor(s), plus (for a BS with subplans/tracks) whichever one
subplan they've chosen.
"""
from __future__ import annotations

import re
from pathlib import Path

from .bulletin_parser import ProgramRequirements, parse_program_text
from .models import Course, RequirementRule, make_placeholder_course

SAMPLES_DIR = Path(__file__).parent.parent / "samples"

# Stanford program pages name their "pick exactly one" alternative-curriculum
# sections in more ways than the literal word "subplan": ARCHA-MIN calls
# its four alternatives "Minor Track 1: ..." / "Minor Track 4: ..." plus a
# "Self-Designed Path"; MELLC-MIN's are "Arabic Track" / "Hebrew Track" /
# etc; ETHSO-MIN mixes "... - General Track" with "... Subplan" for its two
# options. A plain substring check on "subplan" misses all of these, so
# every one of their tracks got required simultaneously instead of "pick
# one" -- confirmed via an actual build_and_solve() run that this made
# ARCHA-MIN INFEASIBLE (needing electives/theory/methods courses from all
# four tracks plus the self-designed path at once).
#
# Matching whole words (not substrings) also matters, not just adds cases:
# GLBLST-MIN has one shared section titled "All Subplans : Global Studies
# General Requirement" (its common core, independent of which subplan a
# student picks) alongside seven real per-track "Subplan: African Studies"
# / etc sections. A plain substring check matches "Subplans" too (it
# contains "subplan"), so that shared section was being wrongly excluded
# right along with the real per-track ones. Word-boundary matching leaves
# plural, collective titles like "All Subplans : ..." or "...Tracks
# Overview" alone while still catching every singular, specific one
# ("Subplan: ...", "... Subplan", "... Track", "Track 1: ...").
_ALT_TRACK_TITLE_RE = re.compile(r"\b(subplan|track|path|option)\b", re.IGNORECASE)


def is_subplan_section(title: str) -> bool:
    return bool(_ALT_TRACK_TITLE_RE.search(title))


def list_subplans(prog: ProgramRequirements) -> list[str]:
    return [sec.title for sec in prog.sections if is_subplan_section(sec.title)]


def rules_from_parsed_program(
    prog: ProgramRequirements, include_subplans: list[str] | None = None
) -> list[RequirementRule]:
    """Flatten a parsed program's non-subplan sections (School Requirements,
    Core, Electives, WIM, Capstone, ...) plus whichever named subplan
    section(s) the caller opts into -- nobody gets every subplan's
    requirements piled on by default, since a BS student picks exactly one
    track."""
    include_subplans = set(include_subplans or [])
    all_subplans = list_subplans(prog)
    if not include_subplans and all_subplans and all(
        is_subplan_section(sec.title) for sec in prog.sections
    ):
        # This program's ENTIRE page is subplan/track sections -- there's
        # no shared base curriculum outside a track at all (e.g. GLBLST-MIN:
        # "African Studies" / "European Studies" / ... with nothing else).
        # Calling this with no subplan selected would otherwise silently
        # return zero rules -- which looks identical to "this program has
        # no requirements", the exact kind of silent-empty-result bug
        # bulletin_parser.py's placeholder mechanism exists to prevent.
        # Default to the first track alphabetically (same idea as CS-BS's
        # explicit AI-subplan default and DATSC-BS's Math/Comp default,
        # just automatic here since there are too many of these programs
        # to hand-pick a default for each -- see PROGRAM_SHEET_BUILDERS).
        include_subplans = {sorted(all_subplans)[0]}
    rules: list[RequirementRule] = []
    for sec in prog.sections:
        if is_subplan_section(sec.title) and sec.title not in include_subplans:
            continue
        if sec.title in ("Honors",):
            continue  # opt-in only, not a graduation requirement
        for r in sec.rules:
            r.source = f"{prog.code} program page: {sec.title}"
            rules.append(r)
    return rules


def build_program_from_sample(
    code: str, include_subplans: list[str] | None = None
) -> tuple[list[RequirementRule], dict[str, str]]:
    """Parse a program from the bundled samples/{code}.txt page-text
    snapshot. These snapshots were fetched via Claude in Chrome from the
    live bulletin page and validated by hand -- refresh the .txt file
    (same fetch-twice-for-hydration approach, see bulletin_parser.py) when
    a program sheet changes.

    Returns (rules, placeholder_courses) -- the second element is a
    key->description dict for any placeholder courses bulletin_parser.py
    had to invent (see models.make_placeholder_course); pass it to
    build_requirements()'s out_placeholders, then merge it into the course
    dict via course_data.add_placeholder_courses() before solving, or
    those placeholder keys won't actually be schedulable."""
    path = SAMPLES_DIR / f"{code}.txt"
    prog = parse_program_text(path.read_text(), code)
    if prog.warnings:
        for w in prog.warnings:
            print(f"  [warn] {code} {w.where}: {w.message}")
    return rules_from_parsed_program(prog, include_subplans=include_subplans), prog.placeholder_courses


def build_cs_bs_core_requirements() -> list[RequirementRule]:
    return [
        RequirementRule(
            label="Required Math Courses",
            kind="all",
            courses=["CS103", "CS109", "MATH19", "MATH20", "MATH21"],
            source="CS-BS program sheet: Mathematics",
        ),
        RequirementRule(
            label="Math Electives (at least 2)",
            kind="at_least_n",
            n=2,
            courses=[
                "CME100", "CME102", "CME104", "CS157", "CS205L", "ENGR108",
                "MATH51", "MATH52", "MATH53", "MATH104", "MATH107", "MATH108",
                "MATH109", "MATH110", "MATH113", "PHIL151",
            ],
            source="CS-BS program sheet: Mathematics",
        ),
        RequirementRule(
            label="Engineering Fundamentals",
            kind="all",
            courses=["CS106B", ("ENGR40M", "ENGR76")],
            source="CS-BS program sheet: Engineering Fundamentals",
        ),
        RequirementRule(
            label="Core Program Requirements",
            kind="all",
            courses=[("CS107", "CS107E"), "CS111", "CS161"],
            source="CS-BS program sheet: Core Program Requirements",
        ),
    ]


# Registry so an app/UI can list "available program sheets" without
# importing each builder function by hand.
#
# A handful of programs need a NON-default choice explicitly spelled out
# (which subplan/track to default to, since build_program_from_sample()
# otherwise includes none of them -- a BS student picks exactly one
# track, see rules_from_parsed_program()'s docstring). Those get their own
# hand-written entry below. Every other program sheet gets a
# mechanically-generated "{CODE} (parsed, full)" entry for every
# samples/{CODE}.txt on disk, covering the full 218-program in-scope set
# (see data/bulletin_program_codes.json, engine/fetch_bulletin.py) without
# needing 218 hand-maintained dict lines.
# Hand-fixed replacements for 14 programs whose generic bulletin_parser.py
# output came back wrong in ways a targeted parser fix would be too risky to
# generalize (see README's "Hand-reasoned fixes for the 14 parser-bug
# programs" section) -- each is a from-scratch, by-hand transcription of the
# raw samples/{CODE}.txt text (engine/manual_fix_batch{A,B,C,D}.py), reusing
# each program's OWN default "{CODE} (parsed, full)" label so it transparently
# replaces the buggy generic parse everywhere that label is already used
# (the sweep script, the app's selector) without adding a confusing duplicate
# entry.
#
# Imported here (after build_program_from_sample is already defined above),
# not at module top, because manual_fix_batchB.py itself calls
# build_program_from_sample() for its non-buggy sections -- importing at the
# top would be a circular import (this module isn't done initializing yet).
from .manual_fix_batchA import MANUAL_FIXES as _MANUAL_FIXES_A
from .manual_fix_batchB import (
    MANUAL_FIXES as _MANUAL_FIXES_B,
    envse_bs_manual_rules as _envse_bs_manual_rules,
    slav_ba_russian_lang_lit_manual_rules as _slav_ba_russian_lang_lit_manual_rules,
    slav_ba_russian_philosophy_manual_rules as _slav_ba_russian_philosophy_manual_rules,
)
from .manual_fix_batchC import MANUAL_FIXES as _MANUAL_FIXES_C
from .manual_fix_batchD import MANUAL_FIXES as _MANUAL_FIXES_D
# Added post-hoc (not one of the original 14 parser-bug programs) -- found
# while validating the double-major/double-degree stacking feature against
# a real CHEM-BS + CS-BS combination. Same standalone-AND/OR root cause as
# batch A/B/C/D -- see manual_fix_chem_bs.py's own docstring.
from .manual_fix_chem_bs import (
    chem_bs_manual_rules as _chem_bs_manual_rules,
    chem_bs_traditional_path_rules as _chem_bs_traditional_path_rules,
    chem_bs_biological_path_rules as _chem_bs_biological_path_rules,
)

_MANUAL_FIXES = {
    **_MANUAL_FIXES_A, **_MANUAL_FIXES_B, **_MANUAL_FIXES_C, **_MANUAL_FIXES_D,
    "CHEM-BS": _chem_bs_manual_rules,
}

_SPECIAL_PROGRAM_SHEET_BUILDERS = {
    # NOTE: build_cs_bs_core_requirements() (a tiny hand-encoded "prove the
    # pipeline works" placeholder from before real bulletin parsing
    # existed -- 4 requirement groups, nowhere near the real CS-BS major)
    # deliberately does NOT get a registry entry here anymore. It used to,
    # as "CS-BS (core subset, v1, hand-encoded)" -- and because dict/set
    # ordering put it alphabetically before "CS-BS (parsed, full, AI
    # subplan)", app.py's old "default to the first selectable sheet" logic
    # silently pre-selected it, so a real user's generated schedule quietly
    # used this placeholder stub instead of the actual major (confirmed:
    # "no CS courses outside of the core" in a real run). The function
    # itself is untouched (smoke_test.py still calls it directly to
    # exercise its AND-group/nested-sub_rule shapes) -- it's just no longer
    # reachable from the program-sheet picker at all, since there's no
    # version of "expose this in the UI" that isn't confusing (see also
    # app.py's picker redesign, which stopped defaulting to ANY sheet).
    "CS-BS (parsed, full, AI subplan)": lambda: build_program_from_sample(
        "CS-BS", include_subplans=["Artificial Intelligence Subplan"]
    ),
    "DATSC-BS (parsed, full, Math/Comp subplan)": lambda: build_program_from_sample(
        "DATSC-BS", include_subplans=["Mathematics and Computation Subplan"]
    ),
    **{f"{code} (parsed, full)": fn for code, fn in _MANUAL_FIXES.items()},
    # Extra named-subplan variants for the two hand-fixed multi-track
    # programs whose other tracks are also independently confirmed working
    # (see manual_fix_batchB.py) -- same convention as CS-BS/DATSC-BS above.
    "ENVSE-BS (parsed, full, Sustainable Energy Systems subplan)": lambda: _envse_bs_manual_rules(
        "Sustainable Energy Systems Subplan"
    ),
    "ENVSE-BS (parsed, full, Sustainable Urban Systems subplan)": lambda: _envse_bs_manual_rules(
        "Sustainable Urban Systems Subplan"
    ),
    "SLAV-BA (parsed, full, Russian Language and Literature subplan)": _slav_ba_russian_lang_lit_manual_rules,
    "SLAV-BA (parsed, full, Russian and Philosophy subplan)": _slav_ba_russian_philosophy_manual_rules,
    # Same convention again: CHEM-BS's "Breadth in Discipline" section is a
    # pick-one-of-two ("Chemistry Pathway"), not a subplan choice in the
    # bulletin's own terms, but it's the same kind of either/or decision a
    # student commits to up front -- so it gets the same named-variant
    # treatment. The plain "CHEM-BS (parsed, full)" entry above (from
    # _MANUAL_FIXES) still exists too, and still models the pick-one-of-two
    # as an at_least_n(1) choice the solver picks for you; these two let a
    # student who already knows which path they're on skip that choice.
    "CHEM-BS (parsed, full, Traditional Chemistry Path)": _chem_bs_traditional_path_rules,
    "CHEM-BS (parsed, full, Biological Chemistry Path)": _chem_bs_biological_path_rules,
}
# Codes already covered by a hand-written entry above (parsed from each
# label's "{CODE} (...)" prefix) -- these don't also get a generic
# default-subplan entry, to avoid a confusing "which CS-BS do I pick"
# duplicate in the selector.
_SPECIALLY_HANDLED_CODES = {
    label.split(" (", 1)[0] for label in _SPECIAL_PROGRAM_SHEET_BUILDERS
}


def _default_builder_for(code: str):
    # Default argument binds `code` at lambda-creation time, not call time
    # -- otherwise every generated lambda in the loop below would close
    # over the SAME final loop variable (a classic late-binding bug).
    return lambda code=code: build_program_from_sample(code)


PROGRAM_SHEET_BUILDERS = dict(_SPECIAL_PROGRAM_SHEET_BUILDERS)
for _path in sorted(SAMPLES_DIR.glob("*.txt")):
    _code = _path.stem
    if _code in _SPECIALLY_HANDLED_CODES:
        continue
    # Registered unconditionally here, even for a program whose bare entry
    # will be dropped again a bit further down (see the "no solver-decides
    # fallback for any program with real alternatives" cleanup pass below)
    # -- the track- and choice-point-discovery loops that come after this
    # both find which codes to look at via `{_code_of(_name) for _name in
    # PROGRAM_SHEET_BUILDERS}`, so a code with zero entries at this point
    # would never be considered for its own (perfectly valid) named
    # subplan variants either. Registering now and removing the bare key
    # later, once it's known whether any real sibling variant exists, keeps
    # both concerns correctly separated.
    PROGRAM_SHEET_BUILDERS[f"{_code} (parsed, full)"] = _default_builder_for(_code)


# Programs confirmed genuinely INFEASIBLE by the full track-checking sweep
# (every track/subplan alternative tried, 25s CP-SAT time limit, 0
# inconclusive/timeout results -- see README's feasibility-sweep section)
# even after: the cross-listing term merge, the two solver "last resort"
# softenings, all 24 course_overrides.json corrections, and hand-reasoned
# rule-tree fixes for all 14 programs that turned out to be parser bugs
# rather than real infeasibility. These 8 are left broken on purpose rather
# than papered over with a guess -- each is a genuine data or scope gap, not
# a bug this codebase can fix by itself:
#   - a truly-required course with no confirmed offering history to fall
#     back on (not even the kind of "just not this year" gap a
#     CourseOverride can responsibly correct), or
#   - a truly-required course that only runs in Summer, which this
#     scheduler's quarter-generation doesn't currently offer as an option.
# Keyed by the exact PROGRAM_SHEET_BUILDERS label so the UI can look a
# selection up directly. Only base "(parsed, full)" labels appear here for
# a program with no real subplan alternatives of its own (a hand-fixed
# variant label is, by construction, one this project already confirmed
# solves) -- EXCEPT for AFRAM-BA and ED-MA below, which DO have named
# subplan variants, because their infeasibility comes from a course
# required by every one of those variants (confirmed directly: re-solving
# each of AFRAM-BA's 3 and ED-MA's 6 named variants came back INFEASIBLE,
# same root cause each time), not something specific to the "no subplan
# chosen" fallback -- so every one of their real variant labels needs its
# own entry here, or picking a specific subplan would look like it dodges
# a problem it doesn't actually dodge. (This is also why AFRAM-BA/ED-MA
# no longer have a bare "(parsed, full)" entry to key a broken flag to in
# the first place -- see the "no solver-decides fallback for any program
# with real alternatives" cleanup pass further down.)
BROKEN_PROGRAMS: dict[str, str] = {
    "AFRAM-BA (parsed, full, Cultures and Arts)": (
        "A required course has no confirmed ExploreCourses offering history "
        "in any of the last several years -- genuine data gap, not a bug. "
        "True of every AFRAM-BA subplan alternative, not just this one."
    ),
    "AFRAM-BA (parsed, full, Science, Technology & Society)": (
        "A required course has no confirmed ExploreCourses offering history "
        "in any of the last several years -- genuine data gap, not a bug. "
        "True of every AFRAM-BA subplan alternative, not just this one."
    ),
    "AFRAM-BA (parsed, full, Ethical, Social, and Political Thought)": (
        "A required course has no confirmed ExploreCourses offering history "
        "in any of the last several years -- genuine data gap, not a bug. "
        "True of every AFRAM-BA subplan alternative, not just this one."
    ),
    "ENERGY-BS (parsed, full)": (
        "A required course has no confirmed ExploreCourses offering history "
        "in any of the last several years -- genuine data gap, not a bug."
    ),
    "ENERGY-MIN (parsed, full)": (
        "A required course has no confirmed ExploreCourses offering history "
        "in any of the last several years -- genuine data gap, not a bug."
    ),
    "CSB-MS (parsed, full)": (
        "A required course has no confirmed ExploreCourses offering history "
        "in any of the last several years -- genuine data gap, not a bug."
    ),
    "HMN GEN-MS (parsed, full)": (
        "A required course has no confirmed ExploreCourses offering history "
        "in any of the last several years -- genuine data gap, not a bug."
    ),
    "CIMGT-MS (parsed, full)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option."
    ),
    "IMMUN-MS (parsed, full)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option."
    ),
    "ED-MA (parsed, full, Curriculum Studies and Teacher Education (CTE) Subplan)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option -- true across "
        "every subplan alternative for this program."
    ),
    "ED-MA (parsed, full, Global and Comparative Education (GCE) Subplan and International Education Policy Analysis (IEPA) Subplan)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option -- true across "
        "every subplan alternative for this program."
    ),
    "ED-MA (parsed, full, Joint Degree with the Graduate School of Business (MA/MBA) Subplan)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option -- true across "
        "every subplan alternative for this program."
    ),
    "ED-MA (parsed, full, Joint Degree with the Public Policy Program (MA/MPP) Subplan)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option -- true across "
        "every subplan alternative for this program."
    ),
    "ED-MA (parsed, full, Joint Degree with the Stanford Law School (MA/JD) Subplan)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option -- true across "
        "every subplan alternative for this program."
    ),
    "ED-MA (parsed, full, Policy, Organization and Leadership Studies (POLS) Subplan)": (
        "A required course only runs in Summer, which this scheduler's "
        "quarter list doesn't currently include as an option -- true across "
        "every subplan alternative for this program."
    ),
}


# Default "how many real alternatives makes this a LONG list, not just a
# normal 2-3-option choice" cutoff for placeholderize_large_elective_pools
# below. Picked by looking at real cases on both sides: CS-BS's Capstone/
# WiM choice (CS191W/CS194W/CS210B, 3 real options) and a typical Math
# Electives-style pick-a-few-from-a-short-list rule stay well under this;
# CS-BS's General/Track Electives pools (dozens of real course
# alternatives) are well over it.
_LARGE_ELECTIVE_POOL_THRESHOLD = 8


def _pool_size(rule: RequirementRule) -> int:
    """How many distinct real alternatives sit directly in this rule's OWN
    `courses` list (sub_rules not counted -- those get walked/rewritten
    separately). An OR-alternatives tuple counts every member (each is a
    real, separate choice); an AND-group counts as ONE alternative (its
    members are a single package, not separate options); a bare string
    counts as one."""
    size = 0
    for item in rule.courses:
        if isinstance(item, tuple) and item and item[0] == "__and__":
            size += 1
        elif isinstance(item, (tuple, list)):
            size += len(item)
        else:
            size += 1
    return size


def placeholderize_large_elective_pools(
    rules: list[RequirementRule], threshold: int = _LARGE_ELECTIVE_POOL_THRESHOLD
) -> tuple[list[RequirementRule], dict[str, Course]]:
    """Real, user-reported complaint: an at_least_n/unit_min rule whose own
    pool is a genuinely LONG list of interchangeable real electives (e.g.
    CS-BS's ~100-course General/Track Electives pool) let the solver commit
    to one arbitrary specific real course -- e.g. PSYCH204A -- for each
    slot. That reads as a firm, meaningful pick even though it's really
    just whichever alternative CP-SAT happened to land on among dozens of
    equally-valid options. Same "don't present an arbitrary choice as fact"
    reasoning already applied to WAYS/COLLEGE/Writing/Language (see
    university_requirements.py) and to unenumerated bulletin pools (see
    models.make_placeholder_course) -- this extends it to a program
    sheet's own large, fully-ENUMERATED elective pools too.

    Only rewrites a rule whose own pool size (see _pool_size -- sub_rules
    excluded, walked/rewritten independently) exceeds `threshold`. A small
    pool (e.g. CS-BS's 3-option Capstone/WiM choice) is left with its real,
    named courses -- naming 2-3 genuine alternatives isn't the "arbitrary
    pick from dozens" problem this fixes, and some of those small pools
    have their own real class-year restrictions (see course_data.py's
    KNOWN_CLASS_YEAR_MIN) that only make sense attached to real course
    keys.

    Returns a NEW list of NEW RequirementRule objects (never mutates the
    input -- every field is copied, sub_rules recursively) plus a dict of
    the placeholder Course objects created, in the same
    key -> Course shape build_all_university_rules's own placeholder dicts
    already use (NOT the key -> str shape build_program_from_sample's
    out_placeholders uses -- pass this dict straight to
    course_data.add_placeholder_courses, same as university_placeholders
    in app.py)."""
    placeholders: dict[str, Course] = {}

    def _rewrite(rule: RequirementRule) -> RequirementRule:
        new_sub_rules = [_rewrite(sr) for sr in rule.sub_rules]
        if rule.kind in ("at_least_n", "unit_min") and _pool_size(rule) > threshold:
            count = rule.n if rule.kind == "at_least_n" else max(1, int((rule.unit_min or 4.0) // 4))
            count = max(1, count or 1)
            slug = re.sub(r"[^A-Z0-9]+", "_", rule.label.upper()).strip("_") or "ELECTIVE"
            new_courses: list = []
            for i in range(count):
                key = f"PLACEHOLDER:{slug}_{i + 1}" if count > 1 else f"PLACEHOLDER:{slug}"
                desc = f"{rule.label} (elective slot {i + 1} of {count})" if count > 1 else rule.label
                placeholders[key] = make_placeholder_course(key, desc)
                new_courses.append(key)
            return RequirementRule(
                label=rule.label, kind="all", courses=new_courses, source=rule.source,
                program_id=rule.program_id,
            )
        return RequirementRule(
            label=rule.label, kind=rule.kind, courses=list(rule.courses), n=rule.n,
            unit_min=rule.unit_min, sub_rules=new_sub_rules, source=rule.source,
            program_id=rule.program_id, sequential=rule.sequential,
            full_year_sequence=rule.full_year_sequence,
        )

    return [_rewrite(r) for r in rules], placeholders


def build_requirements_grouped(
    program_sheet_names: list[str], out_placeholders: dict[str, str] | None = None
) -> dict[str, list[RequirementRule]]:
    """Same builder-calling logic as build_requirements(), but keeps each
    sheet's rules in their own bucket (keyed by the exact sheet display
    name passed in) instead of flattening them into one list. Needed by
    engine/multi_program.py to reason about which rules belong to which
    program when 2+ majors are being planned together (double
    major/double degree) -- build_requirements() itself is unchanged and
    still the right call for every caller that doesn't care about that
    distinction (the app's default single-or-independent-programs path,
    every validation/sweep script, etc.)."""
    grouped: dict[str, list[RequirementRule]] = {}
    for name in program_sheet_names:
        builder = PROGRAM_SHEET_BUILDERS.get(name)
        if builder is None:
            print(f"  [warn] no requirement builder registered for {name!r}, skipping")
            continue
        result = builder()
        if isinstance(result, tuple):
            built_rules, placeholders = result
        else:
            built_rules, placeholders = result, {}
        grouped[name] = built_rules
        if out_placeholders is not None:
            out_placeholders.update(placeholders)
    return grouped


def list_track_options(code: str) -> list[str]:
    """Real named subplans/tracks for a parsed program (see
    is_subplan_section()'s regex) -- empty if this program has none
    in its cached samples/{code}.txt, OR if that section exists on the
    live bulletin page but its content was never actually fetched into
    the sample (confirmed to happen at least once: samples/CS-BS.txt's
    own prose mentions a "Computer Engineering Subplan" by name, but only
    "Artificial Intelligence Subplan" has real section content -- the
    other tracks' course lists were never captured, so this correctly
    returns just the one track it actually has data for, not a name it
    can't back up with real requirements)."""
    path = SAMPLES_DIR / f"{code}.txt"
    if not path.exists():
        return []
    prog = parse_program_text(path.read_text(), code)
    return sorted(list_subplans(prog))


def build_program_track(
    code: str, track: str | None = None
) -> tuple[list[RequirementRule], dict[str, str]]:
    """Rules for one specific named track (or the program's default scope
    if track is None) -- thin wrapper around build_program_from_sample()
    for picking a track outside the hand-picked PROGRAM_SHEET_BUILDERS
    entries (e.g. "CS-BS (parsed, full, AI subplan)"), so a caller isn't
    limited to whichever subplan variants happen to already have their own
    baked-in registry label."""
    return build_program_from_sample(code, include_subplans=[track] if track else None)


def solve_best_track(code: str, other_rules: list[RequirementRule], solve_fn, tracks: list[str] | None = None):
    """Try every real track for `code` (or the given subset of them),
    combine each one's rules with `other_rules` (everything else already
    in the plan -- other programs' requirements, university requirements,
    etc.), and return whichever track's solve is "best": a feasible
    (OPTIMAL/FEASIBLE) result beats an INFEASIBLE one outright, and among
    feasible results, the higher objective_value wins (this project's
    objective already rewards weight-table preferences and penalizes
    unnecessary courses/overage/typical-order violations, so "best
    objective" already means "most optimized" in exactly the sense the
    rest of this app already optimizes for -- see engine/solver.py).

    `solve_fn(rules, placeholders) -> ScheduleResult` is caller-provided
    and deliberately left free of build_and_solve's own parameter list
    (courses, quarter_list, weight_rules, unit cap, ...) -- the caller
    already has all of that in scope, and is responsible for merging
    `placeholders` into its own course dict copy (via
    course_data.add_placeholder_courses()) before calling build_and_solve,
    since each track can need different placeholders.

    Returns (chosen_track_name_or_None, chosen_rules, chosen_placeholders,
    chosen_result). If `code` has no real tracks, tries the single
    "no subplan selected" default scope instead (track=None)."""
    candidate_tracks = tracks if tracks is not None else (list_track_options(code) or [None])
    best = None
    for track in candidate_tracks:
        rules, placeholders = build_program_track(code, track)
        result = solve_fn(other_rules + rules, placeholders)
        rank = (
            0 if result.status in ("OPTIMAL", "FEASIBLE") else 1,
            -(result.objective_value or 0),
        )
        if best is None or rank < best[0]:
            best = (rank, track, rules, placeholders, result)
    _rank, chosen_track, chosen_rules, chosen_placeholders, chosen_result = best
    return chosen_track, chosen_rules, chosen_placeholders, chosen_result


def _code_of(sheet_name: str) -> str:
    """'CS-BS (parsed, full, AI subplan)' -> 'CS-BS'. Duplicated (rather
    than imported) from multi_program.sheet_code() to avoid this module
    depending on multi_program -- multi_program already depends on nothing
    in requirements.py, and this one-liner isn't worth risking a future
    circular import over."""
    return sheet_name.split(" (", 1)[0]


def variant_label(code: str, full_sheet_name: str) -> str:
    """Clean, code-agnostic display name for one program's variant, for a
    UI that lets someone pick a program by CODE first and a specific
    subplan/path/version second (see app.py's picker) -- e.g.
    'CS-BS (parsed, full, AI subplan)' + 'CS-BS' -> 'AI subplan';
    'CHEM-BS (parsed, full)' + 'CHEM-BS' -> 'Whole program (solver decides
    between any built-in choices)' (true of any parsed program with an
    internal pick-one-of-several rule, like CHEM-BS's Chemistry Pathway --
    not a CHEM-BS-specific claim). Defined up here (rather than next to
    list_variant_labels_by_code(), which actually needs it for the UI)
    because the auto-discovery loops just below ALSO need it -- computing
    "what variants already exist for this code" by crudely stripping
    " ()" off a raw label (an earlier version of this file's own bug)
    left the "parsed, full, " prefix in place, so e.g. CS-BS's dedup check
    was comparing "Artificial Intelligence Subplan" against the literal
    string "parsed, full, AI subplan" instead of just "AI subplan" --
    never matched, hence the duplicate a user actually ran into. Using the
    same real display-label logic everywhere avoids that class of bug by
    construction."""
    suffix = full_sheet_name[len(code):].strip()
    if suffix.startswith("(") and suffix.endswith(")"):
        suffix = suffix[1:-1]
    parts = [p.strip() for p in suffix.split(",")]
    meaningful = [p for p in parts if p.lower() not in ("parsed", "full") and p]
    if meaningful:
        return ", ".join(meaningful)
    return "Whole program (solver decides between any built-in choices)"


_LABEL_FILLER_WORDS = {"subplan", "the", "and", "of", "path", "in", "for", "track"}


def _label_words(s: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z]+", s) if w.lower() not in _LABEL_FILLER_WORDS]


def _label_squash(s: str) -> str:
    """'AI Subplan' -> 'ai'; 'Artificial Intelligence Subplan' -> 'artificialintelligence'."""
    return "".join(_label_words(s)).lower()


def _label_initials(s: str) -> str:
    """'Artificial Intelligence Subplan' -> 'ai'."""
    return "".join(w[0] for w in _label_words(s)).lower()


def _same_variant(existing_suffix: str, track: str) -> bool:
    """True if `track` (a newly-discovered track/subplan name) is almost
    certainly the same thing as `existing_suffix` (an already-registered
    variant's display suffix) under a different phrasing -- either a plain
    substring match, or one is exactly the initials of the other's
    significant words (catches abbreviation pairs like "AI subplan" vs.
    "Artificial Intelligence Subplan", confirmed as a real duplicate this
    project produced before this check existed). Deliberately conservative
    -- e.g. "Math/Comp subplan" vs. "Mathematics and Computation Subplan"
    still registers as two separate (if redundant) entries rather than
    risk a false-positive match hiding a genuinely different subplan
    somewhere across 200+ programs this isn't manually reviewing."""
    a, b = existing_suffix.lower(), track.lower()
    if a in b or b in a:
        return True
    return (
        _label_squash(existing_suffix) == _label_initials(track)
        or _label_initials(existing_suffix) == _label_squash(track)
    )


# --- Internal "pick exactly one of these named alternatives" choice points
# --- inside an already-parsed rule tree (distinct from, and a level deeper
# --- than, list_track_options()'s SECTION-title-based subplan detection).
#
# Confirmed via a real cross-program check (prompted directly by a user
# noticing CS-BS only exposed one real subplan): several programs render
# their subplan/emphasis/track choice as a labeled sub-list INSIDE one
# ordinary section, rather than as their own top-level bulletin sections --
# e.g. ANTHR-BA's single "Depth in Discipline" section contains an
# at_least_n(1) rule ("Department Emphasis Courses") whose 3 sub_rules are
# "Cultural & Social Anthropology Emphasis" / "Environmental Anthropology
# Emphasis" / "Medical Anthropology Emphasis", each with its own real
# course list -- section-title matching alone can never see this, since
# there's only one section title ("Depth in Discipline") to begin with.
# Broadening is_subplan_section()'s keyword regex wouldn't have helped
# either (confirmed by inspecting parsed section titles directly: no
# section is ever TITLED "Emphasis" or "Concentration" for these programs
# -- the words only show up on the sub_rules one level down). This walks
# the RULE TREE instead of section titles, so it catches the shape
# directly regardless of what the enclosing section happens to be called.
def _rule_has_real_courses(rule: RequirementRule) -> bool:
    if rule.courses:
        return True
    return any(_rule_has_real_courses(sr) for sr in rule.sub_rules)


# Spot-checking real output across 50 programs turned up two flavors of
# parser noise that technically fit the structural at_least_n(1)-of-
# labeled-sub_rules shape but would make a nonsensical entry in a
# user-facing "which subplan?" dropdown:
#   - bulletin_parser.py falling back to "Unnamed Rule" (its own documented
#     placeholder label when a section has no real heading) or leaving a
#     grading-basis/policy fragment as a label (e.g. "Letter grade:") --
#     always ends up looking distinctly unlike a real track name: either
#     the literal fallback string, or text ending in ":" (every real
#     subplan/track/emphasis name confirmed across this sweep is a plain
#     noun phrase, never colon-terminated).
#   - a single course accidentally parsed as its own "alternative" (e.g.
#     "AA136B - Spacecraft Design Laboratory" sitting next to
#     "Aeronautics and Astronautics" as if they were two different tracks)
#     -- recognizable because it starts with a course-code-shaped prefix
#     (2+ consecutive capital letters immediately followed by a digit,
#     e.g. "AA136B", "CS106A"), which no real track/subplan/emphasis name
#     in this corpus starts with.
_JUNK_CHOICE_LABELS = {"unnamed rule"}
_COURSE_CODE_PREFIX_RE = re.compile(r"^[A-Z]{2,}[A-Z&]*\s*\d")


def _looks_like_real_choice_label(label: str) -> bool:
    s = label.strip()
    if not s or s.lower() in _JUNK_CHOICE_LABELS:
        return False
    if s.endswith(":"):
        return False
    if _COURSE_CODE_PREFIX_RE.match(s):
        return False
    return True


def find_choice_points(rules: list[RequirementRule]) -> list[RequirementRule]:
    """Every at_least_n(exactly 1)-of-2+-distinctly-labeled-sub_rules rule
    anywhere in `rules` (recursing into sub_rules) where at least 2 of
    those alternatives actually carry real course content -- i.e. a
    genuine "commit to exactly one of these named paths" branch point, the
    same shape CHEM-BS's hand-fixed "Chemistry Pathway" already models by
    hand. Deliberately narrow: requires n == 1 exactly (not "at least 1 of
    several, more is fine") and no course list of its own mixed in with
    the sub_rules (a rule that's PARTLY a flat list and partly named
    alternatives isn't this shape), so this only fires for real
    either/or/or commitments, not generic elective pools. Also filters out
    the two parser-noise label shapes documented above -- both the choice
    point's own label and every one of its alternatives' labels have to
    look like real names."""
    found: list[RequirementRule] = []

    def walk(rule: RequirementRule):
        labels = [sr.label for sr in rule.sub_rules]
        if (
            rule.kind == "at_least_n"
            and (rule.n or 1) == 1
            and len(rule.sub_rules) >= 2
            and not rule.courses
            and all(labels)
            and len(set(labels)) == len(labels)
            and _looks_like_real_choice_label(rule.label)
            and all(_looks_like_real_choice_label(l) for l in labels)
            and sum(1 for sr in rule.sub_rules if _rule_has_real_courses(sr)) >= 2
        ):
            found.append(rule)
        for sr in rule.sub_rules:
            walk(sr)

    for r in rules:
        walk(r)
    return found


def list_choice_points(code: str) -> list[tuple[str, list[str]]]:
    """[(choice_point_label, [alternative_label, ...]), ...] for a program,
    parsed at its default scope (no subplan opted in -- a choice point
    living inside an actual subplan section is out of scope here, since
    none of the confirmed real examples are nested that way). Empty for
    any program with no such internal choice point, or with no sample file
    at all -- same "just tell you there's nothing here" contract as
    list_track_options()."""
    path = SAMPLES_DIR / f"{code}.txt"
    if not path.exists():
        return []
    prog = parse_program_text(path.read_text(), code)
    rules = rules_from_parsed_program(prog)
    return [(r.label, [sr.label for sr in r.sub_rules]) for r in find_choice_points(rules)]


def _replace_choice_point(rules: list[RequirementRule], choice_label: str, chosen_label: str) -> bool:
    """Mutate `rules` in place: find the choice-point rule named
    `choice_label` anywhere in the tree and replace it with just the
    `chosen_label` alternative's own content (its sub_rules if it has any,
    otherwise the alternative itself) -- same "flatten out the one branch
    you want, drop the pick-one wrapper" idea already used by hand for
    CHEM-BS's Traditional/Biological Chemistry Path split, just applied
    generically via tree search instead of hand-transcription. Returns
    whether it actually found and replaced anything (callers should treat
    "found nothing" as a bug worth surfacing, not a silent no-op)."""

    def walk(rule_list: list[RequirementRule]) -> bool:
        for i, r in enumerate(rule_list):
            if r.label == choice_label and r.kind == "at_least_n":
                chosen = next((sr for sr in r.sub_rules if sr.label == chosen_label), None)
                if chosen is None:
                    return False
                rule_list[i:i + 1] = list(chosen.sub_rules) if chosen.sub_rules else [chosen]
                return True
            if walk(r.sub_rules):
                return True
        return False

    return walk(rules)


def build_program_choice_variant(
    code: str, choice_label: str, chosen_label: str
) -> tuple[list[RequirementRule], dict[str, str]]:
    """One program, with one internal choice point (see find_choice_points)
    pinned to a specific alternative instead of left for the solver to
    pick between."""
    rules, placeholders = build_program_from_sample(code)
    _replace_choice_point(rules, choice_label, chosen_label)
    return rules, placeholders


# Auto-discover every real named subplan/track (list_track_options() finds
# these directly in a program's raw sample text, independent of whether
# anyone has hand-registered a PROGRAM_SHEET_BUILDERS entry for it yet) and
# register any not already covered above. Without this, a program with
# several real subplans but only one or two hand-picked as named entries
# (confirmed for real: DATSC-BS has a "Quantitative Finance Subplan" with
# no registry entry at all; ENVSE-BS is missing "Sustainable Coastal
# Systems" and "Sustainable Freshwater Systems"; SLAV-BA is missing
# "Slavic Studies Subplan") would leave those subplans reachable only
# through the generic "{code} (parsed, full)" fallback -- which, per
# rules_from_parsed_program()'s own docstring, means "no subplan
# requirements included at all", silently understating the real major for
# anyone on one of the un-registered tracks. A track name only skips
# registration here if an existing label for the same code already
# matches it per _same_variant() -- e.g. CS-BS's own hand-registered "AI
# subplan" already covers "Artificial Intelligence Subplan", so this loop
# adds nothing extra for CS-BS.
#
# The second loop does the same thing one level deeper, for internal
# choice points (see find_choice_points() above) -- skipped entirely for
# any code in _MANUAL_FIXES, since those bypass build_program_from_sample()
# on purpose (they exist BECAUSE the generic parse is known-wrong for that
# specific program -- CHEM-BS's own hand-fixed Traditional/Biological split
# already covers this ground correctly for that one program, and
# auto-discovering a second, generically-parsed variant on top would risk
# resurrecting the exact parser bug the hand-fix was written to route
# around).
for _code in sorted({_code_of(_name) for _name in PROGRAM_SHEET_BUILDERS}):
    for _track in list_track_options(_code):
        _existing_suffixes = [
            variant_label(_code, _name)
            for _name in PROGRAM_SHEET_BUILDERS
            if _code_of(_name) == _code
        ]
        if any(_same_variant(_s, _track) for _s in _existing_suffixes):
            continue
        _label = f"{_code} (parsed, full, {_track})"
        PROGRAM_SHEET_BUILDERS.setdefault(
            _label, (lambda code=_code, track=_track: build_program_track(code, track))
        )

for _code in sorted({_code_of(_name) for _name in PROGRAM_SHEET_BUILDERS}):
    if _code in _MANUAL_FIXES:
        continue
    for _choice_label, _alt_labels in list_choice_points(_code):
        for _alt_label in _alt_labels:
            _existing_suffixes = [
                variant_label(_code, _name)
                for _name in PROGRAM_SHEET_BUILDERS
                if _code_of(_name) == _code
            ]
            if any(_same_variant(_s, _alt_label) for _s in _existing_suffixes):
                continue
            _label = f"{_code} (parsed, full, {_alt_label})"
            PROGRAM_SHEET_BUILDERS.setdefault(
                _label,
                (lambda code=_code, choice=_choice_label, alt=_alt_label: build_program_choice_variant(code, choice, alt)),
            )

# Now that every real named subplan/choice-point variant has been
# registered above (which needed each code's bare entry to exist first --
# see the registration loop's own comment), drop the bare "{CODE} (parsed,
# full)" entry for ANY program that has at least one real named
# alternative, whether or not the bare entry would actually still model a
# genuine, complete choice.
#
# An earlier version of this only removed the bare entry for the "mixed"
# programs where it was actively wrong (see _bare_entry_would_omit_
# subplans()'s docstring, still used below for BROKEN_PROGRAMS bookkeeping)
# -- e.g. CS-MS, ENGR-BS -- and left it in place for a program like
# ANTHR-BA, where the underlying choice is a single embedded pick-one-of-N
# list and the bare entry genuinely does still let CP-SAT choose the best
# option on its own. That distinction is real (see the two loops above:
# one models each subplan as a fully separate section, the other keeps
# every option in one list CP-SAT can search over directly), but on
# reflection it's not a distinction worth surfacing in the picker itself --
# a program with real, named alternatives should always ask a student to
# pick one explicitly, not offer a shortcut that only sometimes means what
# it says. So this generalizes to every code with any sibling entry,
# correct-and-free ones included. CS-BS and DATSC-BS already never had
# this entry at all (see _SPECIALLY_HANDLED_CODES); this just makes every
# other program with real alternatives consistent with them.
#
# NOTE: AFRAM-BA and ED-MA both have a BROKEN_PROGRAMS entry that used to
# be keyed to their now-removed bare label -- both were confirmed broken
# for a reason common to every one of their named variants too (see
# BROKEN_PROGRAMS's own comment), so their entries were moved to each real
# variant label instead of being silently lost when the bare key
# disappeared. Any code newly affected by broadening this from "mixed
# only" to "any sibling" needs the same check before its bare key (if it
# ever carried a BROKEN_PROGRAMS entry) is removed -- confirmed none of
# the other 6 BROKEN_PROGRAMS codes have any named variant at all, so this
# doesn't apply to them.
for _code in sorted({_code_of(_name) for _name in PROGRAM_SHEET_BUILDERS}):
    _bare_key = f"{_code} (parsed, full)"
    _has_sibling = any(
        _name != _bare_key and _code_of(_name) == _code for _name in PROGRAM_SHEET_BUILDERS
    )
    if _bare_key in PROGRAM_SHEET_BUILDERS and _has_sibling:
        del PROGRAM_SHEET_BUILDERS[_bare_key]


def list_variant_labels_by_code() -> dict[str, dict[str, str]]:
    """Every selectable program code -> {clean variant label: full
    PROGRAM_SHEET_BUILDERS key}, for app.py's two-step picker (pick a code,
    then pick which variant of it). Excludes nothing here -- filtering out
    BROKEN_PROGRAMS entries is the caller's job, same as the old flat
    picker already did, since "broken" is about the label, and a code can
    have some broken variants and some working ones."""
    by_code: dict[str, dict[str, str]] = {}
    for full_name in PROGRAM_SHEET_BUILDERS:
        code = _code_of(full_name)
        by_code.setdefault(code, {})[variant_label(code, full_name)] = full_name
    return by_code


def build_requirements(
    program_sheet_names: list[str], out_placeholders: dict[str, str] | None = None
) -> list[RequirementRule]:
    """Build the combined rule list for a set of program sheets. If
    `out_placeholders` is passed, it's updated in place with every
    placeholder course (key -> description) any parsed sheet needed (see
    bulletin_parser.py / models.make_placeholder_course) -- merge those
    into the course dict via course_data.add_placeholder_courses() before
    calling the solver, or those keys won't be schedulable. Hand-encoded
    builders (which return a plain list, no placeholders possible) work
    unchanged."""
    grouped = build_requirements_grouped(program_sheet_names, out_placeholders=out_placeholders)
    rules: list[RequirementRule] = []
    for name in program_sheet_names:
        rules.extend(grouped.get(name, []))
    return rules
