"""
Combining TWO undergraduate majors (double major, or dual/double degree),
or one major plus a secondary major -- see data/university_requirements.json's
"secondary_major"/"double_major"/"double_degree" sections for the underlying
policy this module encodes.

Scope, deliberately narrow (same "don't guess, flag it" philosophy as the
rest of this project):
  - Only -BS/-BA (undergraduate) program sheets are covered here. Stacking
    two graduate programs (-MS/-MA) or an undergrad major with a coterm
    master's has its own separate, not-yet-modeled policy (see
    engine/university_requirements.py's module docstring, which already
    flags coterm/double-degree unit-sharing as future "multi-program
    stacking" work) -- this module is that work, but only for the
    undergrad-undergrad case the university_requirements.json data
    actually covers. Selecting two -MS/-MA sheets (or a major + minor)
    still works exactly as it always has -- a plain union of rules, full
    double-counting, no stacking-mode selector -- this module just adds a
    NEW, opt-in mode for the specific undergrad-major-pair case.
  - Stanford doesn't support triple-majoring, so this only ever reasons
    about a PAIR of majors, never three or more.
  - The three modes differ from each other in exactly two solver-relevant
    ways: how many total units count toward graduation (180 vs 225), and
    whether double-counting the same course across both majors' elective
    pools is allowed at all. Diploma count/listing (the other real
    difference between "secondary major" and "double major") has no
    scheduling consequence and isn't modeled.
  - The "double counting not allowed" modes still have a real, bulletin-
    documented exception: "Intro-level shared requirements (e.g. intro
    math/language) and School of Engineering school-wide requirements may
    still be shared." The bulletin's own example only names math and
    language, but intro physics is the exact same shape of thing -- a
    foundational sequence (Mechanics/E&M/etc.) that many engineering and
    science majors separately list as a requirement, and that two stacked
    majors would routinely expect to share one enrollment for, same as
    calculus. There's no enumerated official list of exactly which courses
    any of this covers, so is_shared_exception_course() below is a best-
    effort, documented allowlist (intro calculus/CME sequence, intro
    physics sequence, first-two-years' language sequences, and the ENGR
    subject as a proxy for Engineering school-wide requirements) --
    surfaced to the user in the app rather than silently assumed, same as
    every other heuristic in this codebase.
"""
from __future__ import annotations

import re

from .models import Course, RequirementRule
from .university_requirements import load_university_requirements

UNDERGRAD_MAJOR_SUFFIXES = ("BS", "BA")
MINOR_SUFFIX = "MIN"
GRAD_SUFFIXES = ("MS", "MA")


def sheet_code(sheet_name: str) -> str:
    """'CS-BS (core subset, v1, hand-encoded)' -> 'CS-BS'."""
    return sheet_name.split(" (", 1)[0]


def sheet_kind(sheet_name: str) -> str:
    """'undergrad_major' | 'minor' | 'grad' | 'other', based on the sheet
    code's trailing -BS/-BA/-MIN/-MS/-MA convention (see engine/requirements.py's
    PROGRAM_SHEET_BUILDERS registry -- every entry follows it)."""
    code = sheet_code(sheet_name)
    suffix = code.rsplit("-", 1)[-1] if "-" in code else ""
    if suffix in UNDERGRAD_MAJOR_SUFFIXES:
        return "undergrad_major"
    if suffix == MINOR_SUFFIX:
        return "minor"
    if suffix in GRAD_SUFFIXES:
        return "grad"
    return "other"


def is_undergrad_major_sheet(sheet_name: str) -> bool:
    return sheet_kind(sheet_name) == "undergrad_major"


def degree_type_of(sheet_name: str) -> str:
    """'BA' or 'BS' (or '' if not an undergrad major sheet) -- used for
    double_degree's "one BA + one BS" check."""
    code = sheet_code(sheet_name)
    suffix = code.rsplit("-", 1)[-1] if "-" in code else ""
    return suffix if suffix in UNDERGRAD_MAJOR_SUFFIXES else ""


# ---------------------------------------------------------------------------
# Stacking modes
# ---------------------------------------------------------------------------

STACK_MODE_SECONDARY = "secondary_major"
STACK_MODE_DOUBLE_MAJOR = "double_major"
STACK_MODE_DOUBLE_DEGREE = "double_degree"


def load_stack_modes(req: dict | None = None) -> dict[str, dict]:
    """One entry per mode, pulled straight from university_requirements.json
    (no numbers invented here) -- label for the UI, total_units for the
    informational graduation check, double_counting_allowed for whether
    solver.py's attribution mechanism engages at all, and
    requires_ba_and_bs for double_degree's own documented requirement."""
    req = req or load_university_requirements()
    sm = req["secondary_major"]
    dm = req["double_major"]
    dd = req["double_degree"]
    return {
        STACK_MODE_SECONDARY: {
            "label": "Secondary major (1 diploma, primary major only listed on it; "
                     "double-counting between majors allowed)",
            "total_units": sm["total_units_required"],
            "double_counting_allowed": sm["double_counting_with_primary_major_allowed"],
            "requires_ba_and_bs": False,
        },
        STACK_MODE_DOUBLE_MAJOR: {
            "label": "Double major (1 diploma, both majors listed on it; "
                     "double-counting between majors NOT allowed, except shared "
                     "intro-level/Engineering-school-wide courses)",
            "total_units": dm["total_units_required"],
            "double_counting_allowed": dm["double_counting_between_majors_allowed"],
            "requires_ba_and_bs": False,
        },
        STACK_MODE_DOUBLE_DEGREE: {
            "label": "Double degree (2 diplomas, one BA + one BS; "
                     "double-counting between majors NOT allowed, except shared "
                     "intro-level/Engineering-school-wide courses)",
            "total_units": dd["total_units_required"],
            "double_counting_allowed": dd["double_counting_between_majors_allowed"],
            "requires_ba_and_bs": True,
        },
    }


# ---------------------------------------------------------------------------
# Shared-exception course allowlist (best-effort, see module docstring)
# ---------------------------------------------------------------------------

# Stanford's standard intro calculus/linear-algebra sequence, confirmed
# against the real course snapshot (same codes CS-BS's own hand-encoded
# Math Electives rule already uses) -- the canonical "intro math" most
# programs' own required-course lists route through.
_INTRO_MATH_KEYS = frozenset({
    "MATH19", "MATH20", "MATH21", "MATH41", "MATH42",
    "MATH51", "MATH52", "MATH53",
    "CME100", "CME102", "CME104",
})

# Stanford's standard intro physics sequence(s) -- confirmed against the
# real course snapshot as exactly the same shape of "foundational sequence
# almost every engineering/science major requires" that intro math is, and
# just as commonly listed in TWO different majors' own required-course/
# elective-pool lists at once (e.g. EE-BS and MATSC-BS both draw on
# PHYSICS41/43/45/61/71/81 in their own rule trees -- confirmed directly,
# not theoretical). Covers the algebra-based sequence (21/22, 23/24, 25/26),
# the standard calc-based sequence (41/41E/42, 43/44, 45/46), and the two
# honors/alternate tracks (61/61L "Mechanics and Special Relativity", 71
# "Quantum and Thermal Physics", 81/81L "E&M Using Special Relativity and
# Vector Calculus") -- deliberately NOT every low-numbered PHYSICS course
# (excludes e.g. PHYSICS16/11N/13N/14N/19N, which are one-off frosh
# seminars/topics courses a major would never actually require, not part
# of any foundational sequence).
_INTRO_PHYSICS_KEYS = frozenset({
    "PHYSICS21", "PHYSICS22",
    "PHYSICS23", "PHYSICS24",
    "PHYSICS25", "PHYSICS26",
    "PHYSICS41", "PHYSICS41E", "PHYSICS42",
    "PHYSICS43", "PHYSICS44",
    "PHYSICS45", "PHYSICS46",
    "PHYSICS61", "PHYSICS61L",
    "PHYSICS71",
    "PHYSICS81", "PHYSICS81L",
})

# Stanford's standard intro chemistry sequence -- user-caught gap (found via
# a real CHEM-BS + CS-BS double-major validation: CHEM31A/CHEM31B/CHEM31E
# and CHEM33 showed up as "restricted" between the two majors' rule trees
# before this was added, since both CHEM-BS's own "Chem 30 Series"/
# "Additional Courses" and CS-BS's Science Elective pool reference them --
# exactly the same "foundational sequence two majors would routinely expect
# to share one enrollment for" shape as intro math/physics above, per
# Stanford's own stated exception for "intro-level shared requirements").
_INTRO_CHEM_KEYS = frozenset({
    "CHEM31A", "CHEM31B", "CHEM31E", "CHEM33",
})

# First-year/second-year language-sequence course codes end in a *LANG
# subject (e.g. FRENLANG, SPANLANG, CHINLANG -- confirmed against the real
# snapshot; upper-division literature/culture courses live under a
# DIFFERENT, non-LANG subject like FRENCH/GERMAN/JAPAN) with a low course
# number. Not a claim about which specific courses satisfy the university
# language requirement (see university_requirements.json's "language"
# section -- that's a separate, manually-input mechanism); this is only
# about which language courses are allowed to double-count between two
# majors per the bulletin's own "intro-level ... language" exception text.
_LANG_SUBJECT_RE = re.compile(r"LANG$")
_LANG_INTRO_MAX_NUMBER = 29  # covers 1,1A,2,2A,3,10-15,21-23 and their suffixed variants; 100+ is upper-division
_COURSE_NUMBER_RE = re.compile(r"^(\d+)")


def _is_intro_language_key(key: str, courses: dict[str, Course]) -> bool:
    course = courses.get(key)
    if course is None or not _LANG_SUBJECT_RE.search(course.subject):
        return False
    m = _COURSE_NUMBER_RE.match(course.code)
    return bool(m) and int(m.group(1)) <= _LANG_INTRO_MAX_NUMBER


def is_shared_exception_course(key: str, courses: dict[str, Course]) -> bool:
    """True if `key` falls under the bulletin's documented "still shareable"
    carve-out (intro math, intro physics, intro language, or Engineering
    school-wide via the ENGR subject as a proxy) -- see module docstring
    for why this is a best-effort allowlist, not an official enumerated
    list."""
    if key in _INTRO_MATH_KEYS or key in _INTRO_PHYSICS_KEYS or key in _INTRO_CHEM_KEYS:
        return True
    course = courses.get(key)
    if course is not None and course.subject == "ENGR":
        return True
    return _is_intro_language_key(key, courses)


# ---------------------------------------------------------------------------
# Finding which courses are actually double-count-restricted between two
# stacked majors
# ---------------------------------------------------------------------------

def _is_and_group(item) -> bool:
    return isinstance(item, tuple) and bool(item) and item[0] == "__and__"


def _course_keys_in_item(item) -> set[str]:
    if _is_and_group(item):
        return {c for c in item[1:] if isinstance(c, str)}
    if isinstance(item, (tuple, list)):
        return {c for c in item if isinstance(c, str)}
    return {item} if isinstance(item, str) else set()


def _pool_course_keys(rule: RequirementRule) -> set[str]:
    """Every course key that sits in an actual CHOICE slot -- i.e. one
    among several interchangeable alternatives -- anywhere in this rule and
    its sub_rules. Two shapes count:
      1. An OR-alternatives item (e.g. ("MATH51","MATH61CM","MATH61DM")) --
         counts regardless of which kind of rule it sits inside, since the
         item itself is the choice, whether that item is one of several
         inside a "pick N of these" at_least_n rule OR just one slot among
         several specifically-named requirements in an "all" rule (e.g.
         CHEM-BS's Foundational Courses is kind="all" -- CHEM121, CHEM131,
         ... are each individually mandatory, but the math slot within it
         is still "MATH51 OR MATH61CM OR MATH61DM", a real choice).
      2. Every course referenced by a genuine "pick N of M" rule
         (kind="at_least_n" with more than one real option, or
         kind="unit_min") -- even a BARE single-course item counts here,
         since the choice is at the rule level (which N of the M do you
         take), not necessarily inside any one item.
    Deliberately excludes a bare single-course item sitting directly in an
    "all" rule (not part of any tuple): that's a specifically-named
    required course with no alternative at all, exactly the "intro-level
    shared requirement" shape the bulletin's own exception text is talking
    about. See engine/solver.py's matching choice to only gate
    at_least_n/unit_min pools (plus OR-tuples, via indicator_sum's
    per-alt attribution) with attribution variables, never a bare "all"
    item, for the same reasoning applied to enforcement."""
    keys: set[str] = set()

    def walk(r: RequirementRule):
        is_real_pool_rule = (
            (r.kind == "at_least_n" and (len(r.courses) + len(r.sub_rules)) > 1)
            or r.kind == "unit_min"
        )
        for item in r.courses:
            is_or_alt_group = isinstance(item, (tuple, list)) and len(item) > 1 and not _is_and_group(item)
            if is_real_pool_rule or is_or_alt_group:
                keys.update(_course_keys_in_item(item))
        for sr in r.sub_rules:
            walk(sr)

    walk(rule)
    return keys


def find_double_count_restricted_keys(
    rules_a: list[RequirementRule],
    rules_b: list[RequirementRule],
    courses: dict[str, Course],
) -> set[str]:
    """Course keys that appear in an elective pool in BOTH programs' rule
    trees, minus the shared-exception allowlist -- these are the ones
    solver.py's attribution mechanism needs to gate so the same physical
    enrollment can't satisfy a counted slot in both majors at once."""
    keys_a = set().union(*(_pool_course_keys(r) for r in rules_a)) if rules_a else set()
    keys_b = set().union(*(_pool_course_keys(r) for r in rules_b)) if rules_b else set()
    shared = keys_a & keys_b
    return {k for k in shared if not is_shared_exception_course(k, courses)}


def tag_program_id(rules: list[RequirementRule], program_id: str) -> list[RequirementRule]:
    """Mutate every TOP-LEVEL rule's .program_id in place (sub_rules don't
    need it -- solver.py's recursion carries the parent's program_id down)
    and return the same list, for convenient chaining. Safe to mutate: every
    PROGRAM_SHEET_BUILDERS entry builds a fresh list of fresh RequirementRule
    objects on each call (confirmed -- none of the hand-encoded, parsed, or
    manual-fix builders return a shared module-level constant), so this
    never leaks into another request/session."""
    for r in rules:
        r.program_id = program_id
    return rules


def build_restricted_course_programs(
    program_a: str,
    rules_a: list[RequirementRule],
    program_b: str,
    rules_b: list[RequirementRule],
    courses: dict[str, Course],
) -> dict[str, list[str]]:
    """Convenience wrapper: the exact restricted_course_programs shape
    engine/solver.py's build_and_solve() expects, for exactly the one pair
    of programs this module supports stacking."""
    restricted = find_double_count_restricted_keys(rules_a, rules_b, courses)
    return {k: [program_a, program_b] for k in restricted}
