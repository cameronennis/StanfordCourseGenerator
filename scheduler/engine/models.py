"""
Core data structures shared across the scheduler engine.

Nothing here is specific to any one student -- CompletedCourse instances
come from parsing *a* transcript (see transcript.py), and WeightRule
instances come from *a* user's weight table (see weights.py). Any person's
data can flow through the same models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# All quarters we ever plan across. "Summer" included since some courses
# (e.g. MATSCI 150 undergrad research) run then, even though most students
# skip it.
QUARTER_NAMES = ("Autumn", "Winter", "Spring", "Summer")


@dataclass
class MeetingPattern:
    """A sampled real meeting pattern for a course, used only as an estimate
    for future quarters -- explorecourses only has confirmed times for
    quarters that have already been scheduled, not ones years out."""
    quarter: str          # e.g. "2026-2027 Winter"
    days: tuple[str, ...]  # e.g. ("Monday", "Wednesday", "Friday")
    start_time: str        # e.g. "12:30:00 PM"
    end_time: str


@dataclass
class Course:
    subject: str
    code: str
    title: str
    units_min: float
    units_max: float
    terms_offered: tuple[str, ...]   # season names this course has appeared in, e.g. ("Autumn","Winter","Spring")
    description: str = ""
    sample_meetings: tuple[MeetingPattern, ...] = field(default_factory=tuple)
    # Prerequisites/corequisites are NOT structured data in explorecourses --
    # they only exist as free text inside `description`. These four fields
    # are a heuristic, best-effort extraction (see course_data.py's
    # extract_prereq_text): the *_text fields are whatever sentence(s) in the
    # description mention "prerequisite"/"corequisite" verbatim, and the
    # *_courses fields are course-code-shaped tokens (e.g. "CS106A") found
    # inside that text. Treat both as a starting point to review, not ground
    # truth -- free text like "or equivalent", "CS 107 or above", or codes
    # split across a slash (e.g. "AA 171/274") won't be parsed correctly.
    prerequisites_text: str = ""
    corequisites_text: str = ""
    prerequisite_courses: tuple[str, ...] = field(default_factory=tuple)
    corequisite_courses: tuple[str, ...] = field(default_factory=tuple)
    # Set only when the "typical" unit count isn't simply units_max (e.g. a
    # placeholder course displaying a 3-5 unit range to the user, but using
    # a single fixed value -- 4 -- for the solver's unit caps/load-balance
    # math under the hood). None for every real, explorecourses-derived
    # Course, which keeps their existing units_max-based behavior unchanged.
    units_typical_override: Optional[float] = None
    # WAYS (Ways of Thinking / Ways of Doing) category codes this course is
    # certified for, e.g. ("ER", "EDP") for a course that counts toward both
    # Ethical Reasoning and Exploring Difference and Power -- straight from
    # explorecourses' own `gers` field (its raw values look like "WAY-ER";
    # the "WAY-" prefix is stripped before storing here), NOT a heuristic
    # extraction like prerequisites -- this is real, structured university
    # data, confirmed by spot-checking real courses. Empty for the large
    # majority of courses that aren't WAYS-certified at all. See
    # data/university_requirements.json's "ways" section for what each code
    # means and how many of each a student needs.
    ways_categories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.subject}{self.code}".upper().replace(" ", "")

    @property
    def units_typical(self) -> float:
        if self.units_typical_override is not None:
            return self.units_typical_override
        return self.units_max or self.units_min


# Some bulletin requirements name no specific courses at all -- either
# genuinely free text ("six math courses at or above Math 51") or an
# unenumerated pool ("any MATH course numbered 100 and above"). bulletin_
# parser.py can't invent which real courses satisfy those, so instead it
# creates a PLACEHOLDER course standing in for the requirement: schedulable
# like a real course (so the plan stays complete and the gap is visible),
# but clearly marked so it's obvious a manual override is needed before
# trusting it. See bulletin_parser.py's ParseWarning for the corresponding
# "I made one of these, go check it" signal.
PLACEHOLDER_PREFIX = "PLACEHOLDER:"


def is_placeholder_key(key: str) -> bool:
    return key.startswith(PLACEHOLDER_PREFIX)


def make_placeholder_course(
    key: str,
    description: str,
    units_min: float = 3.0,
    units_max: float = 5.0,
    units_typical: float = 4.0,
) -> "Course":
    """Build a stand-in Course for a placeholder key (see PLACEHOLDER_PREFIX).
    `key` becomes the Course's own .key unchanged (subject=key, code="" --
    Course.key is just subject+code), so it can be looked up in a course
    dict exactly as bulletin_parser.py / RequirementRule.courses reference
    it. There's no real course here to read a unit count from, so this is
    a rough guess (most Stanford courses run 3-5 units): the UI shows the
    3-5 range (units_min/units_max), while the solver's unit caps and
    load-balance math use a single fixed 4 (units_typical_override) so
    they have one concrete number to work with."""
    return Course(
        subject=key,
        code="",
        title=f"[Placeholder] {description}",
        units_min=units_min,
        units_max=units_max,
        units_typical_override=units_typical,
        terms_offered=QUARTER_NAMES,  # unknown offering pattern -- assume always available
        description=(
            "Stand-in for a degree requirement whose bulletin text didn't list specific "
            "courses. Replace with a real course that satisfies: " + description
        ),
    )


@dataclass
class CourseOverride:
    """A manual correction to a snapshotted course, e.g. 'this is also
    offered in Spring' or 'the real prerequisite is CS106B not CS106A'.
    Kept separate from the snapshot itself so re-fetching course data never
    silently discards a person's corrections."""
    course_key: str
    field: str              # "terms_offered" | "units_min" | "units_max" | "title" | ...
    value: object
    note: str = ""


@dataclass
class RequirementRule:
    """One node in a (possibly nested) degree-requirement tree.

    kind:
      "all"        -- every course in `courses` is required
      "at_least_n" -- at least `n` of `courses` (or of the courses satisfying
                      the nested `sub_rules`) are required
      "unit_min"   -- at least `n` units total across `courses`
    `courses` holds course keys (e.g. "CS106B"); alternatives at the same
    slot (e.g. "CS107 OR CS107E") are represented as a tuple-within-the-list.
    """
    label: str
    kind: str
    courses: list[object] = field(default_factory=list)   # str key, or tuple of alternative keys
    n: Optional[int] = None
    unit_min: Optional[float] = None
    sub_rules: list["RequirementRule"] = field(default_factory=list)
    source: str = ""   # which program sheet / university requirement this came from
    # Which stacked program (short code, e.g. "CS-BS") this TOP-LEVEL rule
    # belongs to -- set only when 2+ undergrad majors are being planned
    # together under a "double major" or "double degree" (see
    # engine/multi_program.py), never for a single program or for the
    # default "secondary major" mode, where double-counting is explicitly
    # allowed and this stays "". solver.py reads this once per top-level
    # rule (sub_rules inherit it via the recursion, not by also setting it
    # on every nested node) to decide which enrollments get gated by a
    # per-program "attribution" variable instead of being freely shared --
    # see solver.py's restricted_course_programs parameter. Empty string
    # (the default) means "don't restrict anything for this rule," which
    # is exactly today's existing behavior for every caller that doesn't
    # know about program stacking.
    program_id: str = ""
    # True only for requirements whose repeated occurrences must land in
    # temporally CONTIGUOUS quarters (e.g. a language sequence -- you can't
    # take level 2 two years after level 1 and call it a coherent sequence).
    # At most one skipped quarter is tolerated between occurrences, and only
    # if that skipped quarter is Summer. This is distinct from an ordinary
    # solo "at_least_n(n>1)" placeholder (e.g. a WAYS category needing 2
    # independent courses), which has no temporal relationship between its
    # occurrences and must NOT get this constraint. solver.py's
    # _walk_repeat_eligibility() reads this to route matching courses into
    # a separate _sequential_repeat_courses set instead of the ordinary
    # _solo_repeat_courses set.
    sequential: bool = False
    # True only for a solo repeat-eligible rule (see `sequential` above)
    # whose occurrences must land in exactly one Autumn-Winter-Spring block
    # of the SAME academic year -- currently just the language requirement
    # when exactly 3 quarters are still needed (see build_language_rule in
    # engine/university_requirements.py). Stronger than `sequential` alone:
    # `sequential` just forbids a real skipped quarter (Summer excepted),
    # which would still allow e.g. Spring/Autumn/Winter split across two
    # academic years -- this instead pins the whole sequence to one
    # specific, real academic-year block starting in Autumn. Meaningless
    # (and unchecked) unless `sequential` is also True.
    full_year_sequence: bool = False


def _is_and_group(item: object) -> bool:
    return isinstance(item, tuple) and bool(item) and item[0] == "__and__"


def all_referenced_course_keys(rules: list["RequirementRule"]) -> set[str]:
    """Every course key that is UNCONDITIONALLY required somewhere in a
    (possibly nested) list of RequirementRule trees -- i.e. actually going
    to be scheduled regardless of anything else, not just one of several
    interchangeable options. Recurses into `sub_rules`. Includes
    placeholder keys too (a caller that only wants real courses should
    filter with is_placeholder_key itself).

    Two shapes are deliberately EXCLUDED, even though an earlier version
    of this function included them -- a real, user-reported overcrediting
    bug (see below):
      1. An OR-alternatives item (a tuple within `courses`, e.g.
         ("MATH51","MATH61CM","MATH61DM")) -- only ONE of these will
         actually be taken, so treating every alternative as "definitely
         happening" is wrong.
      2. Every course inside a genuine "pick N of M" pool (kind
         "at_least_n" with more real alternatives than N, or kind
         "unit_min") -- same reasoning one level up: which N (or how many,
         for unit_min) of the M get taken is a real, unresolved choice.
    An AND-group item (`("__and__", ...)`, e.g. AP credit's "CHEM31A AND
    CHEM31B together") is NOT an alternatives tuple -- when it sits outside
    a real pool rule, every course inside it is just as unconditional as a
    bare single-course item, so all of its members are included. A bare
    single-course item sitting directly in an "all" rule (or in a solo
    "at_least_n" rule -- exactly one alternative, however many times it
    must repeat) is the ordinary unconditional case.

    Built so university requirements (WAYS/COLLEGE/Writing -- see
    engine/university_requirements.py) can credit a course the student is
    DEFINITELY already required to take for some OTHER reason -- a program
    sheet's own specifically-named requirement, or their own manually-added
    course list (app.py's "2d" section) -- not just a course already
    sitting on their completed transcript. Real case this was built for: a
    manually-added CHEM131 already carries real WAYS tags, so a student
    taking it anyway shouldn't ALSO be asked to take a fully separate,
    generic WAYS placeholder for that credit.

    Real bug this stricter version fixes: a CHEM-BS + CS-BS double major's
    combined rule trees reference hundreds of CS elective alternatives
    inside "pick N of M" pools; the old, tuple-flattening version credited
    WAYS/Writing from WAYS-tagged courses buried in THAT kind of pool even
    though none of them are actually guaranteed to be taken -- making
    WAYS/Writing look satisfied by courses the plan never commits to."""
    keys: set[str] = set()
    for rule in rules:
        total_alts = len(rule.courses) + len(rule.sub_rules)
        is_real_pool_rule = (
            (rule.kind == "at_least_n" and total_alts > 1 and (rule.n or 0) < total_alts)
            or rule.kind == "unit_min"
        )
        if not is_real_pool_rule:
            for item in rule.courses:
                if _is_and_group(item):
                    keys.update(c for c in item[1:] if isinstance(c, str))
                elif isinstance(item, (tuple, list)):
                    continue  # OR-alternatives -- not one guaranteed course
                else:
                    keys.add(item)
        if rule.sub_rules:
            keys.update(all_referenced_course_keys(rule.sub_rules))
    return keys


@dataclass
class WeightRule:
    """One row of the tweakable parameter table.

    `type` selects a scoring function implemented in weights.py; `args` are
    its parameters. Adding a new *kind* of preference means adding one new
    function to weights.py -- using an existing kind with new arguments
    (e.g. a different subject or day) never requires new code.
    """
    type: str
    args: dict
    weight: float
    label: str = ""
