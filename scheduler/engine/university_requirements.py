"""
Stanford-wide requirements that apply to every undergraduate regardless of
major -- researched into data/university_requirements.json (WAYS, COLLEGE,
writing, language, units, GPA, residency, pass/fail, double major/secondary
major/double degree, coterm), none of which were wired into the solver
before this module existed. Each build_*_rules() function here returns a
list of RequirementRule the same shape engine/requirements.py's program-
sheet builders use, meant to be appended to whatever major/minor rules a
person selects (see app.py) -- these apply ON TOP, not instead.

What's covered here, and what deliberately isn't (see the project owner's
own scoping decisions before this module was built):

  - WAYS: fully covered, using REAL per-course category data
    (Course.ways_categories, sourced from explorecourses' own `gers`
    field) -- confirmed to be genuine structured university data, not a
    heuristic text extraction like prerequisites are.
  - COLLEGE: fully covered for both the 2-quarter version (current policy,
    through admit year 2026-2027) and a best-guess 3-quarter version
    (Fall 2027+ policy) -- Stanford's own announcement of the extension
    hadn't published the third quarter's exact course list as of this
    project's research, so that part is a clearly-flagged guess, not a
    verified rule.
  - Writing (WR1/WR2): covered via the real PWR1*/PWR2* topic-section
    pools (~80/~65 real alternative sections each) plus ITALIC95W. WIM
    (Writing in the Major) is NOT here -- it's already part of each
    major's own program-sheet rules in requirements.py, not a
    university-wide one. The real WR1-before-WR2 ordering is NOT
    enforced: this project's prerequisite mechanism (engine/solver.py)
    sequences one literal course before another, not one ~80-option pool
    before a ~65-option pool -- low real-world risk since advising
    already sequences this in practice.
  - Language: NOT modeled against real course data at all. Which
    language, and how many quarters are still needed, depends on a
    placement test or AP score this app has no data source for (the
    transcript parser doesn't capture either). Modeled instead as a
    manual "how many quarters do you still need" number the student
    supplies directly (see app.py), turned into a repeatable placeholder
    course -- see build_language_rule().
  - Units (180-unit graduation minimum), GPA, residency, and pass/fail
    rules: NOT modeled as solver constraints. The 180-unit minimum is
    checked informationally, after solving, in app.py rather than as a
    hard constraint -- forcing the solver to actively REACH a unit floor
    conflicts with its deliberate anti-filler bias (see solver.py's
    objective section), which needs a real "pool of acceptable filler
    electives" concept this project doesn't have yet. GPA/residency/
    pass-fail aren't schedule-relevant at all.
  - Double major / secondary major / double degree / coterm unit-sharing
    rules: NOT modeled here on purpose -- these are about how TWO
    programs interact with each other, not a flat university-wide rule,
    and belong with the not-yet-built multi-program-stacking work
    (see README) rather than this module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ortools.sat.python import cp_model

from .models import Course, RequirementRule, make_placeholder_course

UNIVERSITY_REQUIREMENTS_PATH = Path(__file__).parent.parent / "data" / "university_requirements.json"

_WR1_RE = re.compile(r"^PWR1[A-Z]")
_WR2_RE = re.compile(r"^PWR2[A-Z]")
_GLOBAL_PERSPECTIVES_RE = re.compile(r"^COLLEGE1(0[3-9]|1[0-9]|2[0-7])$")


def load_university_requirements(path: str | Path = UNIVERSITY_REQUIREMENTS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _max_ways_assignment(
    by_category: dict[str, list[str]], categories: dict
) -> tuple[int, dict[str, int]]:
    """The real answer to "how many of the 11 total WAYS courses has this
    transcript actually satisfied" -- a small bipartite b-matching (each
    completed course has capacity 1, i.e. can count toward AT MOST ONE
    category; each category has capacity equal to its own
    `courses_required`), solved via CP-SAT rather than a naive per-category
    tally.

    This replaces a real, user-reported bug: the previous version credited
    a dual-tagged completed course to EVERY category it's tagged for
    independently, which can OVER-count. Confirmed against a real
    transcript: COLLEGE119 is tagged both "EDP" and "SI" -- the old code
    let it satisfy WAYS-EDP (0 remaining) AND contribute to WAYS-SI at the
    same time, even though it's one enrollment and Stanford's real policy
    (per the user) only lets it count toward ONE category. The old code's
    "global floor" check (comparing the distinct completed-course count
    against the 11-total requirement) didn't catch this either -- it only
    protects the AGGREGATE total, not each category's own correctness, and
    in this exact case the aggregate total happened to look fine (11
    distinct completed WAYS-tagged courses >= 11 required) while a real
    category (EDP) was still incorrectly marked fully satisfied.

    Returns (max_total_assigned, one_example_per_category_assignment). The
    MAXIMUM TOTAL is always a single well-defined number (a basic
    max-matching fact: even when multiple equally-optimal assignments
    exist, their sizes are identical) -- confirmed by hand for the
    COLLEGE119 case above: crediting it to SI leaves (SI short 1, EDP short
    1); crediting it to EDP leaves (SI short 2, EDP short 0); either way
    the COMBINED shortfall across the two is 2, not 1 or 3. The
    PER-CATEGORY split returned alongside it is just ONE such optimal
    assignment, picked arbitrarily by the solver among ties -- it's
    included only as an illustrative example of a way to close the gap
    (see build_ways_rules below), never as a claim that a specific
    category is the "real" one a specific course counts toward."""
    model = cp_model.CpModel()
    y: dict[tuple[str, str], "cp_model.IntVar"] = {}
    for cat, keys in by_category.items():
        for k in keys:
            y[(k, cat)] = model.NewBoolVar(f"y_{k}_{cat}")
    if not y:
        return 0, {cat: 0 for cat in by_category}

    by_course: dict[str, list] = {}
    for (k, cat), var in y.items():
        by_course.setdefault(k, []).append(var)
    for vars_ in by_course.values():
        model.Add(sum(vars_) <= 1)  # each completed course counts toward AT MOST ONE category

    cat_sums = {}
    for cat, keys in by_category.items():
        vars_ = [y[(k, cat)] for k in keys if (k, cat) in y]
        cat_sum = sum(vars_) if vars_ else 0
        cat_sums[cat] = cat_sum
        if vars_:
            n_required = categories[cat]["courses_required"]
            # No point crediting a category beyond its own requirement --
            # doing so would just waste a course's credit that might be
            # usable elsewhere if it's tagged for 2+ categories.
            model.Add(cat_sum <= n_required)

    model.Maximize(sum(cat_sums.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 0, {cat: 0 for cat in by_category}  # pragma: no cover -- trivially solvable in practice
    per_category = {
        cat: (solver.Value(expr) if not isinstance(expr, int) else expr)
        for cat, expr in cat_sums.items()
    }
    return sum(per_category.values()), per_category


def build_ways_rules(
    courses: dict[str, Course],
    completed_keys: set[str] | None = None,
    req: dict | None = None,
    other_required_keys: set[str] | None = None,
) -> tuple[list[RequirementRule], dict[str, Course]]:
    """WAYS -- see data/university_requirements.json's "ways" section for
    the real 8 category names/counts and the 11-total-courses floor
    (sourced from ways.stanford.edu).

    Deliberately does NOT name a specific real course, or even commit to a
    specific CATEGORY breakdown, for what's still needed -- see
    models.make_placeholder_course and _max_ways_assignment's own
    docstring for why. With dozens (sometimes hundreds) of real courses
    satisfying a given category equally well, having the solver commit to
    one arbitrary specific title (e.g. one particular Social Inquiry
    elective out of ~100 real alternatives) reads as a firm pick rather
    than the open choice it actually is -- confirmed as a real, reported
    point of confusion, not a hypothetical one. A PER-CATEGORY breakdown
    has the same problem one level up: when a completed course is tagged
    for 2+ categories, which specific category "uses" it is a genuine free
    choice with no single correct answer (see _max_ways_assignment) -- so
    rather than presenting an arbitrary tie-broken split as if it were
    fact, this returns ONE combined rule for however many total courses
    are still needed across all 8 categories, which IS always a single,
    correct, assignment-independent number. The one example assignment
    _max_ways_assignment also returns is folded into the rule's `source`
    text purely as an illustrative "here's one way to close the gap," not
    as an enforced or definitive per-category requirement.
    """
    req = req or load_university_requirements()
    completed_keys = completed_keys or set()
    other_required_keys = other_required_keys or set()
    ways = req["ways"]
    min_units = ways.get("min_units_per_course", 3)
    categories = ways["categories"]
    total_required = ways["total_courses_required"]

    # Credit anything already completed, OR already required for some
    # OTHER reason in this same solve (a program sheet's own requirement,
    # or a manually-added course) -- either way, the student is taking it
    # regardless of WAYS, so a real WAYS tag on it shouldn't be ignored
    # just because it isn't on the transcript yet. The one-category-per-
    # course cap below (_max_ways_assignment's `sum(vars_) <= 1`) applies
    # identically either way -- a multi-tagged course (e.g. CHEM131,
    # tagged both AQR and SMA) still only ever counts toward ONE category,
    # whichever an optimal assignment picks, never both at once.
    _ways_credit_keys = completed_keys | other_required_keys
    by_category: dict[str, list[str]] = {cat: [] for cat in categories}
    for key in _ways_credit_keys:
        c = courses.get(key)
        if c is None or not c.ways_categories:
            continue
        # WAYS' "Creative Expression" (CE) category has a documented
        # exception ("may be met with two 1-unit courses in the same
        # department") that a flat pool-based rule can't express (it would
        # need to verify the pair comes from the same department) -- not
        # modeled; CE instead uses a slightly lower 2-unit floor so at
        # least ordinary 2+-unit CE courses are admitted, while every
        # other category uses the real min_units_per_course floor.
        floor = 2 if "CE" in c.ways_categories else min_units
        if c.units_typical < floor:
            continue
        for cat in c.ways_categories:
            if cat in by_category:
                by_category[cat].append(key)

    max_assigned, example_assignment = _max_ways_assignment(by_category, categories)
    total_remaining = max(0, total_required - max_assigned)
    if total_remaining == 0:
        return [], {}

    # Purely informational: which categories still show a gap under ONE
    # example optimal assignment, for the rule's own description -- see
    # this function's docstring on why this is an illustration, not a
    # per-category requirement in its own right.
    example_gaps = [
        f"{cat} ({categories[cat]['courses_required'] - example_assignment.get(cat, 0)} more)"
        for cat in sorted(categories)
        if example_assignment.get(cat, 0) < categories[cat]["courses_required"]
    ]
    # This goes in the LABEL, not just `source` -- app.py's rule-tree
    # display (_rule_lines) only ever prints label/kind/n/courses, never
    # `source`, so a note left only in `source` would be invisible in the
    # "University-wide rules being applied" panel a user actually reads.
    example_note = (
        f" (one way to close the gap: {'; '.join(example_gaps)} -- but if any of "
        "your completed or already-required courses is tagged for 2+ categories, "
        "which specific category it counts toward is your choice, not a fixed "
        "fact, so this exact split isn't the only option)"
        if example_gaps else ""
    )

    placeholder_key = "PLACEHOLDER:WAYS_REMAINING"
    placeholder = make_placeholder_course(
        placeholder_key,
        "WAYS: a real Stanford course covering any category you still need, your "
        "choice (see ways.stanford.edu for the 8 categories and real course "
        "lists). Left as one combined placeholder rather than naming a specific "
        "category, since which category a dual-tagged completed course counts "
        "toward is often a free choice, not a fixed fact -- see this "
        "requirement's own label for one example breakdown.",
        units_min=float(min_units), units_max=float(min_units), units_typical=float(min_units),
    )
    rule = RequirementRule(
        label=(
            f"WAYS: {total_required} total courses across 8 categories -- "
            f"{total_remaining} still needed{example_note}"
        ),
        kind="at_least_n",
        n=total_remaining,
        courses=[placeholder_key],
        source="University requirement: WAYS",
    )
    return [rule], {placeholder_key: placeholder}


def build_college_rule(
    courses: dict[str, Course],
    admit_fall_year: int,
    completed_keys: set[str] | None = None,
    req: dict | None = None,
    other_required_keys: set[str] | None = None,
) -> tuple[RequirementRule, dict[str, Course]]:
    """COLLEGE (Civic, Liberal, and Global Education) -- NOT "COLLEGE101 AND
    COLLEGE102," despite how easy it is to misread the bulletin that way.
    data/university_requirements.json's own "college" section documents the
    real policy verbatim: "Students choose 2 of their first-year's 3
    quarters to enroll in COLLEGE" -- Autumn's COLLEGE101, Winter's
    COLLEGE102, and a Spring Global Perspectives elective (real courses,
    COLLEGE103-127) are three interchangeable SLOTS, any 2 of which
    satisfy the (2-quarter-policy) requirement. A student who took
    COLLEGE101 and a real Global Perspectives elective genuinely never
    needs COLLEGE102 at all.

    This was a real, user-reported bug: this function used to hard-require
    COLLEGE101 AND COLLEGE102 unconditionally (kind="all"), so a student
    who'd already satisfied the requirement via COLLEGE101 + a Global
    Perspectives elective (confirmed against a real uploaded transcript:
    COLLEGE101 in Autumn, COLLEGE119 -- a real 103-127-range elective -- in
    Spring, no COLLEGE102 at all) still had COLLEGE102 show up as required
    in the generated schedule. Fixed by modeling it as "at least 2 of
    {COLLEGE101, COLLEGE102, a Global Perspectives slot}" instead -- if the
    transcript already has a real completed Global Perspectives elective,
    that specific course fills the 3rd slot directly (so 101+102 aren't
    needed either, if the student instead took, say, 102 + a GP elective);
    otherwise the 3rd slot is the same kind of placeholder as before,
    needed only if the solver can't otherwise reach 2 of 3 from what's
    already completed.

    `admit_fall_year` selects which policy applies: 2 quarters (pick 2 of
    the 3 slots above) through the 2026-2027 catalog year, 3 quarters (all
    3 slots) for anyone starting Fall 2027 or later -- see
    data/university_requirements.json's "college" section. The 3rd
    quarter's exact requirement hadn't been published by Stanford as of
    this project's research; modeling the 2027+ policy as "all 3 of the
    same slots, instead of picking 2" is a documented best guess, not a
    verified rule."""
    req = req or load_university_requirements()
    completed_keys = completed_keys or set()
    other_required_keys = other_required_keys or set()
    gp_pool = sorted(k for k in courses if _GLOBAL_PERSPECTIVES_RE.match(k))
    three_quarter = admit_fall_year >= 2027
    source_note = (
        "University requirement: COLLEGE -- 3-quarter version (Fall 2027+ "
        "policy). Stanford hadn't published the exact 3rd-quarter policy "
        "as of this project's research; modeled as needing all 3 of "
        "{COLLEGE101, COLLEGE102, a Global Perspectives elective} rather "
        "than picking 2 of them, as a documented best guess, not a "
        "verified rule."
        if three_quarter else
        "University requirement: COLLEGE -- 2-quarter version (current "
        "policy through the 2026-2027 catalog year): at least 2 of "
        "{COLLEGE101, COLLEGE102, a Global Perspectives elective}."
    )
    placeholders: dict[str, Course] = {}
    # The 3rd slot: if a real Global Perspectives elective is already on
    # the transcript, use THAT specific course key directly (so it counts
    # toward the 2-of-3 total like any other completed slot) instead of
    # also creating a placeholder nobody needs. Otherwise, same "solver
    # picks a real course later" placeholder as WAYS/WR1/WR2 use.
    completed_gp = next((k for k in gp_pool if k in completed_keys or k in other_required_keys), None)
    if completed_gp is not None:
        gp_slot = completed_gp
    else:
        gp_slot = "PLACEHOLDER:COLLEGE_GLOBAL_PERSPECTIVES"
        placeholders[gp_slot] = make_placeholder_course(
            gp_slot,
            "COLLEGE Global Perspectives elective -- a real course from the "
            "COLLEGE 103-127 menu, your choice. Left as a placeholder rather "
            "than naming one specific real section out of the ~25-course menu.",
            units_min=3.0, units_max=3.0, units_typical=3.0,
        )
    slots = ["COLLEGE101", "COLLEGE102", gp_slot]
    rule = RequirementRule(
        label=(
            "University requirement: COLLEGE (3 of 3: COLLEGE101, COLLEGE102, "
            "and a Global Perspectives elective)"
            if three_quarter else
            "University requirement: COLLEGE (2 of 3: COLLEGE101, COLLEGE102, "
            "or a Global Perspectives elective -- any 2)"
        ),
        kind="all" if three_quarter else "at_least_n",
        n=None if three_quarter else 2,
        courses=slots,
        source=source_note,
    )
    return rule, placeholders


def build_writing_rules(
    courses: dict[str, Course],
    completed_keys: set[str] | None = None,
    req: dict | None = None,
    other_required_keys: set[str] | None = None,
) -> tuple[list[RequirementRule], dict[str, Course]]:
    """WR1 and WR2 -- real topic-section pools (PWR1<suffix>/PWR2<suffix>,
    e.g. PWR1AH, PWR2CA), confirmed against the fetched snapshot: ~80 WR1
    sections, ~65 WR2 sections, not a single literal "PWR1"/"PWR2" course
    (that code doesn't exist -- every section has its own topic and its
    own suffix). ITALIC95W (a real, currently-offered WR1 alternative) is
    included too; ESF95W is a documented WR1 alternative per the bulletin
    but doesn't resolve to any course in the current ESF department fetch
    (possibly renumbered or not offered this catalog year) -- omitted
    rather than included as permanently "unplaced" noise, since the ~80
    other options already cover the requirement.

    Each pool is filtered to `units_typical >= 3` -- excludes PWR1WS
    ("PWR 1 Studio," 1 unit), a companion section for multilingual/
    international student writers taken CONCURRENTLY with a real PWR1
    section, per its own description ("taken concurrently with PWR 1"),
    not a standalone alternative to one. `_WR1_RE` alone can't tell it
    apart from a real ~5-unit topic section (both start with "PWR1"), so
    without this filter the solver could satisfy the ENTIRE WR1
    requirement with just this 1-unit studio course alone -- confirmed as
    a real, reported bug, not hypothetical.

    Like build_ways_rules, whichever pool isn't already satisfied by your
    transcript is left as a placeholder rather than naming one specific
    real section out of ~65-80 alternatives -- no rule at all if it's
    already covered."""
    completed_keys = completed_keys or set()
    other_required_keys = other_required_keys or set()
    _wr_credit_keys = completed_keys | other_required_keys
    wr1_pool = [k for k in courses if _WR1_RE.match(k) and courses[k].units_typical >= 3]
    if "ITALIC95W" in courses:
        wr1_pool.append("ITALIC95W")
    wr1_pool.sort()
    wr2_pool = sorted(k for k in courses if _WR2_RE.match(k) and courses[k].units_typical >= 3)

    rules: list[RequirementRule] = []
    placeholders: dict[str, Course] = {}

    if not any(k in _wr_credit_keys for k in wr1_pool):
        placeholder_key = "PLACEHOLDER:WR1"
        placeholders[placeholder_key] = make_placeholder_course(
            placeholder_key,
            "WR1 (Writing and Rhetoric 1) -- a real topic section, your choice "
            "(e.g. PWR1AH or any of the ~80 other real alternatives). Left as a "
            "placeholder rather than naming one specific real section.",
            units_min=5.0, units_max=5.0, units_typical=5.0,
        )
        rules.append(RequirementRule(
            label="University requirement: WR1 (Writing and Rhetoric 1)",
            kind="at_least_n",
            n=1,
            courses=[placeholder_key],
            source="University requirement: Writing (WR1)",
        ))

    if not any(k in _wr_credit_keys for k in wr2_pool):
        placeholder_key = "PLACEHOLDER:WR2"
        placeholders[placeholder_key] = make_placeholder_course(
            placeholder_key,
            "WR2 (Writing and Rhetoric 2) -- a real topic section, your choice "
            "(e.g. PWR2CA or any of the ~65 other real alternatives). Left as a "
            "placeholder rather than naming one specific real section.",
            units_min=5.0, units_max=5.0, units_typical=5.0,
        )
        rules.append(RequirementRule(
            label="University requirement: WR2 (Writing and Rhetoric 2)",
            kind="at_least_n",
            n=1,
            courses=[placeholder_key],
            source=(
                "University requirement: Writing (WR2) -- the university "
                "requires WR1 before WR2; NOT enforced here, since this "
                "project's prerequisite mechanism sequences one literal "
                "course before another, not one open placeholder before "
                "another. Low real-world risk since advising already "
                "sequences this in practice."
            ),
        ))
    return rules, placeholders


LANGUAGE_PLACEHOLDER_KEY = "PLACEHOLDER:UNIVERSITY_LANGUAGE_REQUIREMENT"


def build_language_rule(quarters_needed: int) -> tuple[RequirementRule, dict[str, str | Course]] | None:
    """The language requirement is NOT tied to real course data at all --
    which language, and how many quarters are still needed, depends on a
    placement test or AP score this app has no data source for. Instead,
    the student directly supplies `quarters_needed` (see app.py's "quarters
    of language still needed" input), and this builds a single repeatable
    PLACEHOLDER course requirement for that many quarters -- reusing
    solver.py's existing "solo at_least_n(n>1) rule" repeat-eligibility
    mechanism (see its _walk_repeat_eligibility), the same mechanism
    MATH-MIN's free-text placeholder pool already relies on, so no solver
    changes were needed to support this.

    Returns None if quarters_needed <= 0 (nothing left to schedule)."""
    if quarters_needed <= 0:
        return None
    # Unlike most placeholders (a free-text requirement with no enumerated
    # course list, where 3-5 units is a genuine guess), a language
    # requirement quarter isn't actually uncertain in units -- Stanford's
    # real language sequence courses (the *LANG subject codes -- see
    # engine/multi_program.py's _LANG_SUBJECT_RE) are consistently 5 units
    # each, confirmed against real transcripts. Fixed at 5/5/5 instead of
    # the generic 3-5 range/4-internal default so the informational
    # 180-unit check and the schedule's per-quarter unit totals aren't
    # quietly off by up to 1 unit per quarter of language still needed.
    placeholder = make_placeholder_course(
        LANGUAGE_PLACEHOLDER_KEY,
        "Stanford language requirement -- a specific language course, chosen "
        "by you based on your placement test result or AP credit (not "
        "modeled here; see data/university_requirements.json's 'language' "
        "section for the real policy). Replace with your actual language "
        "course(s) before trusting this schedule.",
        units_min=5.0, units_max=5.0, units_typical=5.0,
    )
    # A full year of language (3 quarters -- Stanford's standard first-year
    # sequence, e.g. SPANLANG1/2/3 or JAPANLNG1/2/3) is, in real practice,
    # always taken as one Autumn-Winter-Spring block of a single academic
    # year, not just "3 quarters without too big a gap" -- confirmed
    # directly per user feedback: the general `sequential` contiguity rule
    # below would still let e.g. Spring/Autumn/Winter (split across two
    # academic years) satisfy the count, which isn't how anyone actually
    # takes a language sequence. Only applies at exactly 3 quarters
    # remaining -- 1 or 2 quarters is presumably a partial sequence (e.g.
    # already placed out of level 1, or into the language via AP credit)
    # with no natural "whole year" shape to pin down, so those stay under
    # the more permissive general contiguity rule only.
    rule = RequirementRule(
        label=f"University requirement: Language ({quarters_needed} quarter(s) remaining)",
        kind="at_least_n",
        n=quarters_needed,
        courses=[placeholder.key],
        source="University requirement: Language (manual input, not tied to real course data)",
        # A language sequence must be taken in temporally contiguous
        # quarters (at most one skipped quarter, and only if it's Summer)
        # -- unlike ordinary solo-repeat placeholders (e.g. WAYS
        # categories), which have no such relationship between occurrences.
        sequential=True,
        full_year_sequence=(quarters_needed == 3),
    )
    # The full Course object, not just its description -- see
    # course_data.add_placeholder_courses()'s docstring for why: passing a
    # bare description string there rebuilds a fresh placeholder with the
    # generic 3-5/4 unit guess, discarding the 5/5/5 set above.
    return rule, {placeholder.key: placeholder}


def build_all_university_rules(
    courses: dict[str, Course],
    admit_fall_year: int,
    language_quarters_needed: int = 0,
    completed_keys: set[str] | None = None,
    req: dict | None = None,
    other_required_keys: set[str] | None = None,
) -> tuple[list[RequirementRule], dict[str, str | Course]]:
    """Convenience entry point for app.py: every university-wide rule this
    module covers (WAYS, COLLEGE, Writing; Language if quarters_needed > 0),
    plus any placeholder courses they need merged into the course dict via
    course_data.add_placeholder_courses() before solving -- same
    (rules, placeholders) contract as requirements.build_requirements(),
    except a value here may be a full Course (not just a description
    string) when the placeholder's units are known rather than guessed --
    see build_language_rule().

    `completed_keys` (new parameter -- every pre-existing caller that still
    omits it gets `None`/empty, meaning "nothing completed yet," so WAYS/
    COLLEGE/Writing behave as if every category still needs its own
    placeholder, same shape as before this parameter existed, just via a
    placeholder instead of a real course pool) is what lets WAYS/COLLEGE/
    Writing skip a placeholder entirely for whatever your transcript
    already covers -- see build_ways_rules's docstring for why naming a
    specific real course for these was replaced with placeholders at all.

    `other_required_keys` (also new): course keys already required by
    something ELSE in this same solve -- a program sheet's own
    requirements, or app.py's manually-added course list (section 2d) --
    that haven't been completed yet but are going to be scheduled
    regardless. Treated identically to `completed_keys` for WAYS/COLLEGE/
    Writing credit purposes (a real, user-reported case: a manually-added
    CHEM131 already carries real WAYS tags, so it shouldn't ALSO force a
    fully separate, generic WAYS placeholder for that same credit) -- see
    models.all_referenced_course_keys() for how app.py computes this set.
    Never widens the one-course-one-WAYS-category cap in
    _max_ways_assignment: a multi-tagged course still only ever counts
    toward a single category, whether it came from completed_keys or
    other_required_keys."""
    req = req or load_university_requirements()
    completed_keys = completed_keys or set()
    other_required_keys = other_required_keys or set()
    placeholders: dict[str, str | Course] = {}

    ways_rules, ways_placeholders = build_ways_rules(courses, completed_keys, req, other_required_keys)
    rules = ways_rules
    placeholders.update(ways_placeholders)

    college_rule, college_placeholders = build_college_rule(
        courses, admit_fall_year, completed_keys, req, other_required_keys
    )
    rules.append(college_rule)
    placeholders.update(college_placeholders)

    writing_rules, writing_placeholders = build_writing_rules(courses, completed_keys, req, other_required_keys)
    rules.extend(writing_rules)
    placeholders.update(writing_placeholders)

    lang = build_language_rule(language_quarters_needed)
    if lang is not None:
        lang_rule, lang_placeholders = lang
        rules.append(lang_rule)
        placeholders.update(lang_placeholders)
    return rules, placeholders
