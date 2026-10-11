"""
Batch D hand-fixes: MATSC-MS, PHILO-MIN.

Both were diagnosed as hitting the same structural CP-SAT limitation: an
"at_least_n" RequirementRule whose `sub_rules` list has exactly ONE entry
can only ever contribute 0 or 1 toward the count, because
solver.py's rule_indicator() collapses an entire sub-rule (however complex
internally) down to a single 0/1 satisfied/not-satisfied auxiliary
variable via _threshold_indicator(). See solver.py's enforce_rule() for
kind == "at_least_n":

    exprs = [indicator_sum(item) for item in rule.courses]
    ...
    exprs.extend(rule_indicator(sr) for sr in rule.sub_rules)
    ...
    model.Add(sum(exprs) >= (rule.n or 0))

-- each sub_rule contributes exactly one term to `exprs`, worth at most 1,
regardless of how many real courses live underneath it. This is NOT a
solver bug: rule_indicator() is deliberately "is this whole (sub-)rule
satisfied, yes/no" for exactly the case where a sub-rule is offered as one
of several *options* under a parent "at least N of these areas" (e.g.
CS-BS's "Math Electives" vs "Core Program Requirements" both counting
toward some outer pick). The bug in both programs below is upstream, in
what the generic bulletin_parser.py produced: either an `n` that assumed
a flat course count while the tree only had one grouped node under it
(MATSC-MS), or real alternative sections that got dropped/merged down to
one instead of being kept as separate sub_rules (PHILO-MIN).

The fix pattern used here, confirmed against solver.py: when the real
requirement is "pick N of these M individual real courses", put them all
directly in one rule's flat `courses` list with that `n` -- courses are
summed directly via indicator_sum(), not funneled through the 0/1
rule_indicator() collapse, so n > 1 works correctly. Reserve `sub_rules`
for genuinely separate multi-part alternatives, and give it as many
entries as there are real alternatives (each contributing its own 0/1),
matching `n` to how many of *those* must be satisfied.
"""
from __future__ import annotations

from .models import RequirementRule, make_placeholder_course  # noqa: F401  (placeholder helper referenced in docstrings/tests)


def matsc_ms_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """MS in Materials Science and Engineering (samples/MATSC-MS.txt).

    Bug found: the generic parser produced one top-level rule,
    "Requirements Overview", kind=at_least_n, n=7, courses=[], with a
    SINGLE sub_rule ("Coursework Requirements", kind=all, wrapping the
    Core and Lab sub-rules). Since that at_least_n has exactly one
    sub_rule, its rule_indicator() contribution is capped at 1, so
    `sum(exprs) >= 7` became `1 >= 7` -- a hard, unconditionally-false
    constraint. This is a genuine number-extraction/tree-shape bug, not a
    smaller true requirement: the bulletin text is explicit that n=7 is
    real ("Complete at least 7 of the following" / "Students complete 21
    units (7 courses)"), it's just that the 7 needs to be checked against
    a FLAT list of the 12 real course options (5 core + 7 lab), not
    against a single collapsed grouping node.

    Real requirement, per the raw text:
      "Students may choose to complete three core courses and four lab
      courses OR four core courses and three lab courses. In other
      words, at least 9 units (3 courses) must be core courses (MATSCI
      211-215) and at least 9 units (3 courses) must be labs (MATSCI
      170-176). The remaining 3 units (1 course) required may be an
      additional core or lab course from the list below."
    i.e. pick >=3 of the 5 core courses AND >=3 of the 7 lab courses AND
    >=7 total across the combined 12-course pool (the 7th/"wildcard"
    course can be either type). Modeled as three sibling at_least_n
    rules under one "all" wrapper so all three hard constraints apply
    together, none of them going through the single-sub_rule collapse.

    Also included: the "Elective Program Course Requirement" section's
    enumerable half (four 200+-level MATSCI elective courses, real course
    keys pulled from the course snapshot, excluding the core/lab/
    research/practical-training/colloquium-style codes the bulletin text
    says don't count). The other half ("12 units of other Technical
    Electives... in science fields, technology, engineering, math,
    natural sciences, or statistics") is a genuinely unenumerable
    cross-department pool -- represented with a PLACEHOLDER unit_min
    rule rather than guessed at.

    Deliberately NOT included: the MS Thesis option (MATSCI 200, 12
    units) -- the bulletin text is explicit this is optional ("Research
    participation is not required for the degree, but there is the
    option..."), so it must not be a mandatory top-level rule.
    """
    core_courses = ["MATSCI211", "MATSCI212", "MATSCI213", "MATSCI214", "MATSCI215"]
    lab_courses = [
        "MATSCI170", "MATSCI171", "MATSCI172", "MATSCI173",
        "MATSCI174", "MATSCI175", "MATSCI176",
    ]

    # 200+-level MATSCI courses confirmed in the course snapshot as real,
    # currently-offered, letter-graded lecture/lab courses (units_min >= 2)
    # -- excluding MATSCI200 (Master's Research, thesis-only per bulletin
    # text), MATSCI211-215 (already counted as Core above), MATSCI230/231/
    # 232 (1-unit colloquium/advising/ethics seminars -- bulletin text
    # says "seminars... cannot apply"), MATSCI299 (Practical Training,
    # explicitly excluded by bulletin text), MATSCI300/399/400/402A/402T/
    # 405/801/802 (PhD-only research, independent study, teaching
    # participation, or 1-unit workshop/lecture-in-other-departments
    # courses -- bulletin text: "workshop and lecture courses in other
    # departments cannot be counted for this requirement"), and anything
    # with an empty terms_offered in the snapshot (not actually
    # schedulable, so including it would only pollute unplaced_required
    # noise without changing whether the n=4 pick is satisfiable).
    matsci_elective_pool = [
        "MATSCI201", "MATSCI205", "MATSCI208", "MATSCI209", "MATSCI210",
        "MATSCI220", "MATSCI222", "MATSCI225", "MATSCI226", "MATSCI249",
        "MATSCI251", "MATSCI256", "MATSCI286", "MATSCI303", "MATSCI320",
        "MATSCI321", "MATSCI322", "MATSCI323", "MATSCI326", "MATSCI327",
        "MATSCI331", "MATSCI333", "MATSCI341", "MATSCI342", "MATSCI346",
        "MATSCI351", "MATSCI380", "MATSCI384", "MATSCI385", "MATSCI405",
    ]

    placeholders = {
        "PLACEHOLDER:matsc_ms_technical_elective": (
            "MATSC-MS Technical Elective: any letter-graded 200+-level lecture/lab "
            "course in a STEM field outside MATSCI (12 units required) -- bulletin "
            "text names no specific courses, cross-department pool intentionally "
            "not guessed at."
        ),
    }
    rules = [
        RequirementRule(
            label="Coursework Requirements (7 of 12 core+lab, >=3 each)",
            kind="all",
            sub_rules=[
                RequirementRule(
                    label="Core Course Requirements",
                    kind="at_least_n",
                    n=3,
                    courses=list(core_courses),
                    source="MATSC-MS program sheet: Core Course Requirements",
                ),
                RequirementRule(
                    label="Laboratory Course Requirements",
                    kind="at_least_n",
                    n=3,
                    courses=list(lab_courses),
                    source="MATSC-MS program sheet: Laboratory Course Requirements",
                ),
                RequirementRule(
                    label="Total Core+Lab Courses (at least 7 of 12)",
                    kind="at_least_n",
                    n=7,
                    courses=core_courses + lab_courses,
                    source=(
                        "MATSC-MS program sheet: Coursework Requirements -- "
                        "'Complete at least 7 of the following' / 'Students "
                        "complete 21 units (7 courses)'; hand-fixed from a "
                        "parser bug where n=7 was checked against a single "
                        "collapsed sub_rule instead of this flat 12-course pool"
                    ),
                ),
            ],
            source="MATSC-MS program sheet: Coursework Requirements",
        ),
        RequirementRule(
            label="MATSCI Elective Course Requirement (4 of the following)",
            kind="at_least_n",
            n=4,
            courses=matsci_elective_pool,
            source=(
                "MATSC-MS program sheet: Elective Program Course Requirement -- "
                "'Four MATSCI Elective Courses are required (12 units). These "
                "must be 200+ Level MATSCI Courses taken for a letter grade "
                "(seminars, labs, and research units cannot apply).'"
            ),
        ),
        RequirementRule(
            label="Technical Elective Course Requirement (12 units, cross-department)",
            kind="unit_min",
            unit_min=12.0,
            courses=["PLACEHOLDER:matsc_ms_technical_elective"],
            source=(
                "MATSC-MS program sheet: Elective Program Course Requirement -- "
                "'12 units of other Technical Electives are required... letter-"
                "graded lecture/lab courses at the 200 level or above in science "
                "fields, technology, engineering, math, natural sciences, or "
                "statistics.' Genuinely unenumerable cross-department pool -- "
                "represented with a placeholder rather than a guessed course list."
            ),
        ),
    ]
    return rules, placeholders


def philo_min_manual_rules() -> tuple[list[RequirementRule], dict[str, str]]:
    """Philosophy Minor (samples/PHILO-MIN.txt).

    Bug found: the generic parser produced a top-level rule labeled
    'History of Philosophy' (mislabeled -- it's actually the "Contemporary
    Philosophy: Complete at least 2 of the following" node), kind=
    at_least_n, n=2, with exactly ONE sub_rule chain instead of the three
    real OR'd alternative "areas" the raw text lists:
      - "Philosophy of Science and Logic" (logic OR philosophy-of-science)
      - "Moral and Political Philosophy" (PHIL2 OR a second course list)
      - "Contemporary Theoretical Philosophy" (a course list)
    Two of those three areas got dropped/merged during parsing, and
    PHIL2 (which really belongs under "Moral and Political Philosophy")
    ended up misattached three levels down under "Philosophy of Science".
    With only one real sub_rule surviving, rule_indicator() capped its
    contribution at 1, so `sum(exprs) >= 2` became `1 >= 2` -- always
    false, hence INFEASIBLE. Confirmed this is a genuine "3 real
    alternatives collapsed to 1" parser bug, not a case where n=2 needs
    revising: the raw text is unambiguous ("Contemporary Philosophy /
    Complete at least 2 of the following:" followed by three named
    sub-sections joined by "OR").

    Two of the three areas' course lists ("Philosophy of Science" and
    "Contemporary Theoretical Philosophy", plus the second alternative
    under "Moral and Political Philosophy") render in the scraped sample
    text only as "Complete at least 1 courses in the following Course
    Lists: Untitled Course Set" -- the underlying course-list widget's
    actual contents were not captured in the text scrape at all, so
    there is no real course data available to enumerate them from (this
    is a genuine data gap, not something to guess at). Those three are
    represented as PLACEHOLDER pools. The only two concretely-named
    requirements in the raw text -- the Logic list (PHIL150/151/154) and
    Moral and Political Philosophy's PHIL2 option -- use their real
    course keys.

    Also included: "History of Philosophy" (a real, separately-named
    top-level section requiring one History of Philosophy course
    numbered 100+, >=3 units) -- the scraped text names no specific
    courses for it either, so it's also a placeholder.

    Deliberately NOT included: "Electives" ("Any course 10 and above
    except for 196-199") -- the raw text gives no explicit unit/course
    count for this section (unlike, say, MATSC-MS's explicit "12 units"),
    and this codebase's generic parser never turns "Minimum Units in the
    Program" page metadata into an enforced unit_min rule for ANY
    program (confirmed: bulletin_parser.py only stores units_min/
    university_units_min as ProgramRequirements metadata, never emits a
    RequirementRule from it) -- so omitting a made-up elective unit
    count here is consistent with how every other program in this corpus
    is already handled, not a special-case relaxation for this one.
    """
    placeholders = {
        "PLACEHOLDER:philo_min_history_of_philosophy_100plus": (
            "PHILO-MIN History of Philosophy: a History of Philosophy course "
            "numbered 100+, minimum 3 units -- no specific courses enumerated "
            "in the scraped bulletin text."
        ),
        "PLACEHOLDER:philo_min_philosophy_of_science": (
            "PHILO-MIN Philosophy of Science (alternative to the Logic list "
            "under Philosophy of Science and Logic) -- the bulletin's course-"
            "list widget for this alternative rendered as an empty 'Untitled "
            "Course Set' in the scraped text, no real course codes captured."
        ),
        "PLACEHOLDER:philo_min_moral_political_list": (
            "PHILO-MIN Moral and Political Philosophy, second alternative to "
            "PHIL2 -- the bulletin's course-list widget for this alternative "
            "rendered as an empty 'Untitled Course Set' in the scraped text, "
            "no real course codes captured."
        ),
        "PLACEHOLDER:philo_min_contemporary_theoretical": (
            "PHILO-MIN Contemporary Theoretical Philosophy -- the bulletin's "
            "course-list widget rendered as an empty 'Untitled Course Set' in "
            "the scraped text, no real course codes captured."
        ),
    }
    rules = [
        RequirementRule(
            label="History of Philosophy",
            kind="all",
            courses=["PLACEHOLDER:philo_min_history_of_philosophy_100plus"],
            source=(
                "PHILO-MIN program sheet: History of Philosophy -- 'Complete a "
                "History of Philosophy course numbered 100 or above and for a "
                "minimum of three units.' No specific courses enumerated in the "
                "scraped bulletin text."
            ),
        ),
        RequirementRule(
            label="Contemporary Philosophy (at least 2 of the following areas)",
            kind="at_least_n",
            n=2,
            sub_rules=[
                RequirementRule(
                    label="Philosophy of Science and Logic",
                    kind="at_least_n",
                    n=1,
                    courses=[
                        ("PHIL150", "PHIL151", "PHIL154"),
                        "PLACEHOLDER:philo_min_philosophy_of_science",
                    ],
                    source=(
                        "PHILO-MIN program sheet: Philosophy of Science and Logic -- "
                        "'Fulfill ANY of the following requirements: Logic (Complete "
                        "at least 1 of PHIL150/PHIL151/PHIL154) OR Philosophy of "
                        "Science (Complete at least 1 courses in the following Course "
                        "Lists: Untitled Course Set)'. The Philosophy of Science course "
                        "list's actual contents were not captured in the scraped text."
                    ),
                ),
                RequirementRule(
                    label="Moral and Political Philosophy",
                    kind="at_least_n",
                    n=1,
                    courses=[
                        "PHIL2",
                        "PLACEHOLDER:philo_min_moral_political_list",
                    ],
                    source=(
                        "PHILO-MIN program sheet: Moral and Political Philosophy -- "
                        "'Fulfill ANY of the following requirements: Unnamed Rule "
                        "(Complete ALL of PHIL2) OR Unnamed Rule (Complete at least 1 "
                        "courses in the following Course Lists: Untitled Course Set)'. "
                        "The second alternative's course list was not captured in the "
                        "scraped text."
                    ),
                ),
                RequirementRule(
                    label="Contemporary Theoretical Philosophy",
                    kind="at_least_n",
                    n=1,
                    courses=["PLACEHOLDER:philo_min_contemporary_theoretical"],
                    source=(
                        "PHILO-MIN program sheet: Contemporary Theoretical Philosophy -- "
                        "'Complete at least 1 courses in the following Course Lists: "
                        "Untitled Course Set'. Course list contents not captured in the "
                        "scraped text."
                    ),
                ),
            ],
            source=(
                "PHILO-MIN program sheet: Contemporary Philosophy -- 'Complete at "
                "least 2 of the following:' governing the three areas above. "
                "Hand-fixed from a parser bug where n=2 was checked against a "
                "single collapsed sub_rule instead of these three real alternatives."
            ),
        ),
    ]
    return rules, placeholders


MANUAL_FIXES = {
    "MATSC-MS": matsc_ms_manual_rules,
    "PHILO-MIN": philo_min_manual_rules,
}
