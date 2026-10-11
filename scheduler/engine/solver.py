"""
The actual optimizer: a real integer program (CP-SAT, via Google OR-Tools),
run locally in Python -- deliberately NOT inside a browser artifact, after
we confirmed (see prototype notes) that browser sandboxing blocks the
WebAssembly a real solver needs, and that pure-JS fallbacks can silently
return wrong answers. This is the piece that guarantees correctness: given
a time limit, CP-SAT either proves it found the best schedule under your
weights, or tells you it didn't have time to prove optimality -- it never
just quietly hands back a worse answer.

Decision variable: x[(course_key, quarter_index)] = 1 if that course is
taken in that quarter. Only created for (course, quarter) pairs where the
course is actually offered in that quarter's season -- this is how
"quarter-offering" constraints (and any CourseOverride correcting them)
get enforced, for free, just by which variables exist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .course_data import high_confidence_prerequisite_groups, is_high_confidence_corequisite
from .models import Course, RequirementRule, WeightRule
from .typical_ordering import find_typical_order_pairs
from .weights import SCORERS


def _season(quarter_label: str) -> str:
    return quarter_label.rsplit(" ", 1)[-1]


# --- Time-conflict detection -------------------------------------------
# Real user-reported bug: CHEM123 and CIVIC49A were scheduled in the same
# quarter with byte-identical Tuesday/Thursday 10:30-11:50 AM meeting times
# -- confirmed against the real sample_meetings data -- and nothing in this
# file (or anywhere else) ever checked two DIFFERENT courses' meeting times
# against each other for overlap. weights.py's soft scorers only ever judge
# one course's own meeting pattern in isolation (morning/afternoon
# preference, day spread, etc.), never a pairwise comparison.

_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)$", re.I)


def _parse_clock_minutes(t: str) -> int | None:
    """Minutes since midnight from a sampled meeting's "12:30:00 PM"-style
    time string, or None if it doesn't parse (don't guess -- same
    philosophy as weights.py's _parse_hour, just minute-precision instead
    of hour-precision since a genuine conflict can hinge on a few minutes)."""
    m = _CLOCK_RE.match(t.strip())
    if not m:
        return None
    hour = int(m.group(1)) % 12
    if m.group(4).upper() == "PM":
        hour += 12
    return hour * 60 + int(m.group(2))


def _meeting_windows_for_season(course: "Course", season: str) -> list[tuple[frozenset, int, int]]:
    """This course's sampled meeting windows whose OWN sampled quarter is
    the same season as the quarter we're checking (e.g. a "Winter"-sampled
    pattern is our best estimate for any future Winter quarter -- the same
    season-matching philosophy ensure_vars_for() already uses for
    terms_offered). Each window is (days_as_a_set, start_minutes,
    end_minutes); only windows where every field actually parses are
    included -- don't guess at a conflict from incomplete data."""
    windows = []
    for m in course.sample_meetings:
        if _season(m.quarter) != season:
            continue
        start = _parse_clock_minutes(m.start_time)
        end = _parse_clock_minutes(m.end_time)
        if start is None or end is None or not m.days:
            continue
        windows.append((frozenset(m.days), start, end))
    return windows


def _windows_conflict(windows_a: list[tuple[frozenset, int, int]], windows_b: list[tuple[frozenset, int, int]]) -> bool:
    """True if any window in windows_a overlaps (shared day AND overlapping
    clock range) any window in windows_b. Split out from _has_time_conflict
    so build_and_solve can precompute+cache each course's windows once per
    season instead of re-parsing on every pairwise comparison."""
    for days_a, start_a, end_a in windows_a:
        for days_b, start_b, end_b in windows_b:
            if not (days_a & days_b):
                continue
            if start_a < end_b and start_b < end_a:
                return True
    return False


def _has_time_conflict(course_a: "Course", course_b: "Course", season: str) -> bool:
    """True if course_a and course_b have any overlapping sampled meeting
    window estimated for the same season. False (not "unknown" -- see
    build_and_solve's own comment on this at the call site) whenever either
    course has no usable sample-meeting data for this season, since a
    conflict can only be hard-enforced from data we actually trust, not
    guessed at from an absence of data."""
    return _windows_conflict(
        _meeting_windows_for_season(course_a, season),
        _meeting_windows_for_season(course_b, season),
    )


def _is_and_group(item) -> bool:
    """True for a compound alternative like ("__and__","MATH51","MATH53")
    -- produced by bulletin_parser.py for slots like "(MATH51 AND MATH53)
    OR (CME100 AND CME102)", where one *alternative* itself requires
    multiple courses together, not just one."""
    return isinstance(item, tuple) and bool(item) and item[0] == "__and__"


def _flatten_alt(item) -> tuple[str, ...]:
    """A rule's course entry is either a single course key, a tuple of
    OR-alternative keys (e.g. ("CS107","CS107E")), or an AND-group tuple
    (see _is_and_group). Normalize to the tuple of real course keys (no
    "__and__" marker)."""
    if _is_and_group(item):
        return item[1:]
    return item if isinstance(item, (tuple, list)) else (item,)


@dataclass
class ScheduleResult:
    status: str                       # "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN"
    objective_value: float | None
    assignment: dict[str, str]        # course_key -> quarter_label (one quarter per key --
                                       # see repeat_quarters below for repeat-eligible courses
                                       # scheduled more than once)
    unplaced_required: list[str]      # required courses that had no valid variable at all
    solve_time_s: float
    # Populated only when status == "INFEASIBLE" *because of* a hard-enforced
    # prerequisite (see build_and_solve's prerequisite-enforcement section
    # below): each (prereq_display, course_key) pair is one the solver proved
    # is jointly unsatisfiable with everything else. `prereq_display` is
    # usually just a course key, but for an OR-group prerequisite (e.g.
    # CS107E's real "CS106B or CS106X") it's the alternatives joined by
    # " or " -- it's a display string, not necessarily a key you can look up
    # in `courses`. Empty if the model is infeasible for an unrelated reason,
    # or if it's not infeasible at all -- this is the caller's signal for
    # "you can offer the user a specific prerequisite to drop," not a general
    # list of every prerequisite that happened to be enforced.
    blocking_prerequisites: list[tuple[str, str]]
    # course_key -> ALL quarters it's scheduled in, chronologically -- only
    # present here (as a key at all) when a repeat-eligible course (a solo
    # at_least_n(n>1) placeholder like the Language requirement, or a
    # unit_min-repeated course) is actually scheduled more than once.
    # `assignment` above only has room for one quarter per key (it's a plain
    # dict, one course_key -> one quarter_label), so before this field
    # existed, scheduling e.g. the Language placeholder in 3 different
    # quarters would silently overwrite assignment[key] down to just the
    # LAST of those 3 quarters -- the solver's internal model was correct
    # (all 3 quarters really were reserved, satisfying the at_least_n(3)
    # constraint) but the caller-visible result hid 2 of the 3 enrollments
    # and undercounted their units. Callers that display or sum units per
    # scheduled course should iterate this dict (falling back to
    # `assignment` for everything else) rather than assuming one quarter per
    # key. Default {} for every existing caller.
    repeat_quarters: dict[str, list[str]] = field(default_factory=dict)
    # (prereq_display, course_key, quarter_label) for every CS-department
    # course whose real prerequisite (see 2b/4c above -- CS doesn't
    # actually enforce these in Axess) ended up NOT satisfied by an
    # earlier quarter in the ACTUAL chosen solution. Never a reason the
    # solve fails or a course gets dropped -- these are soft-nudged, not
    # hard-enforced, so this list is purely informational: a light "these
    # ended up out of order" warning the caller can surface, distinct from
    # `blocking_prerequisites` (which is only ever about a genuine,
    # non-CS, hard-enforced prerequisite making the WHOLE solve
    # infeasible). Empty for every pre-existing caller whenever nothing
    # CS-related ended up out of order.
    prerequisite_warnings: list[tuple[str, str, str]] = field(default_factory=list)


def build_and_solve(
    courses: dict[str, Course],
    completed_keys: set[str],
    requirement_rules: list[RequirementRule],
    weight_rules: list[WeightRule],
    quarter_list: list[str],
    unit_cap_per_quarter: float = 20,
    time_limit_s: float = 30.0,
    restricted_course_programs: dict[str, list[str]] | None = None,
    assume_prereqs_satisfied_for: set[str] | frozenset[str] | None = None,
    course_quarter_exclusions: dict[str, set[int]] | None = None,
) -> ScheduleResult:
    # restricted_course_programs (see engine/multi_program.py, which builds
    # this): course_key -> the >=2 program_ids that are stacked in a mode
    # where double-counting is disallowed (double major / double degree)
    # AND that both reference this course inside an elective pool. None/{}
    # (every existing caller) means "nobody's stacking majors with a
    # double-counting restriction" -- the attribution machinery below adds
    # zero variables/constraints in that case, so this is purely additive:
    # every pre-existing call site's behavior is unchanged.
    restricted_course_programs = restricted_course_programs or {}
    # assume_prereqs_satisfied_for: course keys whose OWN prerequisite
    # constraints (section 2b below) should skip auto-adding a fresh,
    # not-otherwise-required course to satisfy them, on the theory that the
    # caller already knows those prerequisites are covered some other way
    # and doesn't want the solver pulling in an EXTRA earlier course to
    # "prove" it. Built for app.py's manual course-list mode: a student who
    # says "I've already finished freshman year, just plan the next three"
    # doesn't want CS106B auto-inserted in front of CS107E just because
    # they didn't also type CS106B into their list -- they're telling us
    # that prerequisite is already handled.
    #
    # Important scope limit (a real bug caught after this was first built):
    # this does NOT disable ordering between two courses that are BOTH
    # already required in the same solve -- e.g. if CHEM171 and CHEM174 are
    # both manually added, CHEM171 still has to come first, even though
    # CHEM174 is in this set. "Assume satisfied" only ever means "don't
    # auto-add something new," never "stop enforcing an order between
    # things I already asked for." See the `_originally_required_keys`
    # check in section 2b for exactly how this distinction is made.
    #
    # Deliberately NOT applied to corequisites -- a corequisite is a
    # same-quarter pairing ("must take CHEM123 alongside CHEM124"), not a
    # "did I finish this already" fact, so skipping it here would silently
    # let a corequisite pair split across quarters, which is a different
    # and still-real bug risk regardless of how much of a degree is
    # already done. None/{} (every pre-existing call site) is a no-op --
    # unchanged behavior.
    assume_prereqs_satisfied_for = assume_prereqs_satisfied_for or set()
    # course_quarter_exclusions: course_key -> set of quarter INDICES (into
    # quarter_list) that course may never be scheduled in -- a hard "this
    # specific course can't land here" constraint, distinct from everything
    # else in this function that reasons about requirements/prerequisites/
    # conflicts. Built for a real, reported case: a course that's bulletin-
    # documented (or otherwise known) to only actually be taken by students
    # in a later year (e.g. juniors/seniors only) shouldn't be scheduled
    # into an earlier-year quarter just because nothing else forbids it.
    # None/{} (every pre-existing call site) is a no-op -- purely additive.
    course_quarter_exclusions = course_quarter_exclusions or {}
    # The whole model-building + solve pass lives in this inner function so
    # it can be run TWICE when needed: once cheaply (prerequisites as plain
    # hard constraints -- the common, fast path used whenever the schedule
    # is actually feasible), and, only if that comes back INFEASIBLE, once
    # more with prerequisites wrapped in CP-SAT "assumption" literals so
    # SufficientAssumptionsForInfeasibility() can name exactly which
    # prerequisite(s) are responsible. Reifying every prerequisite
    # constraint (OnlyEnforceIf) measurably slows CP-SAT's ability to PROVE
    # optimality even in the feasible case -- confirmed directly: several
    # real programs (EE-BS, INTLR-BA, ENVSE-BS, ME-MS) went from solving in
    # well under a second to hitting the full 25s time limit and only
    # reaching FEASIBLE, not OPTIMAL, purely from the added reification.
    # Paying that cost only on the rare infeasible-diagnosis path (instead
    # of on every solve) keeps the common case fast.
    def _attempt(prereq_use_assumptions: bool):
        model = cp_model.CpModel()
        x: dict[tuple[str, int], "cp_model.IntVar"] = {}
        unplaced_required: list[str] = []

        def ensure_vars_for(key: str):
            """Create x[(key,qi)] for every quarter this course is actually
            offered in, if not already created."""
            if key in completed_keys or key not in courses:
                return
            course = courses[key]
            for qi, q in enumerate(quarter_list):
                if (key, qi) in x:
                    continue
                if _season(q) in course.terms_offered:
                    x[(key, qi)] = model.NewBoolVar(f"x_{key}_{qi}")

        # 1. Create variables for every course mentioned in a requirement rule
        # -- recursing into sub_rules, since a parsed program sheet's rules
        # nest arbitrarily (e.g. "Mathematics" wraps "Calculus", "Math
        # Electives", etc. as sub_rules rather than a flat courses list).
        def _walk_rule_courses(rule: RequirementRule):
            for item in rule.courses:
                for alt in _flatten_alt(item):
                    ensure_vars_for(alt)
            for sr in rule.sub_rules:
                _walk_rule_courses(sr)

        for rule in requirement_rules:
            _walk_rule_courses(rule)

        # 1b. Also create variables for any course a weight rule refers to by
        # name (e.g. interest_boost on an elective not in any requirement,
        # or course_timing's explicit course list).
        for wr in weight_rules:
            maybe_key = wr.args.get("course")
            if maybe_key:
                ensure_vars_for(maybe_key.upper().replace(" ", ""))
            for maybe_key in wr.args.get("courses", []) or []:
                ensure_vars_for(maybe_key.upper().replace(" ", ""))

        # 1b2. Hard per-course quarter exclusions (see
        # course_quarter_exclusions's own comment above) -- zero out the
        # variable for any (course, quarter index) pair the caller flagged
        # as off-limits. A no-op wherever the caller passed nothing.
        for _excl_key, _excl_qis in course_quarter_exclusions.items():
            for _excl_qi in _excl_qis:
                if (_excl_key, _excl_qi) in x:
                    model.Add(x[(_excl_key, _excl_qi)] == 0)

        # 1c. Which courses are safe to let repeat across multiple quarters
        # (see constraint #2 below)? Two real, bulletin-confirmed shapes need
        # this, and a third shape must NOT get it or "pick N distinct courses"
        # semantics would silently break:
        #   (a) unit_min-only courses -- e.g. CHEMENG699/MATSCI200/DLCL189H,
        #       an independent-study/thesis-style course whose own units are
        #       well under some program's required unit_min for that slot, so
        #       it has to be taken more than once to add up. Only safe if the
        #       course is NEVER also used in an "all"/"at_least_n" list
        #       elsewhere (a plain multi-item list assumes each member counts
        #       for at most one of the N things being picked).
        #   (b) a "solo" at_least_n rule: exactly one course listed, with
        #       n > 1 -- structurally impossible to mean anything other than
        #       "take this one repeatable course n times," since there's no
        #       second alternative it could mean "n of." Confirmed directly
        #       against MTLIT-MA's own bulletin prose: "Admitted Coterm
        #       students must enroll in one unit of [MTL334D] every quarter
        #       during Autumn, Winter, and Spring" is rendered on the page as
        #       "Complete at least 3 of the following Courses: MTL334D" --
        #       genuinely one course, meant three times, not a parser miss.
        #   (c) anything appearing as one of several alternatives in a normal
        #       multi-item "all"/"at_least_n" list must stay capped at one
        #       enrollment ever, full stop -- that's case (a)'s and (b)'s
        #       exclusion, and it takes priority if a course somehow shows up
        #       in both a solo-repeat rule and a normal list elsewhere.
        _unit_min_only_courses: set[str] = set()
        _multi_item_courses: set[str] = set()
        _solo_repeat_courses: set[str] = set()
        # For a solo at_least_n(n>1) placeholder (case (b) above), the
        # actual required n IS the minimum number of repeats that must be
        # possible -- a flat global cap smaller than that makes the rule
        # mathematically unsatisfiable regardless of how many quarters are
        # in the horizon. Real case this was built for: WAYS's single
        # combined placeholder (see build_ways_rule in engine/
        # university_requirements.py) can need up to n=11 or more distinct
        # instances for a student with zero completed WAYS credit across
        # 8 categories -- silently infeasible under a flat cap of 6, with
        # zero unplaced-required courses and no obvious error pointing at
        # this, since every individual (course, quarter) pair was
        # perfectly valid on its own. Tracked per-course (not globally)
        # since a flat "raise the constant" fix would just move the same
        # bug to whatever future placeholder needs more than the new
        # constant.
        _solo_repeat_required_n: dict[str, int] = {}
        # Solo-repeat courses whose rule additionally marked itself
        # `sequential=True` (currently only the language requirement -- see
        # build_language_rule in engine/university_requirements.py). These
        # get an extra "no skipped non-Summer quarter between the first and
        # last enrollment" constraint below (#2d) that ordinary solo-repeat
        # placeholders like a WAYS category (2 independent, unrelated
        # courses -- no temporal relationship at all) must NOT get.
        _sequential_repeat_courses: set[str] = set()
        # Solo-repeat, `sequential=True` courses whose rule ALSO marked
        # `full_year_sequence=True` (currently only the language
        # requirement at exactly 3 quarters remaining -- see
        # build_language_rule). Stronger than plain sequential: see
        # RequirementRule.full_year_sequence's own comment and the
        # dedicated constraint section below (#2a-bridge-ii) for exactly
        # what extra this enforces.
        _full_year_repeat_courses: set[str] = set()

        def _walk_repeat_eligibility(rule: RequirementRule):
            if rule.kind == "unit_min":
                for item in rule.courses:
                    for alt in _flatten_alt(item):
                        _unit_min_only_courses.add(alt)
            elif rule.kind == "at_least_n" and len(rule.courses) == 1 and (rule.n or 0) > 1:
                _solo_alts = _flatten_alt(rule.courses[0])
                _solo_repeat_courses.update(_solo_alts)
                for _alt in _solo_alts:
                    _solo_repeat_required_n[_alt] = max(_solo_repeat_required_n.get(_alt, 0), rule.n)
                if rule.sequential:
                    _sequential_repeat_courses.update(_solo_alts)
                    if rule.full_year_sequence:
                        _full_year_repeat_courses.update(_solo_alts)
            else:
                for item in rule.courses:
                    for alt in _flatten_alt(item):
                        _multi_item_courses.add(alt)
            for sr in rule.sub_rules:
                _walk_repeat_eligibility(sr)

        for rule in requirement_rules:
            _walk_repeat_eligibility(rule)

        _repeat_eligible_courses = (_unit_min_only_courses | _solo_repeat_courses) - _multi_item_courses
        _full_year_repeat_courses -= _multi_item_courses
        _sequential_repeat_courses -= _multi_item_courses

        # 2. Global invariant: nobody takes the same course in two quarters --
        # except a repeat-eligible course (see above), which may repeat up to
        # _MAX_UNIT_MIN_REPEATS times. This is a deliberate, bounded "last
        # resort": several real program requirements (a thesis/research/
        # independent-study course repeated across quarters to bank units
        # toward one unit_min threshold, or explicitly re-enrolled in every
        # quarter to satisfy a solo at_least_n count) are otherwise
        # mathematically impossible to satisfy under a strict one-enrollment-
        # ever cap, even though they're routinely satisfied this way in
        # practice. Confirmed via `build_and_solve()` that this was the actual
        # cause of several programs (CHEME-MS, MTLIT-MA, CPLIT-BA, GERST-BA,
        # ...) reporting INFEASIBLE with zero unplaced-required courses --
        # every individual course had a valid quarter, but a single-enrollment
        # cap made an otherwise-real unit or count total unreachable.
        _MAX_UNIT_MIN_REPEATS = 6

        def _repeat_cap_for(key: str) -> int:
            # At least _MAX_UNIT_MIN_REPEATS (the "generous default" for a
            # unit_min-only course with no specific known n), but never
            # LESS than a solo at_least_n rule's own required n -- see
            # _solo_repeat_required_n's comment above for why a flat
            # constant alone made WAYS silently unsatisfiable.
            return max(_MAX_UNIT_MIN_REPEATS, _solo_repeat_required_n.get(key, 0))

        by_course: dict[str, list["cp_model.IntVar"]] = {}
        for (key, qi), var in x.items():
            by_course.setdefault(key, []).append(var)
        for key, vars_ in by_course.items():
            cap = min(_repeat_cap_for(key), len(vars_)) if key in _repeat_eligible_courses else 1
            model.Add(sum(vars_) <= cap)

        # 2a-bridge. Contiguity for `sequential=True` repeat placeholders
        # (currently just the language requirement). Real user-reported bug:
        # a language placeholder scheduled Winter+Spring of junior year, then
        # not again until Spring of SENIOR year -- CP-SAT had no reason to
        # avoid scattering a solo at_least_n(n) course's occurrences anywhere
        # across the whole horizon, since the only existing constraint was a
        # bare total-count cap (immediately above). A real language sequence
        # has to be taken in temporally contiguous quarters; at most one
        # skipped quarter is tolerable, and only if that quarter is Summer
        # (the user's own words: "the requirement can bridge summer quarter
        # ... but it should never skip an entire quarter of courses").
        #
        # Modeled per sequential course key as: for every quarter index k
        # whose season is NOT Summer, if some quarter strictly before k has
        # this course scheduled AND some quarter strictly after k also does,
        # then quarter k itself must have it scheduled too (it's "sandwiched"
        # inside the active span and isn't a Summer quarter, so it may not be
        # skipped). Only a Summer quarter is exempt from this -- so the only
        # gap CP-SAT can ever choose is a single Summer quarter, never a real
        # academic quarter, and never two skipped quarters in a row (two
        # skips would require the *second* skipped quarter to ALSO be exempt,
        # and consecutive quarters are never both Summer).
        for key in _sequential_repeat_courses:
            qis = sorted(qi for (k2, qi) in x.keys() if k2 == key)
            if len(qis) < 3:
                continue  # nothing to sandwich with fewer than 3 candidate quarters
            for idx, qi in enumerate(qis):
                if _season(quarter_list[qi]) == "Summer":
                    continue  # exempt -- may be freely skipped as the one bridge quarter
                before = [x[(key, qi2)] for qi2 in qis[:idx]]
                after = [x[(key, qi2)] for qi2 in qis[idx + 1:]]
                if not before or not after:
                    continue  # qi is the first or last candidate quarter -- can't be "sandwiched"
                before_any = model.NewBoolVar(f"lang_before_{key}_{qi}")
                after_any = model.NewBoolVar(f"lang_after_{key}_{qi}")
                model.AddMaxEquality(before_any, before)
                model.AddMaxEquality(after_any, after)
                model.Add(x[(key, qi)] == 1).OnlyEnforceIf([before_any, after_any])

        # 2a-bridge-ii. "Full academic year" block for `full_year_sequence`
        # repeat placeholders (currently only the language requirement at
        # exactly 3 quarters remaining). Real user request: the general
        # sequential-contiguity rule just above only forbids a real skipped
        # quarter -- it would still happily accept e.g. Spring 2027/Autumn
        # 2027/Winter 2028 as "contiguous," even though that's split across
        # two different academic years and isn't how a language sequence is
        # actually taught or taken. This is a strictly stronger constraint:
        # the course's occurrences must land in EXACTLY one real (Autumn,
        # Winter, Spring) triple sharing the same academic-year label, and
        # nowhere else.
        #
        # Modeled as one boolean per CANDIDATE valid triple (three
        # consecutive quarter_list positions whose seasons are literally
        # Autumn/Winter/Spring in order, with an identical year-label
        # prefix, e.g. "2026-2027 Autumn"/"2026-2027 Winter"/"2026-2027
        # Spring"): if that triple's boolean is on, all three of its
        # quarters are scheduled; if off, none of them are. Exactly one
        # candidate triple must be chosen overall, and every OTHER quarter
        # this course could occupy is forced to zero -- together, this
        # pins the course to precisely one real year-block, never a
        # scattered or cross-year set of 3. If the horizon contains NO
        # valid triple at all (e.g. it starts mid-year and never completes
        # one), `sum(triple_vars) == 1` is simply unsatisfiable with zero
        # candidates -- correctly surfaces as an ordinary INFEASIBLE result
        # rather than silently accepting a non-compliant pattern.
        for key in _full_year_repeat_courses:
            valid_triples: list[tuple[int, int, int]] = []
            for i in range(len(quarter_list) - 2):
                a_label, b_label, c_label = quarter_list[i], quarter_list[i + 1], quarter_list[i + 2]
                if (
                    _season(a_label) == "Autumn"
                    and _season(b_label) == "Winter"
                    and _season(c_label) == "Spring"
                    and a_label.rsplit(" ", 1)[0] == b_label.rsplit(" ", 1)[0] == c_label.rsplit(" ", 1)[0]
                    and (key, i) in x and (key, i + 1) in x and (key, i + 2) in x
                ):
                    valid_triples.append((i, i + 1, i + 2))

            triple_vars = []
            covered_qis: set[int] = set()
            for (a, b, c) in valid_triples:
                tv = model.NewBoolVar(f"lang_year_block_{key}_{a}")
                model.Add(x[(key, a)] + x[(key, b)] + x[(key, c)] == 3).OnlyEnforceIf(tv)
                model.Add(x[(key, a)] + x[(key, b)] + x[(key, c)] == 0).OnlyEnforceIf(tv.Not())
                triple_vars.append(tv)
                covered_qis.update((a, b, c))
            model.Add(sum(triple_vars) == 1)
            for qi in range(len(quarter_list)):
                if (key, qi) in x and qi not in covered_qis:
                    model.Add(x[(key, qi)] == 0)

        # 2b. Prerequisites -- hard-enforced, but ONLY for "high confidence"
        # extractions (course_data.high_confidence_prerequisite_groups: no
        # "recommended"/"suggested" non-binding-guidance language in the
        # source text). Each course's prerequisite text is parsed into
        # AND-clauses,
        # and each clause's course codes are OR-alternatives of each other --
        # so a genuine OR ("CS106B or CS106X") is hard-enforced as "at least
        # one of these before", and a genuine AND ("CS106B and CS103", if it
        # ever appears) would be hard-enforced as two independent
        # requirements, rather than a 2+-course extraction being disqualified
        # outright the way it used to be. "Consent of instructor"/"permission
        # of instructor" is deliberately NOT hedge language here (per explicit
        # product decision: a student can always go ask for that) -- see
        # course_data.py's _PREREQ_HEDGE_RE comment.
        #
        # A course key in assume_prereqs_satisfied_for only skips auto-
        # ADDING a fresh, not-otherwise-required prerequisite course for
        # ITSELF (see that parameter's own comment above) -- if a listed
        # prerequisite alternative is ALREADY required by something else in
        # this same solve, ordering against it still gets enforced (see the
        # `_originally_required_keys` check just below): two courses both
        # explicitly required together should still come out in the right
        # order, at no extra cost, even if one of them is in the "assume
        # satisfied" set.
        #
        # Normally (prereq_use_assumptions=False) each is a bare model.Add() --
        # cheap, and this is what a real solve uses. Only when the caller is
        # specifically retrying to diagnose an INFEASIBLE result
        # (prereq_use_assumptions=True) does each pair get wrapped behind its
        # own boolean "assumption" literal (CP-SAT's AddAssumptions/
        # SufficientAssumptionsForInfeasibility, the standard technique for
        # "which of these optional hard constraints is actually causing
        # infeasibility") -- so a genuinely-conflicting prerequisite is
        # diagnosable (ScheduleResult.blocking_prerequisites) instead of just
        # an opaque INFEASIBLE with no indication of what to do about it. The
        # user's own instruction here was explicit: hard-enforce by default,
        # but never a dead end -- surface exactly which prerequisite to drop
        # (via a CourseOverride on that course's prerequisite_courses field,
        # already supported by course_data.apply_overrides()) rather than
        # silently ignoring it OR silently failing with no path forward.
        #
        # NOTE: SufficientAssumptionsForInfeasibility() returns each literal's
        # own model VARIABLE INDEX (var.Index()), not its position in the list
        # passed to AddAssumptions() -- confirmed empirically (a decoy-variable
        # test put the "same" assumption literals at index 10/11 instead of
        # 0/1). So the lookup below is keyed by lit.Index(), not list position.
        #
        # First pass (always runs): for every course already in play with a
        # high-confidence prerequisite, make sure the PREREQUISITE course
        # also gets its own quarter variables created via ensure_vars_for()
        # -- even if nothing in this solve's own requirement rules
        # separately requires it. Skipping this was a real bug caught while
        # validating this feature: a prerequisite that's real but happens
        # not to be one of the rule tree's own required/elective courses
        # would otherwise have ZERO variables in `x` at all, making "prereq
        # before quarter j" vacuously impossible for every j and forcing the
        # course that needs it to be dropped (or the whole model
        # unsatisfiable) for a reason that has nothing to do with the actual
        # schedule.
        # Each candidate is (key, group) where `group` is a tuple of course
        # keys that are OR-alternatives of each other -- "at least one of
        # group's courses before key" is what gets enforced. A singleton
        # group (the common case) is just the old single-prerequisite
        # behavior; a multi-course group is the new OR-enforcement.
        # Snapshot of every course key ANY requirement/weight rule already
        # needs (step 1/1b above), fixed before this loop's own
        # ensure_vars_for() calls can add anything new to `x` -- used just
        # below to distinguish "this course is already being scheduled for
        # some other reason" from "enforcing this would mean freshly
        # auto-adding a course nobody asked for."
        _originally_required_keys = set(by_course.keys())

        # CS department doesn't actually enforce its own prerequisites in
        # Axess -- real, user-confirmed fact, not a heuristic guess. So for
        # any course key WHOSE OWN subject is CS, a real, high-confidence
        # prerequisite still gets tracked and still gets a soft nudge
        # (below, alongside typical_ordering's Roman-numeral nudges) toward
        # the right order, but never a hard model.Add() that could block a
        # student's plan or make the whole solve INFEASIBLE. This applies
        # regardless of which department the PREREQUISITE course itself
        # belongs to (e.g. a CS course prerequisite'd on a MATH course is
        # still soft, since it's CS's own enrollment check that doesn't
        # apply it) -- only the course being scheduled needs to be CS for
        # this to kick in. Every other department's prerequisites are
        # unaffected and keep the existing hard enforcement.
        _prereq_candidates: list[tuple[str, tuple[str, ...]]] = []
        _soft_prereq_candidates: list[tuple[str, tuple[str, ...]]] = []
        for key in list(by_course.keys()):
            if key in completed_keys:
                continue
            course = courses.get(key)
            if course is None:
                continue
            _cs_soft = course.subject.upper() == "CS"
            for group in high_confidence_prerequisite_groups(course):
                # Drop already-satisfied or unresolvable codes from the
                # group rather than discarding the whole clause -- if even
                # one alternative remains resolvable and unsatisfied, "at
                # least one of the remaining alternatives before" is still
                # the right constraint. If completed_keys already covers
                # any alternative, the whole clause is satisfied already.
                if any(c in completed_keys for c in group):
                    continue
                resolvable = tuple(c for c in group if c in courses)
                if not resolvable:
                    continue  # every alternative is an unresolvable code -- won't guess
                if key in assume_prereqs_satisfied_for:
                    # The caller says key's OWN prereqs are already handled
                    # (see that parameter's docstring above) -- but that's
                    # only meant to stop a fresh, unwanted auto-add of a
                    # course nobody asked for, NOT to disable ordering
                    # between two courses that are BOTH already required in
                    # this same solve. Real bug this fixes: CHEM171 and
                    # CHEM174 both manually added -- CHEM174 being in the
                    # manual list used to blanket-skip its own prereq
                    # check entirely, so nothing stopped CHEM174 from
                    # landing before CHEM171 even though both were already
                    # going to be scheduled regardless. So: narrow
                    # `resolvable` down to only the alternative(s) already
                    # required elsewhere (still enforce order against
                    # those -- free correctness, no new course), and only
                    # skip entirely if NONE of the alternatives are already
                    # required (the original "don't auto-add" intent).
                    already_required = tuple(c for c in resolvable if c in _originally_required_keys)
                    if not already_required:
                        continue
                    resolvable = already_required
                for prereq_key in resolvable:
                    ensure_vars_for(prereq_key)
                if _cs_soft:
                    _soft_prereq_candidates.append((key, resolvable))
                else:
                    _prereq_candidates.append((key, resolvable))

        prereq_assumption_by_index: dict[int, tuple[str, str]] = {}
        _prereq_assumption_lits: list["cp_model.IntVar"] = []
        for key, prereq_group in _prereq_candidates:
            course_vars = [(qi, x[(key, qi)]) for qi in range(len(quarter_list)) if (key, qi) in x]
            if not course_vars:
                continue  # this course itself is unplaced; not a prereq issue
            group_label = "_or_".join(prereq_group)
            lit = model.NewBoolVar(f"prereq_ok_{group_label}_before_{key}") if prereq_use_assumptions else None
            for qi, var in course_vars:
                prereq_before = [
                    x[(prereq_key, k)]
                    for prereq_key in prereq_group
                    for k in range(qi)
                    if (prereq_key, k) in x
                ]
                if prereq_use_assumptions:
                    if not prereq_before:
                        # No valid quarter strictly before this one has any
                        # alternative offered at all -- if this assumption
                        # holds, this specific quarter is off the table.
                        model.Add(var == 0).OnlyEnforceIf(lit)
                    else:
                        model.Add(sum(prereq_before) >= var).OnlyEnforceIf(lit)
                else:
                    if not prereq_before:
                        model.Add(var == 0)
                    else:
                        model.Add(sum(prereq_before) >= var)
            if prereq_use_assumptions:
                prereq_display = " or ".join(prereq_group)
                prereq_assumption_by_index[lit.Index()] = (prereq_display, key)
                _prereq_assumption_lits.append(lit)
        if _prereq_assumption_lits:
            model.AddAssumptions(_prereq_assumption_lits)

        # 2b-bis. Corequisites -- hard-enforced, same "high confidence"
        # single-course confidence gate as prerequisites above
        # (course_data.is_high_confidence_corequisite), just applied to
        # corequisite_courses instead of prerequisite_courses. Real
        # user-reported bug: CHEM124 ("Corequisite: Chem 123.") got scheduled
        # without CHEM123 in the same quarter -- nothing in this file
        # referenced "corequisite" at all before this. Directional and
        # SAME-quarter (a corequisite must accompany the course, not precede
        # it, unlike a prerequisite): if `key` is scheduled in quarter qi,
        # its corequisite must be scheduled in that same quarter qi too (or
        # already satisfied via completed_keys, in which case it never shows
        # up in `courses`/`by_course` as something needing its own variable
        # here). No assumption-literal diagnosis path is needed for coreqs
        # (no UI escape hatch for dropping one exists, unlike prerequisites),
        # so this is always a bare model.Add(), same as the common
        # (non-diagnosis) prerequisite path above.
        _coreq_candidates: list[tuple[str, str]] = []
        for key in list(by_course.keys()):
            if key in completed_keys:
                continue
            course = courses.get(key)
            if course is None or not is_high_confidence_corequisite(course):
                continue
            coreq_key = course.corequisite_courses[0]
            if coreq_key in completed_keys or coreq_key not in courses:
                continue  # already satisfied, or an unresolvable code we won't guess at
            ensure_vars_for(coreq_key)
            _coreq_candidates.append((key, coreq_key))

        for key, coreq_key in _coreq_candidates:
            for qi in range(len(quarter_list)):
                if (key, qi) not in x:
                    continue
                if (coreq_key, qi) in x:
                    model.Add(x[(coreq_key, qi)] >= x[(key, qi)])
                else:
                    # The corequisite isn't even offered this quarter -- this
                    # quarter is off the table for `key` too.
                    model.Add(x[(key, qi)] == 0)

        # 2c-bis. No two courses sharing a quarter may have overlapping
        # sampled meeting times (see _has_time_conflict / _windows_conflict
        # above). Real user-reported case: CHEM123 and CIVIC49A, both
        # Tuesday/Thursday 10:30-11:50 AM, scheduled in the same quarter --
        # confirmed against real sample_meetings data, and nothing anywhere
        # in this codebase had ever checked two different courses' meeting
        # times against each other (weights.py's soft scorers only ever
        # judge one course's own pattern in isolation). Only enforced
        # between courses that both have usable sample-meeting data for that
        # quarter's season -- most course/season pairs have none, so
        # `_windows_by_key_season` below keeps this cheap by filtering those
        # out before the O(n^2) pairwise comparison within each quarter.
        _windows_by_key_season: dict[tuple[str, str], list[tuple[frozenset, int, int]]] = {}

        def _windows_for(key: str, season: str) -> list[tuple[frozenset, int, int]]:
            cache_key = (key, season)
            if cache_key not in _windows_by_key_season:
                course = courses.get(key)
                _windows_by_key_season[cache_key] = (
                    _meeting_windows_for_season(course, season) if course is not None else []
                )
            return _windows_by_key_season[cache_key]

        _keys_by_quarter: dict[int, list[str]] = {}
        for (key, qi) in x.keys():
            _keys_by_quarter.setdefault(qi, []).append(key)

        for qi, keys_here in _keys_by_quarter.items():
            season = _season(quarter_list[qi])
            candidates = [k for k in keys_here if _windows_for(k, season)]
            for i, key_a in enumerate(candidates):
                windows_a = _windows_for(key_a, season)
                for key_b in candidates[i + 1:]:
                    if _windows_conflict(windows_a, _windows_for(key_b, season)):
                        model.Add(x[(key_a, qi)] + x[(key_b, qi)] <= 1)

        _and_aux_counter = [0]

        def _course_indicator(key: str) -> "cp_model.LinearExpr | int":
            """0/1-valued expression for a single, literal course key: 1 if
            already completed, else the (capped-at-1-by-the-global-invariant)
            sum of its per-quarter scheduling vars, or 0 with a note in
            unplaced_required if it has no variables at all."""
            if key in completed_keys:
                return 1
            vars_ = [x[(key, qi)] for qi in range(len(quarter_list)) if (key, qi) in x]
            if not vars_:
                unplaced_required.append(key)
                return 0
            return sum(vars_)

        # 2c. Double-counting attribution (only when restricted_course_
        # programs is non-empty -- see build_and_solve's own comment on the
        # parameter, and engine/multi_program.py which builds it). For each
        # restricted course key shared between 2+ stacked programs, this
        # course's single enrollment (already capped at <=1 by the global
        # invariant above) can be credited toward AT MOST ONE of those
        # programs' elective-pool requirements -- modeled with one boolean
        # "attribution" variable per (key, program) pair, constrained so
        # their sum never exceeds the course's own enrolled/completed
        # indicator. _attributed_indicator() below is what rule-building
        # code calls instead of _course_indicator() directly whenever it
        # knows which program's rule tree it's currently walking; for any
        # course NOT in this dict (the overwhelming default case), it's a
        # pure passthrough to _course_indicator(), so nothing changes for
        # every existing single-program / independent-programs solve.
        _attribution_vars: dict[tuple[str, str], "cp_model.IntVar"] = {}
        for key, program_ids in restricted_course_programs.items():
            if len(program_ids) < 2:
                continue
            ind = _course_indicator(key)
            if isinstance(ind, int) and ind == 0:
                continue  # never enrolled/placed at all -- nothing to attribute
            for pid in program_ids:
                _attribution_vars[(key, pid)] = model.NewBoolVar(f"attr_{key}_{pid}")
            model.Add(sum(_attribution_vars[(key, pid)] for pid in program_ids) <= ind)

        def _attributed_indicator(key: str, program_id: str | None):
            if program_id and (key, program_id) in _attribution_vars:
                return _attribution_vars[(key, program_id)]
            return _course_indicator(key)

        def indicator_sum(item, program_id: str | None = None) -> "cp_model.LinearExpr | int":
            """0/1-valued expression for one requirement "slot":
              - a single course key -> that course's own indicator.
              - an OR-alternatives tuple, e.g. ("CS107","CS107E") -> 1 if ANY
                one of them is completed/scheduled.
              - an AND-group tuple, e.g. ("__and__","MATH51","MATH53") -> 1
                only if ALL of them are completed/scheduled together (one
                "Mathematics Sequence" alternative that itself requires both
                courses). Built via a real CP-SAT auxiliary boolean variable
                and the standard AND-linearization constraints, since simply
                summing/flattening the members would silently turn "both" into
                "at least one".

            `program_id` is only ever non-None when this item is being
            evaluated as part of a specific stacked program's elective pool
            under a double-counting restriction (see _attributed_indicator
            above) -- when there's nothing restricted for these particular
            keys (the default, single-program case), every branch below
            behaves exactly as before program stacking existed.
            """
            if _is_and_group(item):
                members = item[1:]
                if any(m not in courses and m not in completed_keys for m in members):
                    unplaced_required.append("&".join(members))
                    return 0
                member_exprs = [_attributed_indicator(m, program_id) for m in members]
                if all(isinstance(e, int) for e in member_exprs):
                    return 1 if all(e == 1 for e in member_exprs) else 0
                _and_aux_counter[0] += 1
                aux = model.NewBoolVar(f"and_{'_'.join(members)}_{_and_aux_counter[0]}")
                for e in member_exprs:
                    model.Add(aux <= e)
                model.Add(aux >= sum(member_exprs) - (len(member_exprs) - 1))
                return aux

            alts = _flatten_alt(item)
            if any(a in completed_keys for a in alts):
                return 1
            if program_id and any((a, program_id) in _attribution_vars for a in alts):
                # At least one alternative here is double-count-restricted
                # for this program -- route every alt through the (possibly
                # attribution-gated) indicator instead of summing raw
                # per-quarter vars directly, so a restricted alt only counts
                # toward this program's threshold if it was actually
                # attributed to this program (not "spent" on the other one).
                exprs = [_attributed_indicator(a, program_id) for a in alts]
                return sum(exprs)
            vars_ = [x[(a, qi)] for a in alts for qi in range(len(quarter_list)) if (a, qi) in x]
            if not vars_:
                unplaced_required.append("/".join(alts))
                return 0
            return sum(vars_)

        _threshold_aux_counter = [0]

        def _cap_or_alts(item, program_id: str | None = None):
            """For an OR-alternatives group like ("CS107","CS107E"), forbid
            taking both -- it's one requirement slot with a choice of course,
            not license to take every option. (A plain single-course item is
            already capped at <=1 by the global "no course twice" invariant;
            an AND-group's indicator is a single 0/1 aux var by construction,
            so it needs no separate cap.) Safe to apply unconditionally, even
            when this item sits inside an optional/"choose N" branch that may
            not end up chosen -- nobody should double-take alternatives either
            way."""
            if isinstance(item, (tuple, list)) and len(item) > 1 and not _is_and_group(item):
                model.Add(indicator_sum(item, program_id) <= 1)

        def _threshold_indicator(expr, threshold: int, expr_upper_bound: int, tag: str):
            """Return a 0/1 aux var equal to 1 iff `expr >= threshold`, given
            `expr` is guaranteed to lie in [0, expr_upper_bound]. Standard
            big-M double-inequality linearization -- used for every "is this
            (sub-)requirement satisfied" indicator in the tree below, since a
            parsed program sheet's rules nest arbitrarily (e.g. CS-BS's
            "Mathematics" = ALL of {Calculus, Math Electives, ...}, where one
            of those is itself "at least 2 of 16 courses")."""
            if threshold <= 0:
                return 1
            _threshold_aux_counter[0] += 1
            aux = model.NewBoolVar(f"th_{tag}_{_threshold_aux_counter[0]}")
            model.Add(expr >= threshold * aux)
            slack = max(expr_upper_bound - threshold + 1, 0)
            model.Add(expr <= (threshold - 1) + slack * aux)
            return aux

        def rule_indicator(rule: RequirementRule, program_id: str | None = None):
            """0/1 expression: is this entire (sub-)rule satisfied? Used when a
            rule is offered as one of several *options* under a parent
            "at_least_n" (e.g. one of several AI-subplan "Areas"), so it must
            not be hard-enforced on its own -- only counted if chosen.

            `program_id` is threaded down from whichever top-level rule this
            recursion started at (see enforce_rule) -- sub_rules always
            belong to the same stacked program as their parent, so there's
            no need to store it on every nested RequirementRule."""
            if rule.kind == "unit_min":
                unit_terms, cap = _unit_terms_and_cap(rule, program_id)
                if not unit_terms:
                    return 0
                return _threshold_indicator(sum(unit_terms), int(round(rule.unit_min or 0)), cap, "unit")

            exprs = []
            for item in rule.courses:
                exprs.append(indicator_sum(item, program_id))
                _cap_or_alts(item, program_id)
            exprs.extend(rule_indicator(sr, program_id) for sr in rule.sub_rules)
            if not exprs:
                return 1 if rule.kind == "all" else 0
            threshold = len(exprs) if rule.kind == "all" else (rule.n or 1)
            return _threshold_indicator(sum(exprs), threshold, len(exprs), "rule")

        def enforce_rule(rule: RequirementRule, program_id: str | None = None):
            """Hard-apply a top-level (mandatory) requirement rule, recursing
            into any nested sub_rules. A rule reached this way is not optional
            -- unlike rule_indicator(), which is for a rule offered as one
            choice among several.

            `program_id` is rule.program_id from whichever TOP-LEVEL rule
            this call originated at (see the "for rule in requirement_rules"
            loop below) -- "" for every rule that isn't part of a stacked-
            major double-counting restriction, which is passed through as
            None so indicator_sum/_cap_or_alts/_unit_terms_and_cap take
            their original, unattributed code path."""
            if rule.kind == "all":
                for item in rule.courses:
                    model.Add(indicator_sum(item, program_id) >= 1)
                    _cap_or_alts(item, program_id)
                for sr in rule.sub_rules:
                    enforce_rule(sr, program_id)
            elif rule.kind == "at_least_n":
                exprs = [indicator_sum(item, program_id) for item in rule.courses]
                for item in rule.courses:
                    _cap_or_alts(item, program_id)
                exprs.extend(rule_indicator(sr, program_id) for sr in rule.sub_rules)
                if not exprs:
                    # Same failure mode as the dead-course-id bug this project
                    # already fixed once (see README): a mandatory at_least_n
                    # rule with nothing real underneath it -- here from a
                    # parser gap that didn't fall through to a placeholder,
                    # rather than every course option being a dead bulletin
                    # id -- becomes a literal `model.Add(0 >= n)`, an
                    # unconditionally-false hard constraint that reports the
                    # whole program INFEASIBLE for a reason that has nothing
                    # to do with real course availability. Treat "nothing to
                    # enforce" as satisfied rather than impossible, same as
                    # rule_indicator() already does for the optional-context
                    # version of this exact situation, and flag it instead of
                    # silently succeeding.
                    print(f"  [warn] at_least_n rule {rule.label!r} has no real courses or sub-rules to enforce -- treating as satisfied rather than reporting the program INFEASIBLE; this needs a parser fix, not a real graduation requirement gap")
                    return
                model.Add(sum(exprs) >= (rule.n or 0))
            elif rule.kind == "unit_min":
                unit_terms, _cap = _unit_terms_and_cap(rule, program_id)
                if unit_terms:
                    model.Add(sum(unit_terms) >= int(round(rule.unit_min or 0)))
                if rule.sub_rules:
                    print(f"  [warn] unit_min rule {rule.label!r} has sub_rules, which are not yet supported -- ignoring them")

        def _unit_terms_and_cap(rule: RequirementRule, program_id: str | None = None):
            """CP-SAT terms for a unit_min rule's accumulated units, plus an
            upper bound on that sum (needed by _threshold_indicator when this
            rule appears as an optional sub-rule). `cap` has to account for the
            possibility of a unit_min-only course repeating across several of
            its x-vars (see the global invariant above) -- it's `units_typical`
            once per available quarter (up to the same repeat cap), not just
            once total, or _threshold_indicator's slack math would understate
            how high this sum can actually go and misfire.

            For a double-count-restricted alt (program_id given, and this
            key is shared with another stacked program): contribute its
            units at most ONCE, gated by this program's attribution
            variable, rather than per-repeat-enrollment -- a simplification,
            since a restricted-shared elective realistically isn't also a
            repeat-eligible thesis/independent-study course in any real
            program sheet this project has seen, so the two mechanisms
            never actually need to compose."""
            unit_terms = []
            cap = 0
            for item in rule.courses:
                for alt in _flatten_alt(item):
                    u = int(round(courses[alt].units_typical)) if alt in courses else 0
                    if alt in completed_keys:
                        cap += u
                        continue
                    if program_id and (alt, program_id) in _attribution_vars:
                        unit_terms.append(u * _attribution_vars[(alt, program_id)])
                        cap += u
                        continue
                    alt_vars = [x[(alt, qi)] for qi in range(len(quarter_list)) if (alt, qi) in x]
                    repeat_cap = min(_repeat_cap_for(alt), len(alt_vars)) if alt in _repeat_eligible_courses else 1
                    cap += u * repeat_cap
                    unit_terms.extend(u * v for v in alt_vars)
            return unit_terms, cap

        # 3. Requirement constraints. Each top-level rule is mandatory; any
        # nesting underneath (named sub-blocks, "choose N of these named
        # groups") is walked recursively by enforce_rule/rule_indicator above.
        # rule.program_id is only ever set (see engine/multi_program.py) when
        # this rule is part of a double major/double degree's double-
        # counting restriction; "" (every other caller) becomes None here so
        # enforce_rule's recursion takes its original, unattributed path.
        for rule in requirement_rules:
            enforce_rule(rule, rule.program_id or None)

        # 4. Unit cap per quarter -- soft, not hard. A real student can petition
        # for overload units, so a quarter that would otherwise need more than
        # `unit_cap_per_quarter` to fit every mandatory course in the window is
        # a "last resort, but resolvable in reality" situation, not a reason to
        # report the whole program INFEASIBLE. Modeled as an `overage` int var
        # per quarter (0 if within cap, otherwise the amount over) that gets a
        # steep objective penalty -- steep enough that CP-SAT only accepts
        # overage when every other way of scheduling has been exhausted, but
        # never a hard wall that blocks an otherwise-real schedule.
        _OVERAGE_PENALTY = 25
        overage_vars: list["cp_model.IntVar"] = []
        for qi, _q in enumerate(quarter_list):
            quarter_vars = [(int(round(courses[key].units_typical)), var)
                             for (key, vqi), var in x.items() if vqi == qi]
            if not quarter_vars:
                continue
            total_units = sum(u * v for u, v in quarter_vars)
            max_possible = sum(u for u, _v in quarter_vars)
            overage = model.NewIntVar(0, max_possible, f"overage_{qi}")
            model.Add(overage >= total_units - int(round(unit_cap_per_quarter)))
            overage_vars.append(overage)

        # 4b. "Typical ordering" -- a SOFT nudge (never a hard constraint)
        # toward scheduling Roman-numeral-sequenced courses in numeral order
        # (e.g. CHEM151 "Inorganic Chemistry I" before CHEM153 "Inorganic
        # Chemistry II"), even when there's no real bulletin prerequisite
        # linking them -- see engine/typical_ordering.py's module docstring
        # for the user feedback that prompted this and why it's keyed on
        # the title's Roman numeral rather than the course number (some
        # sequences, like MATH51/52/53, are numbered sequentially but can
        # be taken in any order -- a number-based rule would wrongly nudge
        # those too). Only pairs where BOTH courses already have scheduling
        # variables from being referenced elsewhere in this solve get a
        # violation var -- this never pulls in an otherwise-unrequired
        # course just to sequence it, and always loses to any real
        # requirement/prerequisite constraint since it's just an objective
        # term, not a model.Add(...).
        _TYPICAL_ORDER_PENALTY = 3
        _typical_order_violations: list["cp_model.IntVar"] = []
        _tov_counter = [0]
        for early_key, late_key in find_typical_order_pairs(courses):
            if early_key not in by_course or late_key not in by_course:
                continue
            if early_key in completed_keys:
                continue  # already done before any future quarter -- can't be violated
            for qi in range(len(quarter_list)):
                if (late_key, qi) not in x:
                    continue
                late_var = x[(late_key, qi)]
                early_before = [x[(early_key, k)] for k in range(qi) if (early_key, k) in x]
                _tov_counter[0] += 1
                viol = model.NewBoolVar(f"typord_viol_{early_key}_{late_key}_{_tov_counter[0]}")
                if not early_before:
                    # No valid quarter strictly before this one has "early"
                    # offered at all -- scheduling "late" here always
                    # violates the typical order (soft: just costs
                    # objective value, doesn't block the schedule).
                    model.Add(viol == late_var)
                else:
                    model.Add(sum(early_before) + viol >= late_var)
                    model.Add(viol <= late_var)
                _typical_order_violations.append(viol)

        # 4c. CS department's own prerequisites -- a SOFT nudge, same shape
        # as 4b's typical-ordering nudge, for exactly the candidates 2b
        # routed into `_soft_prereq_candidates` instead of the hard-
        # enforced `_prereq_candidates` (see that split's own comment
        # above for why: CS doesn't actually enforce these in Axess, so a
        # student really can take them out of order). Real prerequisites,
        # just never a model.Add() that could block a plan -- a slightly
        # heavier penalty than 4b's Roman-numeral nudge (these ARE real,
        # bulletin-stated prerequisites, just an unenforced department
        # policy, not merely "commonsense sequencing" for an unlinked
        # numeral pair), but still just an objective term.
        _CS_SOFT_PREREQ_PENALTY = 6
        _cs_soft_prereq_violations: list["cp_model.IntVar"] = []
        _cspv_counter = [0]
        for key, prereq_group in _soft_prereq_candidates:
            for qi in range(len(quarter_list)):
                if (key, qi) not in x:
                    continue
                key_var = x[(key, qi)]
                prereq_before = [
                    x[(prereq_key, k)]
                    for prereq_key in prereq_group
                    for k in range(qi)
                    if (prereq_key, k) in x
                ]
                _cspv_counter[0] += 1
                viol = model.NewBoolVar(f"cs_soft_prereq_viol_{key}_{qi}_{_cspv_counter[0]}")
                if not prereq_before:
                    model.Add(viol == key_var)
                else:
                    model.Add(sum(prereq_before) + viol >= key_var)
                    model.Add(viol <= key_var)
                _cs_soft_prereq_violations.append(viol)

        # 5. Objective: sum of every weight rule's contribution, plus a small
        # built-in "anti-filler" cost on every course scheduled. Without this,
        # nothing stops the solver from padding the plan with unnecessary
        # electives just because they're harmless -- e.g. satisfying an
        # "at least 2 of these 16" rule by scheduling all 16, or taking both
        # sides of an (A OR B) requirement. This cost is deliberately tiny;
        # weight rows should generally use magnitudes well above 1 so real
        # preferences clearly dominate it rather than being canceled by it.
        objective_terms = [(-1, var) for var in x.values()]
        objective_terms.extend((-_OVERAGE_PENALTY, ov) for ov in overage_vars)
        objective_terms.extend((-_TYPICAL_ORDER_PENALTY, v) for v in _typical_order_violations)
        objective_terms.extend((-_CS_SOFT_PREREQ_PENALTY, v) for v in _cs_soft_prereq_violations)
        for wr in weight_rules:
            scorer = SCORERS.get(wr.type)
            if scorer is None:
                print(f"  [warn] unknown weight type {wr.type!r}, skipping")
                continue
            objective_terms.extend(scorer(model, x, courses, quarter_list, wr.args, int(round(wr.weight))))
        if objective_terms:
            model.Maximize(sum(coef * var for coef, var in objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        return (
            status, solver, x, unplaced_required, prereq_assumption_by_index, _prereq_assumption_lits,
            _soft_prereq_candidates,
        )

    status, solver, x, unplaced_required, _idx_map, _lits, _soft_prereq_candidates = _attempt(False)

    # If that (cheap, non-reified) attempt came back INFEASIBLE, retry once
    # with prerequisites wrapped in assumption literals purely to identify
    # which one(s) are responsible -- see _attempt's docstring-comment for
    # why this is a second pass rather than the default. The retry's own
    # status/x/unplaced_required are discarded; only its infeasibility
    # diagnosis is used, so the returned assignment/unplaced_required always
    # reflect the real (first) attempt.
    blocking_prerequisites: list[tuple[str, str]] = []
    if status == cp_model.INFEASIBLE:
        status2, solver2, _x2, _unplaced2, idx_map2, lits2, _soft2 = _attempt(True)
        if status2 == cp_model.INFEASIBLE and lits2:
            culprit_idxs = solver2.SufficientAssumptionsForInfeasibility()
            blocking_prerequisites = [idx_map2[i] for i in culprit_idxs if i in idx_map2]

    status_name = solver.StatusName(status)

    assignment: dict[str, str] = {}
    repeat_quarters: dict[str, list[str]] = {}
    prerequisite_warnings: list[tuple[str, str, str]] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        _scheduled_qis: dict[str, list[int]] = {}
        for (key, qi), var in x.items():
            if solver.Value(var):
                _scheduled_qis.setdefault(key, []).append(qi)
        for key, qis in _scheduled_qis.items():
            qis.sort()
            assignment[key] = quarter_list[qis[0]]
            if len(qis) > 1:
                repeat_quarters[key] = [quarter_list[qi] for qi in qis]

        # Did the actual chosen solution end up violating one of the CS
        # soft prerequisites from 4c above? Recomputed directly from the
        # final quarter assignment (not from the viol vars themselves --
        # simpler, and equivalent) so callers get a plain, readable
        # (prereq_display, course_key, quarter_label) list to show as a
        # light warning, never a blocker: "the CS department doesn't
        # enforce this course's prerequisite, but FYI it's now out of
        # order." Only for courses that actually got scheduled at all
        # (unplaced_required already covers the "didn't get scheduled"
        # case separately).
        for key, prereq_group in _soft_prereq_candidates:
            if key not in assignment:
                continue
            key_qi = _scheduled_qis[key][0]
            if any(c in completed_keys for c in prereq_group):
                continue
            if any(pk in _scheduled_qis and any(pq < key_qi for pq in _scheduled_qis[pk]) for pk in prereq_group):
                continue
            prerequisite_warnings.append((" or ".join(prereq_group), key, assignment[key]))

    return ScheduleResult(
        status=status_name,
        objective_value=solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        assignment=assignment,
        unplaced_required=sorted(set(unplaced_required)),
        solve_time_s=solver.WallTime(),
        blocking_prerequisites=blocking_prerequisites,
        repeat_quarters=repeat_quarters,
        prerequisite_warnings=prerequisite_warnings,
    )
