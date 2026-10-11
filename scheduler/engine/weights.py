"""
The tweakable parameter table. Each WeightRule.type below maps to a scoring
function here that contributes linear terms to the solver's objective.

Adding a new *value* of an existing type (a different subject, different
days, a different course) never touches this file -- it's just a new row
in someone's weight table. Adding a genuinely new *kind* of preference
(one no existing type can express) means writing one new function here and
registering it in SCORERS; that new type then becomes available to every
user's weight table, not just the one who needed it.

Every scorer has the same shape:
    scorer(model, x, courses, quarter_list, args, weight) -> list[(coeff, BoolVar)]
returning terms to add (with sign already applied) to the objective being
maximized. `model` is only needed by scorers that must create auxiliary
variables (e.g. unit_load_balance's deviation variable).
"""
from __future__ import annotations

import re

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)$", re.I)


def _season(quarter_label: str) -> str:
    return quarter_label.rsplit(" ", 1)[-1]


def _meets_only_on(course, preferred_days: set[str]) -> bool:
    """True if every sampled meeting pattern we have for this course uses
    only days in preferred_days. False (no bonus) if we have no sample
    data to judge by -- an estimate we can't confirm shouldn't score."""
    if not course.sample_meetings:
        return False
    return all(set(m.days) <= preferred_days for m in course.sample_meetings)


def score_day_concentration(model, x, courses, quarter_list, args, weight):
    """Reward scheduling courses whose (estimated) meeting days fall
    entirely within the requested day set, e.g. {"Tuesday","Thursday"}."""
    preferred_days = set(args["days"])
    terms = []
    for (key, qi), var in x.items():
        course = courses[key]
        if _meets_only_on(course, preferred_days):
            terms.append((weight, var))
    return terms


def score_subject_timing(model, x, courses, quarter_list, args, weight):
    """Reward taking courses of a given subject in the requested phase of
    the plan: 'early' = first half of the remaining quarters, 'late' =
    second half."""
    subject = args["subject"].upper()
    phase = args["phase"]
    n = len(quarter_list)
    midpoint = n / 2
    terms = []
    for (key, qi), var in x.items():
        if courses[key].subject.upper() != subject:
            continue
        in_phase = qi < midpoint if phase == "early" else qi >= midpoint
        if in_phase:
            terms.append((weight, var))
    return terms


def score_course_timing(model, x, courses, quarter_list, args, weight):
    """Reward taking specific named courses in the requested phase of the
    plan: 'early' = first half of the remaining quarters, 'late' = second
    half. Same 'early'/'late'-by-half-of-horizon logic as subject_timing,
    just scoped to an explicit list of course keys instead of matching by
    subject -- for the real case subject_timing can't express: a handful of
    specific courses within the SAME subject that are typically taken at
    different points (e.g. CS106B/CS109/CS103 skew early, CS223A/CS231N
    skew late, even though they're all 'CS')."""
    target_keys = {c.upper().replace(" ", "") for c in args["courses"]}
    phase = args["phase"]
    n = len(quarter_list)
    midpoint = n / 2
    terms = []
    for (key, qi), var in x.items():
        if key not in target_keys:
            continue
        in_phase = qi < midpoint if phase == "early" else qi >= midpoint
        if in_phase:
            terms.append((weight, var))
    return terms


def score_interest_boost(model, x, courses, quarter_list, args, weight):
    """Flat bonus for including a specific (usually elective) course
    anywhere in the plan."""
    target_key = args["course"].upper().replace(" ", "")
    terms = []
    for (key, qi), var in x.items():
        if key == target_key:
            terms.append((weight, var))
    return terms


def score_unit_load_balance(model, x, courses, quarter_list, args, weight):
    """Penalize a quarter's unit load EXCEEDING a target -- one-directional,
    never a reward for reaching it from below.

    Originally used a symmetric |deviation| (via AddAbsEquality), which had
    a real, user-reported side effect: being UNDER target was penalized
    exactly as much as being OVER target, so the solver was rewarded for
    padding a quarter with courses nothing actually requires, purely to
    close the gap -- confirmed as the mechanism behind filler courses like
    PWR1WS showing up with no other reason to be scheduled. The user's own
    words: "Don't bother trying to fill units to meet a certain number --
    that can be chosen by the student according to their interests... If
    the unit_load_balance parameter is set, that's to make sure units are
    relatively evenly distributed and avoid scheduling TOO MANY units in a
    single quarter. Having just 10 units scheduled is fine." So this only
    creates a cost when units_expr exceeds target; being under target costs
    nothing, and the solver has no incentive to add anything just to reach
    it."""
    target = int(round(args["target_units_per_quarter"]))
    terms = []
    for qi, _q in enumerate(quarter_list):
        quarter_vars = [(int(round(courses[key].units_typical)), var)
                         for (key, vqi), var in x.items() if vqi == qi]
        if not quarter_vars:
            continue
        units_expr = sum(u * v for u, v in quarter_vars)
        over = model.NewIntVar(0, 200, f"over_q{qi}")
        model.Add(over >= units_expr - target)
        terms.append((-weight, over))
    return terms


def _parse_hour(t: str) -> int | None:
    """24-hour hour-of-day from a sampled meeting's "12:30:00 PM"-style
    start time, or None if it doesn't parse (don't guess)."""
    m = _TIME_RE.match(t.strip())
    if not m:
        return None
    hour = int(m.group(1)) % 12
    if m.group(4).upper() == "PM":
        hour += 12
    return hour


def _earliest_start_hour(course) -> int | None:
    """The earliest sampled meeting's start hour for this course, or None
    if there's no sample data (or none of it parses) to judge by -- same
    "an estimate we can't confirm shouldn't score" rule as
    _meets_only_on above."""
    hours = [h for h in (_parse_hour(m.start_time) for m in course.sample_meetings) if h is not None]
    return min(hours) if hours else None


def score_time_of_day(model, x, courses, quarter_list, args, weight):
    """Reward courses whose (estimated) earliest sampled meeting starts
    before noon ('morning') or at/after noon ('afternoon'). Covers both
    "shift toward morning" and "shift toward afternoon" preferences via one
    type + a phase arg, same pattern subject_timing already uses for
    early/late. No effect on a course with no sample meeting-time data."""
    phase = args["phase"]
    terms = []
    for (key, qi), var in x.items():
        hour = _earliest_start_hour(courses[key])
        if hour is None:
            continue
        is_morning = hour < 12
        if (phase == "morning") == is_morning:
            terms.append((weight, var))
    return terms


def score_department_diversity(model, x, courses, quarter_list, args, weight):
    """Reward touching more distinct subjects within a quarter -- one bonus
    per subject that shows up at all that quarter (not per course), so
    piling multiple courses from the same subject into a quarter earns
    nothing extra, while pulling in a course from a not-yet-represented
    subject does. Discourages (via opportunity cost against the unit cap,
    not a hard rule) a quarter that ends up entirely one department."""
    terms = []
    subjects_by_quarter: dict[int, dict[str, list]] = {}
    for (key, qi), var in x.items():
        subjects_by_quarter.setdefault(qi, {}).setdefault(courses[key].subject, []).append(var)
    for qi, by_subject in subjects_by_quarter.items():
        for subject, subj_vars in by_subject.items():
            present = model.NewBoolVar(f"subj_present_q{qi}_{subject}")
            model.AddMaxEquality(present, subj_vars)
            terms.append((weight, present))
    return terms


def _meeting_days(course) -> set[str]:
    days: set[str] = set()
    for m in course.sample_meetings:
        days.update(m.days)
    return days


def score_weekday_spread(model, x, courses, quarter_list, args, weight):
    """Reward covering more distinct weekdays within a quarter -- one bonus
    per weekday that has at least one scheduled meeting that quarter, so
    spreading across the week earns more than clustering everything onto
    the same 1-2 days. Same sample-meeting-data caveat as day_concentration
    -- a course with no sample data contributes to no weekday's bonus."""
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    days_by_quarter: dict[int, dict[str, list]] = {}
    for (key, qi), var in x.items():
        for day in _meeting_days(courses[key]) & set(weekdays):
            days_by_quarter.setdefault(qi, {}).setdefault(day, []).append(var)
    terms = []
    for qi, by_day in days_by_quarter.items():
        for day, day_vars in by_day.items():
            present = model.NewBoolVar(f"day_present_q{qi}_{day}")
            model.AddMaxEquality(present, day_vars)
            terms.append((weight, present))
    return terms


SCORERS = {
    "day_concentration": score_day_concentration,
    "subject_timing": score_subject_timing,
    "course_timing": score_course_timing,
    "interest_boost": score_interest_boost,
    "unit_load_balance": score_unit_load_balance,
    "time_of_day": score_time_of_day,
    "department_diversity": score_department_diversity,
    "weekday_spread": score_weekday_spread,
}

WEEKDAY_OPTIONS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

# Schema for the UI to render one properly-typed input widget per arg,
# instead of asking someone to hand-write a JSON object in a spreadsheet
# cell (the old design -- confirmed fragile in practice: a user typing just
# "morning" into that cell, expecting it to be the value rather than a JSON
# blob, got a bare parse error with no path forward except reading source
# code). Each arg entry's "kind" tells the UI which widget to use:
#   "enum"       -> a selectbox restricted to "options"
#   "multi_enum" -> a multiselect restricted to "options"
#   "course_key" -> a text input, upper-cased/space-stripped into a course key
#   "subject"    -> a text input, upper-cased into a subject code
#   "number"     -> a number input
# A type with no args (e.g. department_diversity) has an empty "args" dict.
WEIGHT_TYPE_SCHEMA = {
    "day_concentration": {
        "description": "Prefer courses that meet only on certain days (e.g. cluster classes on Tue/Thu).",
        "args": {
            "days": {"kind": "multi_enum", "options": WEEKDAY_OPTIONS,
                      "help": "Only courses meeting exclusively on these days get the bonus."},
        },
    },
    "subject_timing": {
        "description": "Prefer taking a subject earlier or later across your whole multi-year plan "
                        "(e.g. front-load Chemistry, push Engineering to later years) -- this is about "
                        "WHICH YEAR/QUARTER a subject lands in, not what time of day it meets.",
        "args": {
            "subject": {"kind": "subject", "help": "e.g. 'CHEM', 'CS'"},
            "phase": {"kind": "enum", "options": ("early", "late"),
                      "help": "'early' = first half of your planning horizon, 'late' = second half."},
        },
    },
    "course_timing": {
        "description": "Prefer specific named courses earlier or later across your whole "
                        "multi-year plan -- like subject_timing, but scoped to an explicit list "
                        "of courses instead of a whole subject (for when only SOME courses in a "
                        "subject skew early/late, not the subject as a whole).",
        "args": {
            "courses": {"kind": "course_key_list",
                        "help": "Comma or newline separated course keys, e.g. 'CS106B, CS109, CS103'"},
            "phase": {"kind": "enum", "options": ("early", "late"),
                      "help": "'early' = first half of your planning horizon, 'late' = second half."},
        },
    },
    "interest_boost": {
        "description": "Give a flat bonus to including a specific elective course somewhere in the plan.",
        "args": {
            "course": {"kind": "course_key", "help": "e.g. 'CS221'"},
        },
    },
    "unit_load_balance": {
        "description": "Penalize quarters whose REQUIRED unit load exceeds a target -- one-directional, "
                        "so it caps overly heavy quarters but never pads a light quarter with unrequired "
                        "filler just to reach the target. Falling under the target costs nothing.",
        "args": {
            "target_units_per_quarter": {"kind": "number", "default": 15, "min": 1, "max": 30,
                                          "help": "Cap: quarters at or below this are never penalized."},
        },
    },
    "time_of_day": {
        "description": "Shift classes toward morning or afternoon meeting times -- this is about what "
                        "time of day a class meets, not which quarter/year it falls in (see "
                        "subject_timing for that). No effect on a course with no sampled meeting-time data.",
        "args": {
            "phase": {"kind": "enum", "options": ("morning", "afternoon"),
                      "help": "'morning' = starts before noon, 'afternoon' = starts at/after noon."},
        },
    },
    "department_diversity": {
        "description": "Reward touching more distinct subjects/departments within a quarter -- "
                        "discourages a quarter made up entirely of one department.",
        "args": {},
    },
    "weekday_spread": {
        "description": "Reward covering more distinct weekdays within a quarter (spread classes "
                        "across the week instead of clustering on 1-2 days). Based on estimated "
                        "meeting days; no effect on a course with no sample data.",
        "args": {},
    },
}
