"""
Wraps the `explorecourses` package to build a local JSON snapshot of course
listings, and loads that snapshot back for the solver.

Why a snapshot instead of calling explorecourses live every run: (1) it's
slow to pull every department every time, (2) this engine is meant to also
run from inside a browser-facing app later where we can't call arbitrary
Python packages, and (3) it gives a stable, versioned dataset that manual
CourseOverride corrections layer on top of predictably. Refresh the
snapshot every quarter or so with `refresh_snapshot()`.

explorecourses does NOT give structured prerequisites -- only free-text
descriptions that happen to mention them. It also only has confirmed
meeting times for quarters that have already been built into the registrar
schedule, not ones years in the future. `sample_meetings` below is
therefore a *representative estimate*, not a guarantee, for anything more
than a couple quarters out.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from explorecourses import CourseConnection

from .models import Course, MeetingPattern, CourseOverride, make_placeholder_course

# A starter set of departments -- CS-BS/EE-BS/MATH-MIN's own courses, used
# by smoke_test.py's demo solve. Not remotely exhaustive: once program
# sheets covered every in-scope major/minor (218 of them, see
# fetch_bulletin.py), the *real* set of subjects a full run needs is far
# bigger than this. Use subjects_referenced_by_programs() below to compute
# that real set from what the parsed program sheets actually reference,
# rather than guessing/hand-maintaining a department list.
DEFAULT_DEPARTMENTS = (
    "CS", "MATH", "PHYSICS", "CME", "ENGR", "EE", "CHEM", "BIO",
    "MATSCI", "PWR", "COLLEGE", "MUSIC", "ATHLETIC", "JAPANLNG",
)

# explorecourses' own department codes are usually identical to the course
# subject code (e.g. department "HISTORY" offers "HISTORY" courses), which
# is what lets subjects_referenced_by_programs() feed straight into
# get_courses_by_department() below without a separate subject->department
# lookup table. Known where that assumption breaks: overseas-program
# subjects (OSPOXFRD, OSPPARIS, ...) are cross-listings of courses actually
# homed in a real department elsewhere, not separately browsable
# departments themselves, so they're expected to fail/return nothing and
# don't need chasing down individually.
_COURSE_ID_CACHE_SUBJECT_RE = re.compile(r"^([A-Z][A-Z&]{1,9})\d")


def subjects_referenced_by_programs(cache_path: str = "data/course_id_cache.json") -> tuple[str, ...]:
    """Every distinct subject prefix (e.g. "ARCHLGY" out of "ARCHLGY1 -
    Introduction to Archaeology") that appears anywhere in the resolved
    course-id cache -- i.e. every subject actually referenced by one of the
    218 in-scope program sheets, not a hand-guessed department list. This is
    what a real, full-coverage snapshot fetch should pass as `departments`
    to refresh_snapshot(), instead of the CS/EE/MATH-only DEFAULT_DEPARTMENTS
    above."""
    cache = json.loads(Path(cache_path).read_text())
    subjects: set[str] = set()
    for value in cache.values():
        if not isinstance(value, str) or " - " not in value:
            continue  # skips "__UNRESOLVED__" and anything else not a real "CODE - Title" mapping
        code_part = value.split(" - ", 1)[0].strip()
        m = _COURSE_ID_CACHE_SUBJECT_RE.match(code_part)
        if m:
            subjects.add(m.group(1))
    return tuple(sorted(subjects))

# Split on sentence boundaries so a "prerequisite"/"corequisite" mention
# gets isolated from unrelated text before/after it in the same
# description (these clauses show up mid-paragraph as often as at the end).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PREREQ_COREQ_KEYWORD_RE = re.compile(r"\b(pre|co)requisites?\b", re.IGNORECASE)
# Course-code-shaped tokens, e.g. "CS106A", "CS 106A", "MS&E 111", "Chem
# 123" (a prose cross-reference in title case, e.g. CHEM124's real
# description: "Corequisite: Chem 123." -- confirmed for real, not
# hypothetical, that this title-case form is common enough in bulletin
# prose that requiring an all-caps subject silently missed it, producing
# an empty corequisite_courses for a course whose corequisites_text
# plainly names one). Subject letters are matched case-insensitively for
# this reason and uppercased when the key is built below; the trailing
# digit requirement (immediately after, at most one optional space) still
# keeps this from matching ordinary prose words, since "Chapter 5" or
# "Room 100" would need the same tight digit-adjacency by coincidence.
# Deliberately not cross-checked against known subjects, so it still
# catches prereqs in departments we haven't fetched (e.g. PHIL151 as a CS
# prereq).
_COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,10}(?:&[A-Za-z])?)\s?(\d{1,3}[A-Za-z]{0,3})\b")


def extract_prereq_text(description: str) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    """Best-effort extraction of prerequisite/corequisite info from a course
    description, by isolating whichever sentence(s) literally contain the
    word "prerequisite" or "corequisite" and then pulling course-code-shaped
    tokens out of those sentences.

    Returns (prerequisites_text, prerequisite_courses, corequisites_text, corequisite_courses).

    This is a heuristic, not a parser: phrasing like "or equivalent", "CS107
    or above", codes split across a slash ("AA 171/274"), and prereq info
    that spills into a *following* sentence without repeating the keyword
    (e.g. "Prerequisites: Programming maturity up to CS 107. Familiarity with
    the command line...") won't be fully captured. Good enough as a
    reviewable starting point, not as ground truth.
    """
    prereq_sentences: list[str] = []
    coreq_sentences: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(description or ""):
        matches = list(_PREREQ_COREQ_KEYWORD_RE.finditer(sentence))
        if not matches:
            continue
        # A sentence mentioning both ("Prerequisite: X. Corequisite: Y" split
        # oddly, or "prerequisite or corequisite") goes to prereq by default
        # unless *every* mention in it is "co".
        if all(m.group(1).lower() == "co" for m in matches):
            coreq_sentences.append(sentence.strip())
        else:
            prereq_sentences.append(sentence.strip())

    prereq_text = " ".join(prereq_sentences)
    coreq_text = " ".join(coreq_sentences)
    prereq_courses = tuple(sorted({f"{a}{b}".upper().replace(" ", "") for a, b in _COURSE_CODE_RE.findall(prereq_text)}))
    coreq_courses = tuple(sorted({f"{a}{b}".upper().replace(" ", "") for a, b in _COURSE_CODE_RE.findall(coreq_text)}))
    return prereq_text, prereq_courses, coreq_text, coreq_courses


# Phrases that mean "the named course isn't actually required at all" --
# genuinely non-binding guidance ("recommended", "suggested"), where the
# course is mentioned but the bulletin explicitly says it's optional.
# is_high_confidence_prerequisite()/high_confidence_prerequisite_groups()
# below treat either of these as reason NOT to hard-enforce.
#
# Deliberately NOT included:
#   - "consent of instructor"/"permission of instructor"/"instructor
#     consent"/"instructor permission" -- a student can always go ask for
#     that, so it doesn't make an otherwise-clear course prerequisite
#     (e.g. CS107E's real "Prerequisite: CS106B or CS106X, and consent of
#     instructor.") any less real or worth enforcing.
#   - "equivalent"/"or higher"/"or above"/"or greater" -- a REAL, user-
#     reported bug: CHEM174's actual prerequisite is "CHEM 171 or
#     equivalent," which extracts a perfectly clear, singular, real course
#     (CHEM171) -- "or equivalent" just means transfer/AP credit is ALSO
#     accepted, a pathway this tool has no data source for and can't
#     verify a given student has, so defaulting to the one concrete,
#     named course is strictly more accurate than not enforcing anything
#     at all. Same reasoning covers "or higher"/"or above"/"or greater"
#     (e.g. "CS107 or above" still names CS107 as A valid, literal way to
#     satisfy it). This previously blocked CHEM171 from ever being
#     required before CHEM174, confirmed as a real scheduling gap, not
#     hypothetical -- removing it affects 275 courses corpus-wide (audited
#     directly), the overwhelming majority following this exact same
#     "real named course, plus an unmodeled alternate pathway" shape.
_PREREQ_HEDGE_RE = re.compile(
    r"\b(recommended|suggested)\b",
    re.IGNORECASE,
)

# Splits prerequisite/corequisite text into separate AND-clauses on the
# connector "and" -- e.g. "CS106B or CS106X, and consent of instructor"
# becomes ["CS106B or CS106X, ", " consent of instructor"]. Course codes
# pulled out of the SAME clause are OR-alternatives of each other (any one
# suffices); codes from DIFFERENT clauses are AND'ed together (each clause
# independently required). This is what lets a genuine OR ("CS106B or
# CS106X") be told apart from a genuine AND ("CS106B and CS103") -- the flat,
# connector-blind tuple extract_prereq_text() returns can't distinguish
# these on its own.
_AND_CLAUSE_SPLIT_RE = re.compile(r"\band\b", re.IGNORECASE)


def extract_prereq_groups(text: str) -> tuple[tuple[str, ...], ...]:
    """AND-of-OR-groups structure parsed from prerequisite/corequisite text:
    each inner tuple is a set of course codes that are OR-alternatives of
    each other (any one satisfies that clause); the outer tuple of those is
    what must ALL be independently satisfied. Empty clauses (no course code
    found in them -- e.g. "consent of instructor" on its own) are dropped
    rather than returned as empty groups.

    Still a heuristic layered on top of extract_prereq_text()'s own
    heuristics, not a real parser -- e.g. "either X or Y" is handled fine
    (both codes end up in one clause since "either"/"or" aren't AND-split
    points), but something like "X, unless Y" would not be. Good enough as
    a reviewable starting point, matching this project's existing philosophy
    for this file.
    """
    groups: list[tuple[str, ...]] = []
    for clause in _AND_CLAUSE_SPLIT_RE.split(text or ""):
        codes = tuple(sorted({f"{a}{b}".upper().replace(" ", "") for a, b in _COURSE_CODE_RE.findall(clause)}))
        if codes:
            groups.append(codes)
    return tuple(groups)


def high_confidence_prerequisite_groups(course: "Course") -> tuple[tuple[str, ...], ...]:
    """The AND-of-OR-groups worth hard-enforcing for this course's
    prerequisites, or () if none / not high-confidence.

    Supersedes the old "exactly one extracted course" gate: since
    extract_prereq_groups() can now tell a genuine OR ("CS106B or CS106X")
    apart from a genuine AND ("CS106B and CS103") by splitting on the "and"
    connector, a 2+-course extraction is no longer automatically excluded --
    each parsed clause becomes its own hard "at least one of this clause's
    courses" requirement, and multiple clauses are all separately enforced.
    Real case this was built for: CS107E's "Prerequisite: CS106B or CS106X,
    and consent of instructor." now yields (("CS106B", "CS106X"),) -- one
    clause, two OR-alternatives -- instead of being disqualified outright
    for having 2 extracted courses.

    The _PREREQ_HEDGE_RE check is applied PER CLAUSE (after the same
    AND-split extract_prereq_groups uses), not to the whole source text --
    a real, user-reported gap the previous whole-text version had: CS349H/
    EE349's "CS107 or CS107E (required) and EE180 (recommended)" has the
    hedge word "recommended" describing ONLY the EE180 clause, but checking
    the whole text blanked out the CS107/CS107E clause too, even though
    it's explicitly marked "required" and has no hedge language of its
    own. Now only the clause that actually contains a hedge word is
    dropped -- see high_confidence_recommended_courses() below for where
    that dropped clause's course(s) go instead (recognized as real,
    bulletin-stated advice, just not hard-enforced). Self-references (a
    clause whose only code is the course itself) are filtered out of that
    clause; a clause that becomes empty as a result (from either filter)
    is dropped entirely.
    """
    if not course.prerequisite_courses:
        return ()
    cleaned: list[tuple[str, ...]] = []
    for clause in _AND_CLAUSE_SPLIT_RE.split(course.prerequisites_text or ""):
        if _PREREQ_HEDGE_RE.search(clause):
            continue
        codes = tuple(sorted({f"{a}{b}".upper().replace(" ", "") for a, b in _COURSE_CODE_RE.findall(clause)}))
        filtered = tuple(c for c in codes if c != course.key)
        if filtered:
            cleaned.append(filtered)
    return tuple(cleaned)


def high_confidence_recommended_courses(course: "Course") -> tuple[str, ...]:
    """Courses named in a HEDGED clause of this course's prerequisite text
    (e.g. the "EE180 (recommended)" half of CS349H's "CS107 or CS107E
    (required) and EE180 (recommended)") -- real, bulletin-stated advice,
    just not something the solver should hard-enforce the way it does
    high_confidence_prerequisite_groups() above. The two functions are
    complementary and clause-disjoint: a clause with no hedge word feeds
    the groups function (as a hard requirement); a clause WITH a hedge
    word feeds this one instead (as advice) -- no clause's courses appear
    in both, and no clause is silently dropped by one without surfacing
    in the other. Self-references are filtered out, same as above. Flat
    (not grouped by clause/OR-alternative) since nothing currently needs
    the finer structure for advisory courses -- just the set of names
    worth surfacing to a student."""
    if not course.prerequisite_courses:
        return ()
    recommended: set[str] = set()
    for clause in _AND_CLAUSE_SPLIT_RE.split(course.prerequisites_text or ""):
        if not _PREREQ_HEDGE_RE.search(clause):
            continue
        for a, b in _COURSE_CODE_RE.findall(clause):
            code = f"{a}{b}".upper().replace(" ", "")
            if code != course.key:
                recommended.add(code)
    return tuple(sorted(recommended))


def is_high_confidence_prerequisite(course: "Course") -> bool:
    """True only for the narrow case worth hard-enforcing in the solver:
    exactly one extracted prerequisite course, with no hedge language (see
    _PREREQ_HEDGE_RE) anywhere in the sentence it came from, and not a
    (nonsensical) self-reference.

    Kept around (in addition to the newer, more general
    high_confidence_prerequisite_groups() above, which solver.py now
    actually uses) as a simple single-course convenience check for any
    other caller that just wants a yes/no answer without the group
    structure."""
    if len(course.prerequisite_courses) != 1:
        return False
    only = course.prerequisite_courses[0]
    if only == course.key:
        return False
    if _PREREQ_HEDGE_RE.search(course.prerequisites_text or ""):
        return False
    return True


def is_high_confidence_corequisite(course: "Course") -> bool:
    """Same gate as is_high_confidence_prerequisite, applied to
    corequisite_courses instead -- exactly one extracted course, no hedge
    language in the sentence it came from, not a self-reference. Confirmed
    a real, previously-unenforced case this gate correctly admits: CHEM124's
    corequisites_text is "Corequisite: Chem 123." with no hedge words,
    extracting to exactly one course (CHEM123) once the case-sensitivity
    fix in _COURSE_CODE_RE above is applied.

    Corequisites don't get the OR-group treatment prerequisites do above --
    no real corequisite case in this catalog has needed it, and a
    same-quarter OR-corequisite constraint is a materially different (and
    currently unbuilt) shape than the before-quarter prerequisite one."""
    if len(course.corequisite_courses) != 1:
        return False
    only = course.corequisite_courses[0]
    if only == course.key:
        return False
    if _PREREQ_HEDGE_RE.search(course.corequisites_text or ""):
        return False
    return True


def _season(term: str) -> str:
    """'2026-2027 Winter' -> 'Winter'"""
    return term.rsplit(" ", 1)[-1]


def _state_path(out_path: str) -> Path:
    return Path(out_path).with_suffix(".fetch_state.json")


# A cross-listed course's title always names every *other* code it's
# cross-listed under in a trailing parenthetical, e.g. explorecourses'
# own title for the course ARCHA-MIN calls "ANTHRO129C" is literally
# "A Deep Dive Into the Indian Ocean: From Prehistory to the Modern Day
# (ANTHRO 229C, ARCHLGY 129C, ARCHLGY 229C, EBS 129, EBS 229, OCEANS 129C,
# OCEANS 229C)". Confirmed empirically (not just theorized) that this
# matters: explorecourses only attaches real Section/term data to ONE of
# a course's cross-listed codes -- calling get_courses_by_department and
# looking at *that specific* course object's own `.sections` for ANTHRO
# 129C returns zero, even in a quarter where the class is definitely
# being taught, because the section data lives on the ARCHLGY 129C
# listing of the exact same class instead. Across the full snapshot,
# 45.6% of courses show empty terms_offered, and of those, cross-listed
# ones are measurably more likely to be empty than non-cross-listed ones
# (51% vs 39%) -- consistent with "the section data is just attached to
# a sibling code," not "these are all genuinely unoffered."
_TITLE_CROSSLIST_RE = re.compile(r"\(([^)]+)\)\s*$")
_CROSSLIST_CODE_RE = re.compile(r"^([A-Z][A-Z&]{1,9})\s*(\d[A-Z0-9]*)$")


def _crosslist_siblings(title: str) -> list[str]:
    """Course keys named in a title's trailing cross-listing parenthetical,
    e.g. "...(ANTHRO 229C, ARCHLGY 129C)" -> ["ANTHRO229C", "ARCHLGY129C"].
    Returns [] for a title with no such parenthetical, or one that doesn't
    parse as a list of course codes (e.g. a course whose real title just
    happens to end in a parenthetical aside)."""
    m = _TITLE_CROSSLIST_RE.search(title)
    if not m:
        return []
    keys = []
    for part in m.group(1).split(","):
        cm = _CROSSLIST_CODE_RE.match(part.strip())
        if cm:
            keys.append((cm.group(1) + cm.group(2)).upper().replace(" ", ""))
    return keys


def merge_cross_listed_terms(courses: dict[str, dict]) -> tuple[dict[str, dict], int]:
    """Returns a new courses dict (never mutates the input) where every
    course's `terms_offered` is unioned with its cross-listed siblings'
    (found via _crosslist_siblings, using each course's OWN dict entry --
    this only ever draws on data already in `courses`, no new fetching).
    Cross-listed codes form a clique (each names every other), so this
    closes over the whole group via union-find rather than a single hop,
    in case any individual listing's title is missing a sibling or two.

    Second return value is how many courses gained at least one term they
    didn't have before -- worth logging/reporting, since it's the concrete
    measure of how much this recovers versus how many courses are still
    apparently unoffered even after accounting for cross-listing."""
    keys = list(courses.keys())
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for k, c in courses.items():
        for sib in _crosslist_siblings(c.get("title", "")):
            if sib in courses:
                union(k, sib)

    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)

    merged = {k: dict(v) for k, v in courses.items()}
    gained = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        union_terms = set()
        for m in members:
            union_terms.update(courses[m].get("terms_offered", []))
        if not union_terms:
            continue
        for m in members:
            before = set(courses[m].get("terms_offered", []))
            if union_terms - before:
                gained += 1
            merged[m]["terms_offered"] = sorted(union_terms)

    return merged, gained


def refresh_snapshot(
    out_path: str,
    year: str,
    departments: tuple[str, ...] = DEFAULT_DEPARTMENTS,
    *,
    pace_s: float = 1.5,
    max_retries: int = 4,
    resume: bool = True,
    time_budget_s: float | None = None,
) -> dict:
    """Fetch course listings for the given departments/year via
    explorecourses and write a JSON snapshot to out_path. Returns the
    snapshot dict as well. Safe to re-run periodically (overrides are
    stored separately, see overrides.py usage in requirements/solver --
    they are never touched by this).

    Two things matter once `departments` is more than a handful (e.g. the
    full ~190-subject set from subjects_referenced_by_programs(), instead
    of the CS/EE/MATH starter list): explorecourses.stanford.edu can be
    flaky under load (real, observed 503s -- not something specific to any
    one environment, confirmed by hitting the same endpoint from a live
    browser too) and a run this size can take longer than one sandboxed
    shell call comfortably allows (bulletin.stanford.edu's own fetch hit
    this same wall, see README). So: each department gets `max_retries`
    attempts with exponential backoff (1x, 2x, 4x, 8x `pace_s`) before
    being logged as failed and skipped rather than aborting the whole run,
    every department is paced `pace_s` apart even on success (mirrors the
    "don't look like a bot" lesson learned fetching bulletin.stanford.edu),
    and -- when `resume=True` (the default) -- both an existing snapshot at
    `out_path` and a small sidecar state file next to it (`*.fetch_state.
    json`: which department codes have already been attempted, and which
    of those failed every retry) are loaded first, and any department
    already marked done is skipped outright. That sidecar is what makes
    "already attempted, zero real courses" distinguishable from "never
    tried" -- inferring it from the snapshot's contents alone can't do
    that. Both the snapshot and the state file are rewritten after every
    single department, not just at the end, mirroring fetch_bulletin.py's
    same fix for the same reason: a run that gets cut off mid-way (by
    `time_budget_s`, a hard error, or the sandboxed shell call's own
    ~180s-observed ceiling) never loses progress, and a plain re-run of
    the same call picks up exactly where it left off. Pass resume=False
    to ignore both and start clean. `time_budget_s`, if given, stops the
    loop (cleanly, having just saved) once that many seconds have elapsed
    since the call started, printing how many departments are still
    unattempted so the next invocation's scope is obvious."""
    connect = CourseConnection()
    start = time.monotonic()
    deadline = start + time_budget_s if time_budget_s is not None else None

    courses_out: dict[str, dict] = {}
    # Only ever contains departments that *succeeded* -- a failure (like
    # today's real explorecourses.stanford.edu 503s, confirmed outside this
    # environment too) is deliberately left retryable on the next resumed
    # run rather than marked "done", since most failures here are
    # transient service issues, not "this department doesn't exist."
    succeeded: set[str] = set()
    failed_departments: list[str] = []
    if resume:
        existing_path = Path(out_path)
        if existing_path.exists():
            try:
                courses_out = json.loads(existing_path.read_text()).get("courses", {})
            except Exception as e:
                print(f"  [warn] couldn't read existing snapshot at {out_path} to resume from: {e}")
        state_path = _state_path(out_path)
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                succeeded = set(state.get("succeeded", []))
                failed_departments = list(state.get("failed", []))
            except Exception as e:
                print(f"  [warn] couldn't read fetch state at {state_path} to resume from: {e}")
        # A department can only be "never succeeded" if it isn't currently
        # in `succeeded` -- prune any stale overlap from an older run
        # (e.g. one that failed once, then quietly succeeded with zero
        # courses on a later resumed run, before this reconciliation
        # existed) so the failure report stays accurate going forward.
        failed_departments = [d for d in failed_departments if d not in succeeded]

    def _save():
        snapshot = {"year": year, "departments": list(departments), "courses": courses_out}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(snapshot, indent=2))
        _state_path(out_path).write_text(json.dumps(
            {"succeeded": sorted(succeeded), "failed": sorted(set(failed_departments))}, indent=2
        ))
        return snapshot

    stopped_early = False
    remaining_at_stop: list[str] = []
    for dept_code in departments:
        if dept_code in succeeded:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            stopped_early = True
            remaining_at_stop = [d for d in departments if d not in succeeded]
            break
        dept_courses = None
        for attempt in range(max_retries):
            try:
                dept_courses = connect.get_courses_by_department(dept_code, year=year)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [warn] failed to fetch department {dept_code} after {max_retries} attempts: {e}")
                else:
                    backoff = pace_s * (2 ** attempt)
                    time.sleep(backoff)
        if dept_courses is None:
            failed_departments.append(dept_code)
            _save()
            time.sleep(pace_s)
            continue
        for c in dept_courses:
            terms_offered = sorted({_season(s.term) for s in c.sections}) or []
            sample_meetings = []
            for s in c.sections:
                if s.component in ("LEC", "SEM") and s.schedules:
                    sched = s.schedules[0]
                    if sched.days and sched.start_time:
                        sample_meetings.append({
                            "quarter": s.term,
                            "days": list(sched.days),
                            "start_time": sched.start_time,
                            "end_time": sched.end_time,
                        })
            prereq_text, prereq_courses, coreq_text, coreq_courses = extract_prereq_text(c.description or "")
            # c.gers is explorecourses' real, structured WAYS certification
            # field -- values look like "WAY-ER" (Ethical Reasoning),
            # "WAY-EDP" (Exploring Difference and Power), etc.; a course not
            # certified for any WAYS category returns ('',) rather than an
            # empty tuple, so the blank entries need filtering out. Confirmed
            # against real courses (COLLEGE/AFRICAAM/DANCE cross-lists) that
            # this is genuine per-course WAYS data, not a guess -- unlike
            # prerequisites, no heuristic text-extraction involved. One
            # category's raw code has an embedded hyphen of its own
            # ("WAY-A-II", Aesthetic and Interpretive Inquiry) that doesn't
            # match data/university_requirements.json's "AII" key -- strip
            # ALL hyphens after removing the "WAY-" prefix (a no-op for
            # every other category) rather than special-casing just this
            # one, in case Stanford ever adds another hyphenated code.
            ways_categories = tuple(sorted({
                g[4:].replace("-", "") for g in (c.gers or ()) if g.startswith("WAY-")
            }))
            course = Course(
                subject=c.subject,
                code=c.code.strip(),
                title=c.title,
                units_min=float(c.units_min),
                units_max=float(c.units_max),
                terms_offered=tuple(terms_offered),
                description=c.description or "",
                sample_meetings=tuple(MeetingPattern(**m) for m in sample_meetings[:4]),
                prerequisites_text=prereq_text,
                corequisites_text=coreq_text,
                prerequisite_courses=prereq_courses,
                corequisite_courses=coreq_courses,
                ways_categories=ways_categories,
            )
            courses_out[course.key] = _course_to_dict(course)
        succeeded.add(dept_code)
        # A department that failed on an earlier resumed run (e.g. during
        # a transient explorecourses.stanford.edu outage) but just
        # succeeded now shouldn't still show up as "never succeeded" in
        # the persisted report.
        failed_departments = [d for d in failed_departments if d != dept_code]
        print(f"  fetched {dept_code}: {len(dept_courses)} courses")
        _save()
        time.sleep(pace_s)

    if stopped_early:
        print(
            f"  stopped at time budget ({time_budget_s:.0f}s); "
            f"{len(remaining_at_stop)} department(s) still unattempted this run: {remaining_at_stop}",
        )
    if failed_departments:
        print(f"  [warn] {len(failed_departments)} department(s) never succeeded: {sorted(set(failed_departments))}")

    courses_out, gained = merge_cross_listed_terms(courses_out)
    if gained:
        print(f"  cross-listing merge: {gained} course(s) gained a term they were individually missing")

    return _save()


def _course_to_dict(c: Course) -> dict:
    d = asdict(c)
    return d


def _course_from_dict(d: dict) -> Course:
    meetings = tuple(MeetingPattern(**m) for m in d.get("sample_meetings", []))
    d2 = dict(d)
    d2["sample_meetings"] = meetings
    d2["terms_offered"] = tuple(d2["terms_offered"])
    d2["prerequisite_courses"] = tuple(d2.get("prerequisite_courses", ()))
    d2["corequisite_courses"] = tuple(d2.get("corequisite_courses", ()))
    # .get() with a default handles snapshots fetched before this field
    # existed -- they just come back with no WAYS data, not an error.
    d2["ways_categories"] = tuple(d2.get("ways_categories", ()))
    return Course(**d2)


def load_snapshot(path: str) -> dict[str, Course]:
    raw = json.loads(Path(path).read_text())
    return {key: _course_from_dict(d) for key, d in raw["courses"].items()}


_DEFAULT_OVERRIDES_PATH = "data/course_overrides.json"


def load_overrides(path: str = _DEFAULT_OVERRIDES_PATH) -> list[CourseOverride]:
    """Load manually-recorded corrections (see CourseOverride's docstring)
    from a JSON file: a plain list of {course_key, field, value, note}
    objects. Missing file -> no overrides, not an error, since a fresh
    checkout won't have any yet. The most common real use so far: a course
    that's genuinely taught (confirmed via several years of ExploreCourses
    history, e.g. ARCHLGY145's every-other-year Winter offering) but
    happened to have zero sections in whichever single academic year
    course_data.py's snapshot was fetched for -- recorded here instead of
    hand-editing the snapshot, so a future `refresh_snapshot()` re-run
    never silently discards the correction."""
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    return [CourseOverride(**o) for o in raw]


def save_override(override: CourseOverride, path: str = _DEFAULT_OVERRIDES_PATH) -> None:
    """Append one override to the JSON file (creating it if needed),
    de-duplicating on (course_key, field) -- a repeated correction for the
    same course+field replaces the earlier one rather than piling up."""
    p = Path(path)
    existing = load_overrides(path) if p.exists() else []
    existing = [o for o in existing if not (o.course_key == override.course_key and o.field == override.field)]
    existing.append(override)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(o) for o in existing], indent=2))


def apply_overrides(courses: dict[str, Course], overrides: list[CourseOverride]) -> dict[str, Course]:
    """Apply manual corrections on top of a loaded snapshot. Returns a new
    dict; never mutates the snapshot in place, so re-loading is always safe."""
    import copy
    courses = copy.deepcopy(courses)
    for ov in overrides:
        c = courses.get(ov.course_key)
        if c is None:
            print(f"  [warn] override for unknown course {ov.course_key}, skipping")
            continue
        if ov.field == "terms_offered":
            existing = set(c.terms_offered)
            existing.update(ov.value if isinstance(ov.value, (list, tuple)) else [ov.value])
            c.terms_offered = tuple(sorted(existing))
        elif hasattr(c, ov.field):
            setattr(c, ov.field, ov.value)
        else:
            print(f"  [warn] override field {ov.field} not recognized, skipping")
    return courses


# Hand-curated, best-effort list of courses known -- from real bulletin
# text, not guessed -- to require class standing beyond what explorecourses'
# own `description` text (and this file's prerequisite extraction, see
# extract_prereq_text/high_confidence_prerequisite_groups) ever captures.
# Stanford's Writing in the Major / Capstone enrollment restriction is
# enforced by Axess against a student's earned-units count, not stated as
# an ordinary course prerequisite -- so nothing already in this codebase
# would ever catch it. Same "don't guess, flag it" scope as every other
# hand-curated table here (the AP credit chart, multi_program.py's shared-
# exception allowlist): only a specific, confirmed course goes on this
# list, never a subject- or level-wide guess. Value is the minimum number
# of FULL academic years that must already be completed before the course
# may be scheduled (2 = junior standing, 3 = senior standing) -- see
# class_year_quarter_exclusions below for how this turns into actual
# quarter-index exclusions for a specific run's horizon.
KNOWN_CLASS_YEAR_MIN: dict[str, int] = {
    # User-confirmed directly: "CHEM176 is taken entirely by juniors and
    # seniors."
    "CHEM176": 2,
    # CS-BS's own bulletin text: "CS 191W, 194W, or 210B will satisfy both
    # the Writing in the Major (WiM) and Capstone requirements" -- Writing
    # in the Major is a university-wide requirement restricted to senior
    # standing (confirmed: Axess blocks WIM-tagged courses for students
    # under ~135 units), so all three real alternatives for this specific
    # slot share the same restriction regardless of which one a student
    # ends up taking.
    "CS191W": 3,
    "CS194W": 3,
    "CS210B": 3,
}


def class_year_quarter_exclusions(
    quarter_list: list[str],
    years_completed_at_horizon_start: int,
    restrictions: dict[str, int] | None = None,
) -> dict[str, set[int]]:
    """Turn KNOWN_CLASS_YEAR_MIN (or a caller-supplied override -- same
    shape) into the `course_quarter_exclusions` argument
    engine/solver.py's build_and_solve() expects, for one specific run's
    horizon.

    `years_completed_at_horizon_start`: how many FULL academic years the
    student has already completed as of quarter_list[0] (0 = starting
    fresh as a freshman, 1 = starting sophomore year, ...).

    Works for a horizon starting on ANY season, not just Autumn (app.py's
    own "Starting season" picker defaults to Winter) -- quarter_list[0]'s
    academic year is exactly `years_completed_at_horizon_start` by
    definition, regardless of which season it is; every LATER quarter at
    index qi (qi >= 1) belongs to one more completed year for each Autumn
    quarter that has occurred at some index j with 1 <= j <= qi (an Autumn
    quarter starting is exactly what "one more year now completed" means
    academically -- the previous Autumn-Winter-Spring block just
    finished). quarter_list[0] itself is never counted this way even if
    it's an Autumn quarter -- that one's just where the horizon starts, not
    a completed transition.

    A course requiring `min_years` is excluded from every quarter index
    where that year count is still below `min_years`; a course requiring
    fewer years than the horizon already starts with gets no exclusions at
    all (nothing to exclude)."""
    restrictions = restrictions if restrictions is not None else KNOWN_CLASS_YEAR_MIN
    years_completed_by_qi: list[int] = []
    running = years_completed_at_horizon_start
    for qi, q in enumerate(quarter_list):
        if qi >= 1 and _season(q) == "Autumn":
            running += 1
        years_completed_by_qi.append(running)
    exclusions: dict[str, set[int]] = {}
    for key, min_years in restrictions.items():
        excluded = {qi for qi, yrs in enumerate(years_completed_by_qi) if yrs < min_years}
        if excluded:
            exclusions[key] = excluded
    return exclusions


def add_placeholder_courses(
    courses: dict[str, Course], placeholders: dict[str, str | Course]
) -> dict[str, Course]:
    """Merge in placeholder courses created by bulletin_parser.py (see
    models.make_placeholder_course) for requirements whose bulletin text
    named no specific real course. Without this, a requirement rule can
    reference a "PLACEHOLDER:..." key that isn't in the course dict at
    all, so the solver would just treat it as unschedulable (silently
    landing in ScheduleResult.unplaced_required) instead of actually
    scheduling a visible stand-in. Returns a new dict; never mutates the
    snapshot in place.

    Each value is normally a bare description string (the vast majority of
    placeholders: a free-text requirement with no enumerated course list,
    where make_placeholder_course()'s default 3-5 unit guess is genuinely
    the best anyone can do). A caller that already knows real, non-guessed
    units for its placeholder (e.g. the language-requirement placeholder --
    Stanford's language sequence is consistently 5 units, not a 3-5 guess)
    can instead pass the already-built Course object directly, and it's
    used as-is rather than rebuilt with the generic defaults -- confirmed
    this was a real bug otherwise: building the language placeholder with
    units_typical=5.0 upstream did nothing, because this function used to
    unconditionally call make_placeholder_course(key, value) itself,
    silently discarding whatever Course the caller had already built and
    replacing it with a fresh 3-5/4-unit one."""
    if not placeholders:
        return courses
    merged = dict(courses)
    for key, value in placeholders.items():
        if key not in merged:
            merged[key] = value if isinstance(value, Course) else make_placeholder_course(key, value)
    return merged


if __name__ == "__main__":
    # Usage:
    #   python -m engine.course_data                                  # DEFAULT_DEPARTMENTS only
    #   python -m engine.course_data 2026-2027 data/courses_snapshot.json --all-referenced
    #   python -m engine.course_data 2026-2027 data/courses_snapshot.json --all-referenced --time-budget=150
    #
    # --all-referenced fetches every subject any of the 218 in-scope program
    # sheets actually references (see subjects_referenced_by_programs()),
    # not just the CS/EE/MATH starter set -- this is the real full-coverage
    # run. It's resumable and time-budgeted (see refresh_snapshot()'s
    # docstring): safe to re-run the exact same command repeatedly until
    # the "still unattempted" list it prints is empty.
    import sys
    args = sys.argv[1:]
    time_budget_s = 150.0
    for a in list(args):
        if a.startswith("--time-budget="):
            time_budget_s = float(a.split("=", 1)[1])
            args.remove(a)
    all_referenced = "--all-referenced" in args
    if all_referenced:
        args.remove("--all-referenced")

    year = args[0] if len(args) > 0 else "2026-2027"
    out = args[1] if len(args) > 1 else "data/courses_snapshot.json"
    departments = subjects_referenced_by_programs() if all_referenced else DEFAULT_DEPARTMENTS
    print(f"fetching {len(departments)} department(s) for {year} -> {out} (time budget {time_budget_s:.0f}s)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    snap = refresh_snapshot(out, year, departments, time_budget_s=time_budget_s)
    print(f"\nWrote {len(snap['courses'])} courses to {out}")
