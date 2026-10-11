This tool leverages Google OR-Tools CP-SAT to optimize theoretical Stanford program schedules for undergrad/master's students.

This is the first relatively working demo! I'll keep adding features and making architectural changes for a bit, so instructions will come later. Here's the rundown: Run the python code through streamlit. Upload your unofficial transcript. Pick your programs (majors, minors, coterms) OR add a .txt file of your intended courses.



# Stanford Schedule Generator

Local app that plans a multi-quarter Stanford course schedule, optimized
against a weighted preference table, starting from an uploaded transcript.

## Setup

- `pip install -r requirements.txt`
- Optional: set `ANTHROPIC_API_KEY` in the environment (only needed for the
  natural-language change box; everything else works without it)

## Run

- `streamlit run app.py`
- Opens a local page (usually `localhost:8501`): upload transcript, pick
  program sheets, edit the weight table, generate

## Refreshing course data

- Current snapshot is catalog year 2026-2027
- Refresh: `python -m engine.course_data 2026-2027 data/courses_snapshot.json`
- Add departments to fetch by editing `DEFAULT_DEPARTMENTS` in
  `engine/course_data.py`

## Adding/refreshing a program sheet

- Fetch one: `python -m engine.fetch_bulletin {CODE}` -- writes
  `samples/{CODE}.txt`, updates `data/course_id_cache.json`. Resumable if
  it times out; re-run the same command to pick up where it left off.
- Fetch every missing in-scope program: `python -m engine.fetch_bulletin --all-in-scope`
- `PROGRAM_SHEET_BUILDERS` (`engine/requirements.py`) auto-registers every
  `samples/*.txt` file -- only add a `_SPECIAL_PROGRAM_SHEET_BUILDERS`
  entry for a program that needs a specific default subplan/track
- Check console `[warn]` lines after parsing -- flags anything the parser
  couldn't confidently resolve (placeholder courses, etc.)

## What's implemented

- **Course data**: `explorecourses` snapshot (`data/courses_snapshot.json`),
  manual corrections layered via `data/course_overrides.json`
- **Transcript**: PDF parsing (`engine/transcript.py`) into completed
  courses + AP credit
- **AP credit**: score -> course-credit substitution
  (`engine/ap_credit.py`)
- **Program sheets**: parsed from bulletin.stanford.edu text
  (`engine/bulletin_parser.py`); some programs needed hand-written fixes
  (`engine/manual_fix_batch*.py`, `manual_fix_chem_bs.py`)
- **Track/subplan selection**: `list_track_options()`, `build_program_track()`,
  `solve_best_track()` in `engine/requirements.py` -- `solve_best_track`
  tries every valid track and keeps the best-objective result. Not yet
  wired into `app.py`'s UI (only hand-picked subplan variants are
  selectable today)
- **Double major / double degree stacking** (`engine/multi_program.py`):
  secondary major / double major / double degree modes, with a
  double-counting restriction between two majors' elective pools (shared
  intro math/physics/chem/language/ENGR courses are exempted)
- **University-wide requirements** (`engine/university_requirements.py`):
  WAYS, COLLEGE, Writing (WR1/WR2), Language -- generic placeholders,
  credited from completed courses and from any other course already
  required elsewhere in the same solve
- **Prerequisites**: high-confidence extraction from bulletin text
  (`engine/course_data.py`), hard-enforced in the solver -- except CS
  department courses, which are soft-nudged only (CS doesn't enforce its
  own prerequisites in Axess); see `ScheduleResult.prerequisite_warnings`
- **Corequisites**: hard-enforced, same quarter
- **Typical ordering** (`engine/typical_ordering.py`): soft nudge for
  Roman-numeral course sequences ("... I" before "... II"), not a hard
  prerequisite
- **Class-year restrictions** (`engine/course_data.py`:
  `KNOWN_CLASS_YEAR_MIN` / `class_year_quarter_exclusions`): hand-curated
  list of courses that require junior/senior standing, hard-excluded from
  earlier quarters
- **Large elective pools**: pools with more than ~8 real alternatives get
  rewritten into a generic named placeholder instead of the solver picking
  one arbitrary real course (`placeholderize_large_elective_pools` in
  `engine/requirements.py`)
- **Manual course list mode**: type or upload a `.txt` of exact course
  codes; prerequisites for courses in that list are assumed satisfied
  unless another listed course is the actual prerequisite
- **Weight table** (`engine/weights.py`): `day_concentration`,
  `subject_timing`, `course_timing`, `interest_boost`, `unit_load_balance`,
  `time_of_day`, `department_diversity`, `weekday_spread`
- **Natural-language change box** (`engine/nl_change.py`): plain-English
  request -> `CourseOverride` or `WeightRule` via an Anthropic API call;
  queued, applied on the next Generate click
- **Solver** (`engine/solver.py`): real CP-SAT (OR-Tools) -- unit cap per
  quarter, no duplicate enrollment (except designated repeat-eligible
  courses), no meeting-time conflicts, corequisites same-quarter,
  prerequisites (non-CS); proves optimality or reports it ran out of time

## Known gaps

- Track/subplan picker (`solve_best_track`, etc.) not wired into `app.py`'s UI
- Prerequisite/corequisite extraction is heuristic text-matching, not a
  real parser
- Program-sheet AND/OR-connector parsing is heuristic; some programs
  needed hand-written fixes
- Units/GPA/residency/pass-fail/coterm status aren't solver constraints
  (informational only)

## Project layout

```
app.py                          Streamlit UI
engine/
  models.py                     Core dataclasses (Course, CompletedCourse via transcript.py,
                                 RequirementRule, WeightRule, CourseOverride)
  transcript.py                 PDF transcript -> completed courses
  ap_credit.py                  AP score -> real course-credit chart
  course_data.py                explorecourses -> JSON snapshot, override application,
                                 prerequisite extraction, class-year restrictions
  requirements.py               Program-sheet requirement trees (hand-encoded + parsed),
                                 large-elective-pool placeholderizing
  bulletin_parser.py            bulletin.stanford.edu program-page text -> RequirementRule tree
  fetch_bulletin.py             No-browser fetcher: bulletin.stanford.edu page -> samples/{CODE}.txt
  multi_program.py              Double major / double degree stacking
  university_requirements.py    WAYS / COLLEGE / Writing / Language
  typical_ordering.py           Soft "Part I before Part II" nudges
  weights.py                    Weight-table scoring functions + schema
  nl_change.py                  Natural-language change box (Anthropic API)
  solver.py                     OR-Tools CP-SAT model + solve
data/
  courses_snapshot.json         Real course data snapshot
  course_overrides.json         Manual corrections layered on top
  university_requirements.json  University-wide requirements reference data
  bulletin_program_codes.json   Master list of bulletin program codes
  course_id_cache.json          Course id -> "SUBJ### - Title" cache for fetch_bulletin.py
samples/
  *.txt                          Cached bulletin page-text fixtures, one per program sheet
smoke_test.py                   End-to-end sanity check script (no UI)
```
