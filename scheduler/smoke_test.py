import sys
from engine.transcript import parse_transcript_pdf
from engine.course_data import load_snapshot, add_placeholder_courses
from engine.requirements import build_requirements, PROGRAM_SHEET_BUILDERS, build_cs_bs_core_requirements
from engine.models import WeightRule

TRANSCRIPT_PATH = "/sessions/compassionate-eager-ptolemy/mnt/Stanford Course Generator/Transcript.pdf"
SNAPSHOT_PATH = "data/courses_snapshot.json"

completed_rows, ap_credits = parse_transcript_pdf(TRANSCRIPT_PATH)
completed_keys = {c.key for c in completed_rows}   # in-progress courses count as "spoken for" too
print(f"Completed/in-progress from transcript: {sorted(completed_keys)}")

courses = load_snapshot(SNAPSHOT_PATH)
print(f"Courses in snapshot: {len(courses)}")

# Sanity-check every registered program sheet builds without error before
# running the main solve below (catches parser/registry regressions early).
for name in PROGRAM_SHEET_BUILDERS:
    ph: dict[str, str] = {}
    built = build_requirements([name], out_placeholders=ph)
    extra = f", {len(ph)} placeholder course(s)" if ph else ""
    print(f"  [ok] {name!r} -> {len(built)} top-level rule(s){extra}")

rules = build_cs_bs_core_requirements()
for r in rules:
    print(" rule:", r.label, r.kind, r.courses, r.n)

quarter_list = [
    "2026-2027 Winter", "2026-2027 Spring",
    "2027-2028 Autumn", "2027-2028 Winter", "2027-2028 Spring",
    "2028-2029 Autumn", "2028-2029 Winter", "2028-2029 Spring",
]

weight_rules = [
    WeightRule(type="subject_timing", args={"subject": "CS", "phase": "early"}, weight=3, label="front-load CS core"),
    WeightRule(type="interest_boost", args={"course": "CS221"}, weight=5, label="want to take AI"),
]

from engine.solver import build_and_solve
result = build_and_solve(
    courses=courses,
    completed_keys=completed_keys,
    requirement_rules=rules,
    weight_rules=weight_rules,
    quarter_list=quarter_list,
    unit_cap_per_quarter=18,
    time_limit_s=20,
)

print("\nSTATUS:", result.status)
print("OBJECTIVE:", result.objective_value)
print("SOLVE TIME:", result.solve_time_s)
print("UNPLACED (no valid quarter var at all):", result.unplaced_required)
print("\nASSIGNMENT:")
for key, q in sorted(result.assignment.items(), key=lambda kv: quarter_list.index(kv[1])):
    print(f"  {q:20s} {key}  ({courses[key].title})")

# --- Second pass: EE-BS (parsed, full) ------------------------------------
# This specifically exercises two things the hand-encoded CS-BS rules above
# never touch: AND-group alternatives (e.g. "MATH51 AND MATH53" OR "CME100
# AND CME102", one Mathematics Sequence choice among several) and nested
# sub_rules (e.g. "Mathematics" = ALL of {Calculus, Mathematics Sequence,
# EE Math, Statistics/Probability}) -- both only meaningful once a real
# parsed program sheet, not the flat hand-encoded subset, is run through
# the solver.
print("\n" + "=" * 70)
print("EE-BS (parsed, full) -- exercising AND-groups + nested sub_rules")
ee_rules = build_requirements(["EE-BS (parsed, full)"])
# Drop "Electives": the bulletin page's own text for this rule is
# genuinely open-ended ("any letter-graded lecture/lab EE course"), not a
# finite list -- the parser (correctly, per its design) only captured the
# 4 explicitly-named example courses, whose units can't reach the stated
# 24-unit minimum. That's a real gap in the SOURCE text's machine-
# parseability, not a solver bug -- excluded here so this smoke test
# isolates the thing it's actually meant to check.
ee_rules = [r for r in ee_rules if r.label != "Electives"]
ee_result = build_and_solve(
    courses=courses,
    completed_keys=completed_keys,
    requirement_rules=ee_rules,
    weight_rules=[],
    quarter_list=quarter_list,
    unit_cap_per_quarter=20,
    time_limit_s=20,
)
print("STATUS:", ee_result.status)
print("OBJECTIVE:", ee_result.objective_value)
print("UNPLACED (no valid quarter var at all):", ee_result.unplaced_required)
math_seq_courses = {"MATH51", "MATH53", "CME100", "CME102", "MATH52"}
taken_math_seq = sorted(k for k in (completed_keys | set(ee_result.assignment)) if k in math_seq_courses)
print("Mathematics Sequence courses taken/completed:", taken_math_seq)
print("\nASSIGNMENT:")
for key, q in sorted(ee_result.assignment.items(), key=lambda kv: quarter_list.index(kv[1])):
    print(f"  {q:20s} {key}  ({courses[key].title})")

# --- Third pass: MATH-MIN (parsed, full) ----------------------------------
# This program sheet's entire requirement is free text ("six math courses
# at or above Math 51") with no enumerated course list at all -- it
# specifically exercises the placeholder-course path: build_requirements'
# out_placeholders, course_data.add_placeholder_courses() merging them into
# the course dict, and the solver actually scheduling them (not just
# leaving them unplaced) since they're marked as offered every quarter.
print("\n" + "=" * 70)
print("MATH-MIN (parsed, full) -- exercising placeholder courses")
math_min_placeholders: dict[str, str] = {}
math_min_rules = build_requirements(["MATH-MIN (parsed, full)"], out_placeholders=math_min_placeholders)
print(f"placeholders created: {len(math_min_placeholders)}")
for k, v in math_min_placeholders.items():
    print(f"  {k} -> {v}")
math_min_courses = add_placeholder_courses(courses, math_min_placeholders)
math_min_result = build_and_solve(
    courses=math_min_courses,
    completed_keys=completed_keys,
    requirement_rules=math_min_rules,
    weight_rules=[],
    quarter_list=quarter_list,
    unit_cap_per_quarter=20,
    time_limit_s=20,
)
print("STATUS:", math_min_result.status)
print("UNPLACED (no valid quarter var at all):", math_min_result.unplaced_required)
print("\nASSIGNMENT:")
for key, q in sorted(math_min_result.assignment.items(), key=lambda kv: quarter_list.index(kv[1])):
    print(f"  {q:20s} {key}  ({math_min_courses[key].title})")
