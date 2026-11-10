"""
Turn a user's plain-English request for a small schedule change into a
structured `CourseOverride` or `WeightRule`, via a real Anthropic API call
-- this is the "natural-language override box" the README's "Stubbed"
section flagged as a natural next step (see its "No natural-language
override box yet" entries).

Design rules this module and its caller (app.py) MUST follow, matching the
prerequisite drop-for-me feature's own precedent:
  - A parsed change is a PERSONAL fact about this user's session, never a
    correction to the shared course data. The caller is responsible for
    appending the result to `st.session_state.session_overrides` /
    `st.session_state.weight_rows` (in-memory, per-session) -- this module
    itself never touches disk, never calls `save_override()`, and doesn't
    even import Streamlit.
  - Parsing does NOT trigger a re-solve. This module has no idea a solver
    exists; it just returns a structured `ParsedChange` for the caller to
    log and queue. Multiple requests can queue up before the next
    "Generate schedule" click actually applies them all at once.

Runs as a normal local Python process (this app isn't a sandboxed browser
artifact), so a real API call here is fine. Requires the `anthropic`
package and an `ANTHROPIC_API_KEY` environment variable set in the shell
this app is launched from -- both are treated as optional: if either is
missing, `parse_change_request()` returns a clear, actionable error
instead of raising, so the rest of the app keeps working with this one
feature simply unavailable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    # Walks up from the current working directory looking for a .env file
    # -- finds the project root's .env (one level up from this scheduler/
    # folder) even when `streamlit run app.py` is launched from inside
    # scheduler/. Doesn't override a real shell-exported env var if one
    # already exists (override=False is the default). A bare `os.environ`
    # set in the real shell works fine without this -- this only matters
    # for the .env-file case.
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed -- only a real shell env var will work

import re

from .models import (
    CourseGrouping, CourseOverride, QuarterRestriction, RequirementRemoval, RequirementSubstitution, WeightRule,
)
from .weights import WEIGHT_TYPE_SCHEMA

_CLASS_YEAR_NAMES = ("Freshman", "Sophomore", "Junior", "Senior")

# A cheap, fast model is the right fit here -- this is a single-sentence
# classification+extraction task with a small, fixed set of possible
# outputs (one CourseOverride field or one WeightRule type), not open-
# ended reasoning. Overridable via env var without a code change, same
# spirit as this project's other environment-driven config (e.g.
# SNAPSHOT_PATH's neighbors).
DEFAULT_MODEL = os.environ.get("SCHEDULER_NL_MODEL", "claude-haiku-4-5-20251001")

TOOL_NAME = "record_schedule_change"

# Kept in sync with Course's real fields (see engine/models.py) -- these
# are exactly the fields apply_overrides() already knows how to set, minus
# `subject`/`code` (identity, never a "correction") and `sample_meetings`/
# `ways_categories` (structured data with no sensible free-text edit).
_VALID_OVERRIDE_FIELDS = (
    "terms_offered",
    "units_min",
    "units_max",
    "units_typical_override",
    "title",
    "description",
    "prerequisite_courses",
    "corequisite_courses",
)
_FLOAT_FIELDS = ("units_min", "units_max", "units_typical_override")
_COURSE_LIST_FIELDS = ("prerequisite_courses", "corequisite_courses")
_VALID_SEASONS = ("Autumn", "Winter", "Spring", "Summer")


@dataclass
class ParsedChange:
    """The result of parsing one natural-language change request.

    `change_type` is always one of "course_override" / "weight_rule" /
    "requirement_substitution" / "quarter_restriction" / "unrecognized" --
    exactly one of the four typed fields is set for the first four;
    `error` is set (a plain-English, user-facing reason) whenever
    `change_type == "unrecognized"`, including for outright failures (no
    API key, network error, etc.) so the caller has one field to check and
    display either way. `summary` is always present -- a one-line
    human-readable description for the change log, even on failure (falls
    back to the original sentence)."""
    change_type: str
    summary: str
    course_override: Optional[CourseOverride] = None
    weight_rule: Optional[WeightRule] = None
    requirement_substitution: Optional[RequirementSubstitution] = None
    quarter_restriction: Optional[QuarterRestriction] = None
    requirement_removal: Optional[RequirementRemoval] = None
    course_grouping: Optional[CourseGrouping] = None
    error: Optional[str] = None


def _describe_weight_types() -> str:
    """Dynamically render every registered weight type + its argument
    schema from WEIGHT_TYPE_SCHEMA, so this prompt never drifts out of
    sync with what the app actually supports -- same "don't hand-list
    what a registry already knows" philosophy as PROGRAM_SHEET_BUILDERS'
    own auto-discovery."""
    lines = []
    for wt, info in sorted(WEIGHT_TYPE_SCHEMA.items()):
        arg_bits = []
        for arg_name, spec in info["args"].items():
            kind = spec["kind"]
            if kind == "enum":
                arg_bits.append(f"{arg_name}=one of {list(spec['options'])}")
            elif kind == "multi_enum":
                arg_bits.append(f"{arg_name}=subset of {list(spec['options'])}")
            elif kind == "number":
                arg_bits.append(f"{arg_name}=number ({spec.get('min', '?')}-{spec.get('max', '?')})")
            elif kind == "course_key":
                arg_bits.append(f"{arg_name}=a course key like 'CS221'")
            elif kind == "subject":
                arg_bits.append(f"{arg_name}=a subject code like 'CS'")
        args_desc = "; ".join(arg_bits) if arg_bits else "no arguments"
        lines.append(f'  "{wt}": {info["description"]} Arguments: {args_desc}.')
    return "\n".join(lines)


_COURSE_KEY_RE = re.compile(r"\b([A-Za-z]{2,10})\s*0*(\d{1,3}[A-Za-z]{0,2})\b")


def _find_mentioned_courses(sentence: str, courses: Optional[dict]) -> dict:
    """Best-effort scan for real course keys referenced in `sentence` (e.g.
    "CS 229" or "CS229" both match "CS229"), returning
    {course_key: current_terms_offered} for any that exist in `courses`.
    Used so the LLM can see a course's CURRENT terms_offered before deciding
    the full corrected list -- without this, "also offered in Spring" and
    "only offered in Spring" look identical to a model with no memory of
    what's already on file."""
    if not courses:
        return {}
    found = {}
    for m in _COURSE_KEY_RE.finditer(sentence):
        key = f"{m.group(1).upper()}{m.group(2).upper()}"
        course = courses.get(key)
        if course is not None:
            found[key] = tuple(getattr(course, "terms_offered", ()) or ())
    return found


def _system_prompt(mentioned_courses: Optional[dict] = None, rule_labels: Optional[list] = None) -> str:
    context_block = ""
    if rule_labels:
        context_block += "\n\nRequirement labels currently in this student's plan (for requirement_removal's rule_label -- copy one EXACTLY):\n" + "\n".join(f"  - {l}" for l in rule_labels[:400]) + "\n"
    if mentioned_courses:
        lines = "\n".join(
            f"  - {key}: currently {list(terms) or '[]'}"
            for key, terms in mentioned_courses.items()
        )
        context_block += f"""

Current data on file for course(s) mentioned in the request below (use this \
to compute the FULL correct terms_offered list -- it is always a complete \
replacement of what's on file, never just the newly-mentioned season(s)):
{lines}
"""
    return f"""You parse ONE plain-English request from a Stanford student who is \
using a course-schedule planning tool, and turn it into exactly one structured \
change via the {TOOL_NAME} tool. Always call that tool -- never reply in plain text.
{context_block}

There are exactly six kinds of change this tool can represent, plus a \
fallback for anything else.

TRUST THE STUDENT. They know their own petitions, waivers, advisor \
approvals and personal circumstances. Never refuse a request because it \
"needs an advisor" or "is a program-level change" -- if it fits one of the \
shapes below, record it. Only use `unrecognized` when the request truly \
can't be expressed by any shape below, or is too ambiguous to act on.

1. course_override -- a FACT about one specific real course, or a personal \
exception that applies to THIS student only (not a general scheduling \
preference). Course keys are the subject code and catalog number \
concatenated with no space, e.g. "CS106A", "COLLEGE102", "MATH51". Valid \
fields:
   - terms_offered: which seasons a course is really offered (a list drawn \
     from {list(_VALID_SEASONS)}) -- use when the student says a course is \
     also/only offered in some quarter that the current data seems to miss, \
     OR when the student wants to personally restrict a course to fewer \
     quarters than it's actually offered (e.g. "only plan CS229 for Winter \
     for me"). The value you record is ALWAYS the complete final list of \
     seasons, not a delta: if the student says "also" offered somewhere, \
     that season gets ADDED to the course's current terms (shown above, if \
     given); if the student says "only"/"restrict to" a season, the final \
     list contains ONLY that season and nothing else, even if the course is \
     really offered in other seasons too -- this is a plain replacement, \
     applied exactly as recorded.
   - prerequisite_courses / corequisite_courses: a list of course keys, or \
     an EMPTY list to mean "waive this entirely for me" (e.g. AP credit, a \
     transfer course, a placement exam, "I already know this material").
   - units_min / units_max / units_typical_override: numbers, when the \
     student says a course's real unit count differs from what's shown.
   - title / description: free text, for a correction to how a course is \
     labeled/described.
   Only use course_override for something that's true about the COURSE \
   ITSELF or a prerequisite/corequisite waiver -- never for a general \
   preference like "I like morning classes" (that's a weight_rule below).

2. weight_rule -- a scheduling PREFERENCE (nice-to-have, not a hard fact), \
matching exactly one of these existing types. `args` must use exactly the \
argument names listed for that type, nothing else:
{_describe_weight_types()}
   Pick a `weight` between 1 and 10 reflecting how strongly the student \
   expressed it (a mild "I'd rather" is low, e.g. 2-3; "I really need"/"must" \
   is high, e.g. 7-10); default to 3 if there's no signal either way.

3. requirement_substitution -- the student wants ONE specific real course \
to count in place of (or as an extra alternative for) another specific \
real course, somewhere in THEIR OWN plan, for personal reasons (an \
upcoming curriculum change they already know about, a department-approved \
swap, equivalent content, etc.) -- e.g. "let CHEM141 count in place of \
CHEM181, it's planned in next year's version of the chemistry major." \
This is NOT "beyond scope" or a program-level edit -- it's a personal, \
this-session-only adjustment, exactly as in-bounds as a prerequisite \
waiver (course_override's prerequisite_courses field above), just applied \
to a degree requirement instead of a prerequisite. Fields: required_key \
(the course currently required) and substitute_key (the course that \
should now also satisfy that same slot). Only use this when the student \
names TWO specific real courses and a clear "instead of" / "in place of" / \
"counts for" relationship -- if they're vague about which courses, or \
describe a request that touches a whole subplan/multiple requirements at \
once, prefer `unrecognized`.

4. quarter_restriction -- the student wants ONE specific real course \
pinned to an EXACT (class year, season) target, stated with "no \
compromises" certainty -- e.g. "enforce CHEM171 in sophomore spring." \
Fields: course_key, class_year (one of {list(_CLASS_YEAR_NAMES)}), season \
(one of {list(_VALID_SEASONS)}). Do NOT use course_override's \
terms_offered for this kind of request even if it mentions a season --  \
terms_offered only restricts which SEASONS a course could ever run in \
(e.g. "Spring only"), which does nothing to pin WHICH year's Spring; a \
request that names a specific class year (freshman/sophomore/junior/ \
senior, or "this year"/"next year" relative to where the student already \
is) needs quarter_restriction instead. If the student only names a \
season with no class year at all ("I only want to take this in Winter"), \
that's course_override's terms_offered, not this.

5. requirement_removal -- the student says a requirement doesn't apply to \
them and should be dropped (a waiver, petition, transfer credit, advisor \
approval -- you don't need a reason). Fill EXACTLY ONE of:
   - course_key: a specific required course to drop, e.g. "remove the CS221 \
     requirement" -> CS221. This waives every required slot for that course \
     (it does not ban the course or remove it from elective lists).
   - rule_label: a whole named requirement, e.g. "drop the Track C \
     requirement", "I don't need the language requirement". Copy the label \
     EXACTLY from the requirement labels listed above; if none clearly \
     matches, use `unrecognized` and say which labels were closest.

6. course_grouping -- two or more specific courses must be taken in the \
same academic year (scope "year", e.g. "CHINLANG2 and CHINLANG3 must be in \
the same year") or the same quarter (scope "quarter", e.g. "take CHEM33 and \
CHEM36 together"). Fields: course_keys (2+ course keys), scope.

7. unrecognized -- use this only when the request doesn't fit any shape \
above (e.g. adding a whole new major), or is too vague to act on confidently. Set \
unrecognized_reason to a short, direct, plain-English explanation you'd \
be comfortable showing the student directly.

Always fill in `summary`: one short, self-contained sentence describing \
exactly what the change does, understandable without seeing the student's \
original wording (this is shown to them in a running change log).

Don't guess at a course key you're not confident about -- if the student \
names a course ambiguously or you're not sure of the exact code, prefer \
`unrecognized` and say so, rather than inventing a plausible-looking key."""


def _build_tool_schema() -> dict:
    return {
        "name": TOOL_NAME,
        "description": "Record your interpretation of the user's requested small change to their course plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "change_type": {
                    "type": "string",
                    "enum": [
                        "course_override", "weight_rule", "requirement_substitution",
                        "quarter_restriction", "requirement_removal", "course_grouping", "unrecognized",
                    ],
                },
                "summary": {
                    "type": "string",
                    "description": "One short, self-contained, plain-English sentence describing what this change does.",
                },
                "course_override": {
                    "type": "object",
                    "properties": {
                        "course_key": {"type": "string"},
                        "field": {"type": "string", "enum": list(_VALID_OVERRIDE_FIELDS)},
                        "value": {
                            "description": (
                                "New value for `field`: a list of season names for "
                                "terms_offered; a list of course keys (possibly empty) "
                                "for prerequisite_courses/corequisite_courses; a number "
                                "for units_min/units_max/units_typical_override; a "
                                "string for title/description."
                            )
                        },
                        "note": {"type": "string"},
                    },
                    "required": ["course_key", "field", "value"],
                },
                "weight_rule": {
                    "type": "object",
                    "properties": {
                        "weight_type": {"type": "string", "enum": sorted(WEIGHT_TYPE_SCHEMA.keys())},
                        "args": {"type": "object"},
                        "weight": {"type": "number"},
                        "label": {"type": "string"},
                    },
                    "required": ["weight_type", "args"],
                },
                "requirement_substitution": {
                    "type": "object",
                    "properties": {
                        "required_key": {"type": "string"},
                        "substitute_key": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["required_key", "substitute_key"],
                },
                "quarter_restriction": {
                    "type": "object",
                    "properties": {
                        "course_key": {"type": "string"},
                        "class_year": {"type": "string", "enum": list(_CLASS_YEAR_NAMES)},
                        "season": {"type": "string", "enum": list(_VALID_SEASONS)},
                        "note": {"type": "string"},
                    },
                    "required": ["course_key", "class_year", "season"],
                },
                "requirement_removal": {
                    "type": "object",
                    "properties": {
                        "course_key": {"type": "string"},
                        "rule_label": {"type": "string"},
                        "note": {"type": "string"},
                    },
                },
                "course_grouping": {
                    "type": "object",
                    "properties": {
                        "course_keys": {"type": "array", "items": {"type": "string"}},
                        "scope": {"type": "string", "enum": ["year", "quarter"]},
                        "note": {"type": "string"},
                    },
                    "required": ["course_keys", "scope"],
                },
                "unrecognized_reason": {"type": "string"},
            },
            "required": ["change_type", "summary"],
        },
    }


def _coerce_override_value(field: str, value) -> object:
    if field == "terms_offered":
        seasons = value if isinstance(value, (list, tuple)) else [value]
        out = []
        for s in seasons:
            s = str(s).strip().title()
            if s not in _VALID_SEASONS:
                raise ValueError(f"{s!r} isn't a real season ({', '.join(_VALID_SEASONS)})")
            out.append(s)
        if not out:
            raise ValueError("terms_offered needs at least one season")
        return out
    if field in _FLOAT_FIELDS:
        return float(value)
    if field in _COURSE_LIST_FIELDS:
        keys = value if isinstance(value, (list, tuple)) else ([] if value in (None, "") else [value])
        return [str(k).strip().upper().replace(" ", "") for k in keys]
    return str(value)


def _build_course_override(raw: dict, sentence: str, summary: str, known_course_keys) -> ParsedChange:
    co = raw.get("course_override") or {}
    course_key = str(co.get("course_key", "")).strip().upper().replace(" ", "")
    field = co.get("field")
    if not course_key or field not in _VALID_OVERRIDE_FIELDS:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error="The model's course_override response was incomplete -- try rephrasing with the exact course code.",
        )
    if known_course_keys is not None and course_key not in known_course_keys:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error=f"Couldn't find {course_key!r} in the current course data -- double-check the course code.",
        )
    try:
        value = _coerce_override_value(field, co.get("value"))
    except (ValueError, TypeError) as e:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error=f"Couldn't apply that change to {course_key}: {e}",
        )
    note = co.get("note") or f"Added via natural-language request (this session only): {sentence!r}"
    return ParsedChange(
        change_type="course_override",
        summary=summary,
        course_override=CourseOverride(course_key=course_key, field=field, value=value, note=note),
    )


def _build_requirement_substitution(raw: dict, sentence: str, summary: str, known_course_keys) -> ParsedChange:
    rs = raw.get("requirement_substitution") or {}
    required_key = str(rs.get("required_key", "")).strip().upper().replace(" ", "")
    substitute_key = str(rs.get("substitute_key", "")).strip().upper().replace(" ", "")
    if not required_key or not substitute_key:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error="The model's requirement_substitution response was incomplete -- try naming both courses explicitly.",
        )
    if known_course_keys is not None:
        missing = [k for k in (required_key, substitute_key) if k not in known_course_keys]
        if missing:
            return ParsedChange(
                change_type="unrecognized", summary=summary,
                error=f"Couldn't find {', '.join(missing)!r} in the current course data -- double-check the course code(s).",
            )
    note = rs.get("note") or f"Added via natural-language request (this session only): {sentence!r}"
    return ParsedChange(
        change_type="requirement_substitution",
        summary=summary,
        requirement_substitution=RequirementSubstitution(
            required_key=required_key, substitute_key=substitute_key, note=note,
        ),
    )


def _build_quarter_restriction(raw: dict, sentence: str, summary: str, known_course_keys) -> ParsedChange:
    qr = raw.get("quarter_restriction") or {}
    course_key = str(qr.get("course_key", "")).strip().upper().replace(" ", "")
    class_year_name = str(qr.get("class_year", "")).strip().title()
    season = str(qr.get("season", "")).strip().title()
    if not course_key or class_year_name not in _CLASS_YEAR_NAMES or season not in _VALID_SEASONS:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error="The model's quarter_restriction response was incomplete -- try naming the course, "
                  "class year (Freshman/Sophomore/Junior/Senior), and season explicitly.",
        )
    if known_course_keys is not None and course_key not in known_course_keys:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error=f"Couldn't find {course_key!r} in the current course data -- double-check the course code.",
        )
    note = qr.get("note") or f"Added via natural-language request (this session only): {sentence!r}"
    return ParsedChange(
        change_type="quarter_restriction",
        summary=summary,
        quarter_restriction=QuarterRestriction(
            course_key=course_key, class_year=_CLASS_YEAR_NAMES.index(class_year_name), season=season, note=note,
        ),
    )


def _build_requirement_removal(raw: dict, sentence: str, summary: str, known_course_keys, rule_labels) -> ParsedChange:
    rr = raw.get("requirement_removal") or {}
    course_key = str(rr.get("course_key", "") or "").strip().upper().replace(" ", "")
    rule_label = str(rr.get("rule_label", "") or "").strip()
    note = rr.get("note") or f"Added via natural-language request (this session only): {sentence!r}"
    if bool(course_key) == bool(rule_label):
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error="Couldn't tell whether to drop a specific course or a whole requirement -- try naming one.",
        )
    if course_key:
        if known_course_keys is not None and course_key not in known_course_keys:
            return ParsedChange(
                change_type="unrecognized", summary=summary,
                error=f"Couldn't find {course_key!r} in the current course data -- double-check the course code.",
            )
        return ParsedChange(change_type="requirement_removal", summary=summary,
                            requirement_removal=RequirementRemoval(course_key=course_key, note=note))
    if rule_labels is not None:
        norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
        exact = [l for l in rule_labels if norm(l) == norm(rule_label)]
        if not exact:
            return ParsedChange(
                change_type="unrecognized", summary=summary,
                error=f"No requirement labeled {rule_label!r} in your current plan -- check the rule listing "
                      "under 2b/2c and use its exact name.",
            )
        rule_label = exact[0]
    return ParsedChange(change_type="requirement_removal", summary=summary,
                        requirement_removal=RequirementRemoval(rule_label=rule_label, note=note))


def _build_course_grouping(raw: dict, sentence: str, summary: str, known_course_keys) -> ParsedChange:
    cg = raw.get("course_grouping") or {}
    keys = list(dict.fromkeys(str(k).strip().upper().replace(" ", "") for k in (cg.get("course_keys") or []) if str(k).strip()))
    scope = str(cg.get("scope", "year")).strip().lower()
    if len(keys) < 2 or scope not in ("year", "quarter"):
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error="A same-year/same-quarter rule needs at least two course codes.",
        )
    if known_course_keys is not None:
        missing = [k for k in keys if k not in known_course_keys]
        if missing:
            return ParsedChange(
                change_type="unrecognized", summary=summary,
                error=f"Couldn't find {', '.join(missing)} in the current course data -- double-check the code(s).",
            )
    note = cg.get("note") or f"Added via natural-language request (this session only): {sentence!r}"
    return ParsedChange(change_type="course_grouping", summary=summary,
                        course_grouping=CourseGrouping(course_keys=keys, scope=scope, note=note))


def _coerce_weight_arg(arg_name: str, spec: dict, value):
    kind = spec["kind"]
    if kind == "enum":
        options = spec["options"]
        matches = [o for o in options if str(value).strip().lower() == o.lower()]
        if not matches:
            raise ValueError(f"{arg_name}={value!r} must be one of {list(options)}")
        return matches[0]
    if kind == "multi_enum":
        options = spec["options"]
        raw_vals = value if isinstance(value, (list, tuple)) else [value]
        out = []
        for v in raw_vals:
            matches = [o for o in options if str(v).strip().lower() == o.lower()]
            if not matches:
                raise ValueError(f"{arg_name} value {v!r} must be one of {list(options)}")
            out.append(matches[0])
        if not out:
            raise ValueError(f"{arg_name} needs at least one value")
        return out
    if kind == "number":
        return float(value)
    if kind == "course_key":
        return str(value).strip().upper().replace(" ", "")
    if kind == "subject":
        return str(value).strip().upper()
    return value


def _build_weight_rule(raw: dict, sentence: str, summary: str) -> ParsedChange:
    wr = raw.get("weight_rule") or {}
    weight_type = wr.get("weight_type")
    if weight_type not in WEIGHT_TYPE_SCHEMA:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error=f"{weight_type!r} isn't one of this app's existing preference types.",
        )
    schema_args = WEIGHT_TYPE_SCHEMA[weight_type]["args"]
    raw_args = wr.get("args") or {}
    coerced_args = {}
    try:
        for arg_name, spec in schema_args.items():
            if arg_name not in raw_args:
                raise ValueError(f"missing {arg_name!r} for {weight_type}")
            coerced_args[arg_name] = _coerce_weight_arg(arg_name, spec, raw_args[arg_name])
    except (ValueError, TypeError) as e:
        return ParsedChange(
            change_type="unrecognized", summary=summary,
            error=f"Couldn't build that preference: {e}",
        )
    try:
        weight = float(wr.get("weight", 3.0))
    except (ValueError, TypeError):
        weight = 3.0
    label = wr.get("label") or summary
    return ParsedChange(
        change_type="weight_rule",
        summary=summary,
        weight_rule=WeightRule(type=weight_type, args=coerced_args, weight=weight, label=label),
    )


def parse_change_request(
    sentence: str,
    *,
    known_course_keys: Optional[set] = None,
    courses: Optional[dict] = None,
    rule_labels: Optional[list] = None,
    model: str = DEFAULT_MODEL,
) -> ParsedChange:
    """The one entry point this module exposes. Never raises -- every
    failure mode (no `anthropic` package, no API key, a network/API error,
    a malformed or unconvincing model response) comes back as a
    `ParsedChange(change_type="unrecognized", error=...)` instead, so a
    caller can always just log `result.summary`/`result.error` and move on."""
    sentence = (sentence or "").strip()
    if not sentence:
        return ParsedChange(change_type="unrecognized", summary="(empty request)", error="Nothing was typed.")

    try:
        import anthropic
    except ImportError:
        return ParsedChange(
            change_type="unrecognized", summary=sentence,
            error="The `anthropic` package isn't installed -- run `pip install anthropic` to enable this feature.",
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ParsedChange(
            change_type="unrecognized", summary=sentence,
            error="No ANTHROPIC_API_KEY set in the environment this app was launched from -- "
                  "set it and restart the app to enable this feature.",
        )

    client = anthropic.Anthropic(api_key=api_key)
    tool = _build_tool_schema()
    mentioned_courses = _find_mentioned_courses(sentence, courses)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_system_prompt(mentioned_courses, rule_labels),
            tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": sentence}],
        )
    except Exception as e:
        return ParsedChange(change_type="unrecognized", summary=sentence, error=f"LLM request failed: {e}")

    raw = next(
        (block.input for block in resp.content if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME),
        None,
    )
    if raw is None:
        return ParsedChange(
            change_type="unrecognized", summary=sentence,
            error="The model didn't return a structured response -- try rephrasing.",
        )

    change_type = raw.get("change_type")
    summary = raw.get("summary") or sentence

    if change_type == "course_override":
        return _build_course_override(raw, sentence, summary, known_course_keys)
    if change_type == "weight_rule":
        return _build_weight_rule(raw, sentence, summary)
    if change_type == "requirement_substitution":
        return _build_requirement_substitution(raw, sentence, summary, known_course_keys)
    if change_type == "quarter_restriction":
        return _build_quarter_restriction(raw, sentence, summary, known_course_keys)
    if change_type == "requirement_removal":
        return _build_requirement_removal(raw, sentence, summary, known_course_keys, rule_labels)
    if change_type == "course_grouping":
        return _build_course_grouping(raw, sentence, summary, known_course_keys)
    reason = raw.get("unrecognized_reason") or "Couldn't map that request to a course fact or a scheduling preference."
    return ParsedChange(change_type="unrecognized", summary=summary, error=reason)
