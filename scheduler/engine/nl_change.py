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

from .models import CourseOverride, WeightRule
from .weights import WEIGHT_TYPE_SCHEMA

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
    "unrecognized" -- exactly one of `course_override`/`weight_rule` is
    set for the first two; `error` is set (a plain-English, user-facing
    reason) whenever `change_type == "unrecognized"`, including for
    outright failures (no API key, network error, etc.) so the caller has
    one field to check and display either way. `summary` is always
    present -- a one-line human-readable description for the change log,
    even on failure (falls back to the original sentence)."""
    change_type: str
    summary: str
    course_override: Optional[CourseOverride] = None
    weight_rule: Optional[WeightRule] = None
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


def _system_prompt() -> str:
    return f"""You parse ONE plain-English request from a Stanford student who is \
using a course-schedule planning tool, and turn it into exactly one structured \
change via the {TOOL_NAME} tool. Always call that tool -- never reply in plain text.

There are exactly two kinds of change this tool can represent, plus a \
fallback for anything else:

1. course_override -- a FACT about one specific real course, or a personal \
exception that applies to THIS student only (not a general scheduling \
preference). Course keys are the subject code and catalog number \
concatenated with no space, e.g. "CS106A", "COLLEGE102", "MATH51". Valid \
fields:
   - terms_offered: which seasons a course is really offered (a list drawn \
     from {list(_VALID_SEASONS)}) -- use when the student says a course is \
     also/only offered in some quarter that the current data seems to miss.
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

3. unrecognized -- use this whenever the request doesn't clearly fit \
either shape above, asks to change something this tool has no \
representation for (e.g. a hard graduation requirement, adding a whole new \
major, something needing a human advisor), or is too vague to act on \
confidently. Set unrecognized_reason to a short, direct, plain-English \
explanation you'd be comfortable showing the student directly.

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
                    "enum": ["course_override", "weight_rule", "unrecognized"],
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
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_system_prompt(),
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
    reason = raw.get("unrecognized_reason") or "Couldn't map that request to a course fact or a scheduling preference."
    return ParsedChange(change_type="unrecognized", summary=summary, error=reason)
