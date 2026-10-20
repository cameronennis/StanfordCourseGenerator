"""
Basic local Streamlit app tying the engine together: upload a transcript,
pick program sheets, edit the weight table, generate a schedule.

Run with:  streamlit run app.py
(from inside this project's folder, after `pip install -r requirements.txt`)

v1 scope: this is deliberately minimal -- one program sheet (a CS-BS core
subset), a handful of weight types, no natural-language override box yet.
It's meant to prove the whole pipeline (transcript -> real course data ->
requirements -> weighted CP-SAT solve -> viewable schedule) works end to
end with real data, so the next round of refined parameters has a working
skeleton to slot into rather than a blank page.
"""
from __future__ import annotations

import difflib
import re
import tempfile
from pathlib import Path

import streamlit as st

from engine.course_data import (
    load_snapshot, apply_overrides, load_overrides,
    add_placeholder_courses, class_year_quarter_exclusions, DEFAULT_DEPARTMENTS,
)
from engine.models import CourseOverride, RequirementRule, WeightRule, all_referenced_course_keys, is_placeholder_key
from engine.requirements import (
    PROGRAM_SHEET_BUILDERS, BROKEN_PROGRAMS, build_requirements_grouped,
    list_variant_labels_by_code, placeholderize_large_elective_pools,
)
from engine.multi_program import (
    STACK_MODE_SECONDARY, STACK_MODE_DOUBLE_MAJOR, STACK_MODE_DOUBLE_DEGREE,
    build_restricted_course_programs, degree_type_of, is_undergrad_major_sheet,
    load_stack_modes, sheet_code, tag_program_id,
)
from engine.ap_credit import guess_ap_equivalencies, flatten_course_items
from engine.nl_change import parse_change_request
from engine.solver import build_and_solve
from engine.transcript import parse_transcript_pdf
from engine.university_requirements import build_all_university_rules
from engine.weights import WEIGHT_TYPE_SCHEMA

st.set_page_config(page_title="Stanford Schedule Generator", layout="wide")
st.title("Stanford theoretical course-schedule generator")
st.caption(
    "V1: Start with Transcript or  Manual List, Adjust Parameters, and Generate Plans. LLM requirement adjustment in progress."
)

SNAPSHOT_PATH = Path(__file__).parent / "data" / "courses_snapshot.json"
OVERRIDES_PATH = Path(__file__).parent / "data" / "course_overrides.json"


@st.cache_resource(show_spinner="Loading course snapshot...")
def _load_base_courses(snapshot_path: str, overrides_path: str) -> dict:
    """The ~15,000-course snapshot plus the canonical, checked-in overrides
    (see the comment on the call site below) -- everything about this is
    static until the app process restarts (or someone re-runs
    refresh_snapshot()/hand-edits course_overrides.json, which does need a
    restart to pick up, same as any other st.cache_resource). Streamlit
    reruns this ENTIRE script top-to-bottom on every single widget
    interaction -- every checkbox, every "Remove" click in section 2d --
    so without caching, parsing ~20MB of JSON into 15k+ Course objects and
    then deep-copying the whole thing to apply 24 overrides (see
    apply_overrides' docstring on why it deep-copies) reran on EVERY one of
    those, measured directly at ~0.9s combined. That's exactly why
    clicking "Remove" on a manually-added course felt like it had a
    per-click delay blocking the next click: the removal itself is a
    single list.pop(), but the rest of the script still had to pay that
    ~0.9s tax before the page could redraw.

    Deliberately st.cache_resource, not st.cache_data: cache_data returns a
    fresh deep copy of the cached value on every call (the right default
    for most cached data, to protect the cache from a caller mutating its
    result) -- for a 15k-object dict, that copy would cost almost as much
    as the original parse, defeating the point. cache_resource instead
    hands back the SAME dict object every time, which is safe here only
    because every downstream consumer (apply_overrides, add_placeholder_
    courses) already returns a new dict / deep-copies before mutating
    anything -- confirmed nothing in this file mutates a Course object in
    this dict in place.
    """
    base = load_snapshot(snapshot_path)
    return apply_overrides(base, load_overrides(overrides_path))


# ---------------------------------------------------------------------------
# 1. Course data snapshot
# ---------------------------------------------------------------------------
if not SNAPSHOT_PATH.exists():
    st.error(
        f"No course snapshot found at {SNAPSHOT_PATH}. Run "
        f"`python -m engine.course_data <year> {SNAPSHOT_PATH}` once to build one."
    )
    st.stop()

# Apply the CANONICAL, maintainer-curated corrections in data/course_
# overrides.json (see CourseOverride's docstring) -- e.g. ARCHLGY145's real
# biennial rotation. These are shared, checked-into-the-repo data fixes
# that apply to everyone; this line was previously missing entirely (the
# file existed and was used by every validation script, but never actually
# wired into the app itself), which silently ignored all 24 corrections.
#
# IMPORTANT: this file is NOT where a user's own in-app choices go (e.g.
# "drop this prerequisite, I have AP credit"). Those are personal to that
# person's session, not a fact about the course that should change what
# every other user of this app sees -- they live in st.session_state
# instead (see "2b. Your session-only corrections" below) and are applied
# on top of, not written into, this shared file. The same separation
# should hold for the natural-language override box mentioned in the
# README's "Stubbed" section when it's built: an LLM turning a user's
# sentence into a CourseOverride must apply it in-memory for that session,
# never call save_override() to persist it into this shared file.
#
# Both the raw snapshot parse and this overrides application are cached
# together in _load_base_courses() above (see its docstring) -- they used
# to run fresh on every single rerun of this script, which is most of why
# routine interactions (like removing a manually-added course in section
# 2d) felt like they had a built-in delay.
courses = _load_base_courses(str(SNAPSHOT_PATH), str(OVERRIDES_PATH))
st.sidebar.success(f"Course snapshot loaded: {len(courses)} courses across {len(DEFAULT_DEPARTMENTS)} departments")

# ---------------------------------------------------------------------------
# 1b. Session-only corrections (this user's own choices, never persisted to
# the shared data files -- see the big comment above). A list of
# CourseOverride, kept purely in Streamlit's per-session state so it
# resets when the session ends and never touches anyone else's app run.
# ---------------------------------------------------------------------------
if "session_overrides" not in st.session_state:
    st.session_state.session_overrides = []

# Only pay apply_overrides' deep-copy cost (see its docstring) when there's
# actually something to apply -- the common case (no "drop this
# prerequisite" escape hatch used yet) is the overwhelming majority of
# reruns, and apply_overrides(courses, []) would still deep-copy the whole
# ~15k-course dict for nothing every single time otherwise.
if st.session_state.session_overrides:
    courses = apply_overrides(courses, st.session_state.session_overrides)

if st.session_state.session_overrides:
    with st.sidebar.expander(f"Your session-only corrections ({len(st.session_state.session_overrides)})"):
        st.caption("Personal to this session only -- never saved to shared data, and gone if you restart the app.")
        for ov in st.session_state.session_overrides:
            st.text(f"{ov.course_key}.{ov.field} = {ov.value!r}\n  {ov.note}")
        if st.button("Clear my session corrections"):
            st.session_state.session_overrides = []
            st.rerun()

@st.cache_data(show_spinner="Parsing transcript...")
def _parse_transcript_cached(file_bytes: bytes):
    """Cached on the uploaded file's own bytes -- Streamlit keeps a
    transcript attached to the file_uploader across reruns until it's
    replaced or cleared, so without this, every single unrelated widget
    interaction (e.g. removing a manually-added course in section 2d) was
    re-running pdfplumber over the whole PDF again (~0.15s measured on a
    real transcript) for a result that hadn't changed. Keyed on the raw
    bytes rather than a filename/path, so re-uploading the exact same file
    still hits the cache, and a genuinely different file (different bytes)
    correctly reparses."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return parse_transcript_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. Transcript upload
# ---------------------------------------------------------------------------
st.header("1. Your starting point")
uploaded = st.file_uploader("Upload your unofficial Stanford transcript (PDF)", type=["pdf"])

completed_rows = []
ap_credits = []
treat_in_progress_as_done = True
ap_substituted_keys: set[str] = set()
if uploaded is not None:
    try:
        completed_rows, ap_credits = _parse_transcript_cached(uploaded.getvalue())
    except Exception as e:
        st.error(f"Couldn't parse this transcript: {e}")

    if completed_rows:
        st.success(f"Parsed {len(completed_rows)} course rows from your transcript.")
        with st.expander("Review parsed courses (fix anything that looks wrong before generating)"):
            for c in completed_rows:
                st.text(f"{c.key:10s}  {c.title[:55]:55s}  {c.units_attempted:5.2f}u  "
                         f"{'IN PROGRESS' if c.in_progress else c.grade:12s} [{c.term}]")
            st.caption(
                "This parser is a first pass -- if a row looks wrong, it's a parsing "
                "bug, not something wrong with your transcript. Manual correction "
                "here is a planned feature; for now, note it and we'll fix the parser."
            )

    _in_progress_rows = [c for c in completed_rows if c.in_progress]
    if _in_progress_rows:
        treat_in_progress_as_done = st.checkbox(
            f"Treat your {len(_in_progress_rows)} in-progress course(s) as already completed",
            value=True,
            help="On by default: in-progress courses count as 'spoken for' and satisfy any "
                 "requirement that references them. Turn this off if that term is just a "
                 "preliminary/preemptive registration before add/drop or before the term has "
                 "even started, and you want the solver free to reconsider those courses "
                 "(swap them for alternatives, drop them, etc.) rather than assuming they're "
                 "locked in. If you turn this off, set 'Starting catalog year'/'Starting "
                 "season' below to that same term so the solver re-plans it instead of "
                 "skipping straight past it.",
        )
        if not treat_in_progress_as_done:
            st.caption(
                "Not counted as completed: " + ", ".join(sorted(c.key for c in _in_progress_rows))
            )

    ap_substituted_keys: set[str] = set()
    if ap_credits:
        _ap_total = sum(a.units for a in ap_credits)
        _ap_parts = [
            f"{a.title.replace('Advanced Placement ', '')}: {a.units:.0f}u"
            for a in ap_credits
        ]
        st.info(
            f"Found {_ap_total:.0f} units of AP credit on your transcript "
            f"({', '.join(_ap_parts)}). Counted toward the informational 180-unit total "
            "below either way."
        )
        # AP credit CAN exempt you from specific course requirements, per
        # Stanford's own External Test Credit Equivalency Chart --
        # engine/ap_credit.py hand-transcribes the Chemistry/Calculus/
        # Physics/CS/Economics tables from that page (see its module
        # docstring for exactly what's covered, what's deliberately left
        # unmapped -- Language/Latin -- and how AND/OR "Courses Satisfied"
        # cells are read). Auto-applies where the transcript's own units
        # unambiguously pin down one chart row; asks you to pick when two
        # rows share the same unit award (Stanford's transcript can't tell
        # them apart either, e.g. AP Physics C Mechanics shows 4 units for
        # both a 4 and a 5).
        with st.expander("Does this AP credit exempt you from a specific course?", expanded=True):
            st.caption(
                "Pre-filled from Stanford's own equivalency chart (hand-transcribed into "
                "engine/ap_credit.py from an export of that page -- it isn't fetched live, so "
                "re-check against the current page if you suspect it's changed). Uncheck/edit "
                "anything that doesn't match your actual score."
            )
            for _i, a in enumerate(ap_credits):
                _candidates = guess_ap_equivalencies(a.title)
                _matches = [c for c in _candidates if abs(c.units - a.units) < 0.01]
                if len(_matches) == 1:
                    _row = _matches[0]
                    _keys = sorted(flatten_course_items(_row.satisfies))
                    _label = f"{a.title} -> {_row.exam_label}: satisfies {', '.join(_keys)}"
                    if _row.note:
                        _label += f"  ({_row.note})"
                    if st.checkbox(_label, value=True, key=f"ap_auto_{_i}"):
                        ap_substituted_keys.update(_keys)
                elif len(_matches) > 1:
                    st.caption(
                        f"**{a.title}** ({a.units:.0f}u) -- {len(_matches)} possible matches; "
                        "the transcript's units alone can't tell which score applies:"
                    )
                    _options = ["Not sure / none of these"] + [c.exam_label for c in _matches]
                    _choice = st.radio(
                        "Which is yours?", _options, key=f"ap_choice_{_i}", label_visibility="collapsed"
                    )
                    if _choice != "Not sure / none of these":
                        _row = next(c for c in _matches if c.exam_label == _choice)
                        _keys = sorted(flatten_course_items(_row.satisfies))
                        st.caption(f"-> satisfies {', '.join(_keys)}" + (f"  ({_row.note})" if _row.note else ""))
                        ap_substituted_keys.update(_keys)
                else:
                    st.caption(
                        f"**{a.title}** ({a.units:.0f}u) -- not in the built-in chart yet "
                        "(Language/Latin credit isn't auto-mapped -- see engine/ap_credit.py). "
                        "Enter it manually below if it substitutes for something in your plan."
                    )
                _manual = st.text_input(
                    f"Additional/override course code(s) for {a.title}",
                    value="",
                    key=f"ap_manual_{_i}",
                    help="Comma-separated. Use this if the built-in chart got it wrong for your "
                         "case, or for an exam it doesn't cover at all.",
                )
                for _code in _manual.split(","):
                    _code = _code.strip().upper().replace(" ", "")
                    if _code:
                        ap_substituted_keys.add(_code)
else:
    st.info("No transcript uploaded yet -- you can still explore the tool with zero completed courses.")

completed_keys = {
    c.key for c in completed_rows if not c.in_progress or treat_in_progress_as_done
}
completed_keys |= ap_substituted_keys

# ---------------------------------------------------------------------------
# 3. Program sheets -- two-step picker: pick a program by CODE, then pick
# which specific subplan/path/version of it applies to you. Deliberately no
# default selection anywhere in here (see list_variant_labels_by_code()'s
# docstring in engine/requirements.py for why a silent default -- the old
# picker pre-selected "CS-BS (core subset, v1, hand-encoded)", a tiny v1
# placeholder nowhere near the real major -- is actively dangerous: a real
# user's generated schedule quietly used that stub instead of the real
# CS-BS requirements).
# ---------------------------------------------------------------------------
st.header("2. Requirements to plan for")
_variants_by_code = list_variant_labels_by_code()
_selectable_codes = sorted(
    code for code, variants in _variants_by_code.items()
    if any(full_name not in BROKEN_PROGRAMS for full_name in variants.values())
)
selected_codes = st.multiselect(
    "Which program(s) are you working toward?",
    options=_selectable_codes,
    default=[],
    help="Pick by code (e.g. 'CS-BS', 'CHEM-BS'). You'll choose the specific subplan/path/"
         "version for each one right below -- nothing is pre-selected, since guessing the "
         "wrong one silently understates the real major.",
)

selected_sheets: list[str] = []
for _code in selected_codes:
    _variants = _variants_by_code[_code]
    _working = {label: full for label, full in _variants.items() if full not in BROKEN_PROGRAMS}
    _broken = {label: full for label, full in _variants.items() if full in BROKEN_PROGRAMS}
    _placeholder = "-- Select a version/subplan --"
    _choice = st.selectbox(
        f"{_code}: which version/subplan?",
        options=[_placeholder] + sorted(_working.keys()),
        index=0,
        key=f"variant_choice_{_code}",
    )
    if _choice != _placeholder:
        selected_sheets.append(_working[_choice])
    else:
        st.caption(f"{_code} won't be included in your plan until you pick a version above.")
    if _broken:
        with st.expander(f"{_code}: {len(_broken)} broken variant(s) -- not selectable"):
            for _label, _full in _broken.items():
                st.caption(f"{_label}: {BROKEN_PROGRAMS[_full]}")

placeholder_courses: dict[str, str] = {}
_grouped_rules = build_requirements_grouped(selected_sheets, out_placeholders=placeholder_courses)
if placeholder_courses:
    courses = add_placeholder_courses(courses, placeholder_courses)
    st.sidebar.warning(
        f"{len(placeholder_courses)} placeholder course(s) created for requirements whose "
        f"bulletin text named no specific course (see the expander below). They'll show up "
        f"in your schedule under their description -- swap in a real course once you know "
        f"what fits."
    )

# ---------------------------------------------------------------------------
# 2c. Stacking two undergrad majors (secondary major / double major / double
# degree) -- see engine/multi_program.py's module docstring for exactly what
# this does and doesn't cover (undergrad -BS/-BA pairs only; a major +
# unrelated minor, or any -MS/-MA combination, is unaffected and keeps
# working exactly as it always has: a plain union of rules, full
# double-counting, no mode to pick). Only appears when exactly 2 undergrad
# major sheets are selected above, since Stanford doesn't support triple-
# majoring and this feature has nothing to do when there's just one.
# ---------------------------------------------------------------------------
_selected_majors = [n for n in selected_sheets if is_undergrad_major_sheet(n)]
stack_mode = STACK_MODE_SECONDARY
graduation_unit_target = 180  # overwritten below if 2 majors are stacked in double_degree mode
restricted_course_programs: dict[str, list[str]] = {}

if len(_selected_majors) == 2:
    st.header("2c. Combining your two majors")
    _stack_modes_info = load_stack_modes()
    _name_a, _name_b = _selected_majors
    _degree_types = {degree_type_of(_name_a), degree_type_of(_name_b)}
    # Double degree isn't a preference, it's a definitional requirement --
    # Stanford's own policy is strictly one BA + one BS, never two BAs or
    # two BSs (see data/university_requirements.json's double_degree.
    # degree_type_requirement). So this isn't an "informational, proceed
    # anyway" warning like most soft checks in this project: if the two
    # selected majors aren't one of each, "double degree" is removed from
    # the option list entirely, the same way BROKEN_PROGRAMS are removed
    # from the program-sheet selector above rather than just flagged.
    _available_modes = [STACK_MODE_SECONDARY, STACK_MODE_DOUBLE_MAJOR]
    if _degree_types == {"BA", "BS"}:
        _available_modes.append(STACK_MODE_DOUBLE_DEGREE)
    else:
        _a_type, _b_type = degree_type_of(_name_a), degree_type_of(_name_b)
        st.caption(
            f"\"Double degree\" isn't offered below: it requires exactly one BA + one BS, "
            f"and your two selected majors are {_a_type} + {_b_type}."
        )
    stack_mode = st.selectbox(
        "How do these two majors combine?",
        options=_available_modes,
        format_func=lambda m: _stack_modes_info[m]["label"],
        index=0,
        help="Only the double-counting rule and the graduation unit total (180 vs. "
             "225) affect the schedule below -- diploma count/listing has no "
             "scheduling consequence and isn't modeled.",
    )
    _mode_info = _stack_modes_info[stack_mode]
    graduation_unit_target = _mode_info["total_units"]
    if not _mode_info["double_counting_allowed"]:
        _prog_a, _prog_b = sheet_code(_name_a), sheet_code(_name_b)
        rules_a, rules_b = _grouped_rules[_name_a], _grouped_rules[_name_b]
        restricted_course_programs = build_restricted_course_programs(
            _prog_a, rules_a, _prog_b, rules_b, courses
        )
        tag_program_id(rules_a, _prog_a)
        tag_program_id(rules_b, _prog_b)
        if restricted_course_programs:
            with st.expander(
                f"Courses that can't double-count between {_prog_a} and {_prog_b} "
                f"({len(restricted_course_programs)})"
            ):
                st.caption(
                    "These appear in an elective pool in BOTH majors' rule trees. Under "
                    f"{_mode_info['label'].split(' (', 1)[0].lower()}, the same enrollment can't "
                    "satisfy a counted slot in both at once -- you'll need a distinct course "
                    "for at least one of them. Shared intro-level math/language and "
                    "Engineering school-wide courses are exempted per Stanford's own policy "
                    "(best-effort list, not an official enumeration -- see engine/"
                    "multi_program.py)."
                )
                st.text(", ".join(sorted(restricted_course_programs)))
        else:
            st.caption(
                "No shared elective-pool courses found between these two majors -- the "
                "double-counting restriction has nothing to actually restrict here."
            )

requirement_rules: list = []
# Any requirement whose own real course pool is a genuinely LONG list of
# interchangeable electives (e.g. CS-BS's ~100-course General/Track
# Electives pools) gets rewritten into a generic placeholder here, same
# reasoning as WAYS/COLLEGE/Writing/Language's own placeholders below --
# see placeholderize_large_elective_pools's docstring. Real, user-reported
# complaint this fixes: the solver was committing to one arbitrary
# specific real course (e.g. PSYCH204A) out of dozens of equally-valid
# alternatives for a slot like this, which reads as a firm pick rather
# than the open choice it actually is. A small pool (e.g. CS-BS's 3-option
# Capstone/WiM choice) is left with its real, named courses -- see this
# function's own docstring for the size cutoff.
for _name in selected_sheets:
    _sheet_rules, _sheet_elective_placeholders = placeholderize_large_elective_pools(
        _grouped_rules.get(_name, [])
    )
    if _sheet_elective_placeholders:
        courses = add_placeholder_courses(courses, _sheet_elective_placeholders)
        placeholder_courses.update(_sheet_elective_placeholders)
    requirement_rules.extend(_sheet_rules)

def _flat_course_keys(item) -> set[str]:
    if isinstance(item, tuple) and item and item[0] == "__and__":
        return {c for c in item[1:] if isinstance(c, str)}
    if isinstance(item, (tuple, list)):
        return {c for c in item if isinstance(c, str)}
    return {item} if isinstance(item, str) else set()


def _rule_lines(r, indent=0, completed_keys: set[str] | None = None) -> list[str]:
    """Render a requirement rule and its nested sub_rules (if any) --
    a parsed program sheet's rules nest (e.g. "Mathematics" wraps
    "Calculus", "Math Electives", etc. as sub_rules rather than a flat
    courses list), so only printing r.courses would silently show an
    empty list for those.

    When `completed_keys` is passed, also flags which of this rule's own
    course pool are already in it -- this is the ONLY place completion
    status shows up in this listing (it's otherwise just the static rule
    tree), so a pool that's already satisfied by your transcript won't
    look any different from one that still needs courses unless you check
    here. This doesn't attempt to duplicate solver.py's actual
    satisfaction logic (nested sub_rules, AND-groups within an at_least_n,
    etc.) -- it's a simple "which of these course keys do you already
    have" flag per rule, not a solver-equivalent OPTIMAL/blocked verdict."""
    pad = "  " * indent
    line = f"{pad}[{r.kind}] {r.label}"
    if r.n:
        line += f"  (need {r.n})"
    if r.unit_min:
        line += f"  (need {r.unit_min} units)"
    if r.courses:
        line += f": {r.courses}"
    elif r.sub_rules:
        # A "wrapper" rule has no courses of its own -- its sub_rules ARE
        # the options, not just further nested detail -- so spell out the
        # pick-N-of-these-M relationship instead of leaving it to be
        # inferred from indentation and "(need N)" alone. This matters
        # most for a Track-B/Track-C-shaped structure (e.g. CS-BS's AI
        # subplan Track B: "pick 2 of these 4 areas, each area itself
        # needs just 1 course" -- 2 courses total, not 4): without this,
        # "[at_least_n] Track B Requirement (need 2)" followed by 4
        # equally-indented child rules reads as if one course were needed
        # from EACH of the 4, not 2 of the 4 overall.
        if r.kind == "at_least_n":
            line += f"  -- pick {r.n} of the {len(r.sub_rules)} options below, each satisfied on its own terms"
        elif r.kind == "all":
            line += f"  -- complete ALL {len(r.sub_rules)} of the items below"
    if completed_keys and r.courses:
        flat: set[str] = set()
        for item in r.courses:
            flat |= _flat_course_keys(item)
        matched = sorted(flat & completed_keys)
        if matched:
            line += f"   <- ALREADY HAVE: {', '.join(matched)}"
    lines = [line]
    for sr in r.sub_rules:
        lines.extend(_rule_lines(sr, indent + 1, completed_keys))
    return lines


with st.expander(f"Requirement rules being applied ({len(requirement_rules)})"):
    st.caption(
        "\"<- ALREADY HAVE\" flags which of a rule's own course pool your transcript "
        "already covers -- it's not a full satisfied/not-satisfied verdict for the whole "
        "rule (that's what the generated schedule's status actually proves)."
    )
    for r in requirement_rules:
        for line in _rule_lines(r, completed_keys=completed_keys):
            st.text(line)

# ---------------------------------------------------------------------------
# 2d. Your own course list (optional) -- for someone who already knows
# exactly which courses satisfy their requirements and just wants this
# app's CP-SAT sequencing (prerequisites, corequisites, time conflicts,
# unit caps, weight preferences), not its program-sheet requirement
# parsing. Can be used with NO program sheet selected above (a fully
# custom plan) or layered on top of one or more selected sheets (an
# ADDITIONAL hard-required set, on top of whatever they already require)
# -- both are supported in the same session; which one you get is purely a
# function of whether you also picked anything in section 2 above.
#
# Deliberately hard-required (kind="all"), same as any other requirement
# in this app -- if a course you list here can't actually fit (a
# prerequisite that can't land in time, a real time conflict, etc.), the
# solve reports INFEASIBLE with a reason, same as any other hard
# requirement, rather than silently dropping something you specifically
# asked for.
#
# "2b. University-wide requirements" below stays fully available while
# this is active -- it's not mutually exclusive with a custom list. Leave
# it OFF for a fully self-contained plan where you're personally vouching
# that your list already covers everything, or leave it ON to have the
# solver "fill in" whatever WAYS/COLLEGE/Writing/Language courses your own
# list doesn't already happen to satisfy, same as it would on top of any
# program sheet.
# ---------------------------------------------------------------------------
st.header("2d. Your own course list (optional)")
use_manual_courses = st.checkbox(
    "Add specific courses yourself, on top of (or instead of) the above",
    value=False,
    help="Turn this on if you already know exactly which courses satisfy your "
         "requirements and just want them sequenced -- skip program sheets "
         "entirely for a fully custom plan, or select some in section 2 above "
         "AND add more here. Every course you add here is hard-required, same "
         "as a real degree requirement.",
)
if use_manual_courses:
    st.caption(
        "\"2b. University-wide requirements\" below still applies if you leave it "
        "checked -- it'll \"fill in\" with additional courses for whatever WAYS/"
        "COLLEGE/Writing/Language requirements your own list doesn't already cover. "
        "Uncheck it there if your list is meant to be fully self-contained."
    )
    if "manual_course_keys" not in st.session_state:
        st.session_state.manual_course_keys = []
    if "manual_course_gen" not in st.session_state:
        st.session_state.manual_course_gen = 0

    _mgen = st.session_state.manual_course_gen
    _code_input = st.text_input(
        "Enter the exact course code and press Enter (e.g. \"chem131\")",
        key=f"manual_course_code_{_mgen}",
        help="Case-insensitive, spaces ignored -- 'chem 131' and 'CHEM131' both work. "
             "This is a single request, not the two-step search-then-pick before: type "
             "the code you already know and hit Enter.",
    )
    if _code_input.strip():
        _code = _code_input.strip().upper().replace(" ", "")
        if _code in courses and not is_placeholder_key(_code):
            if _code not in st.session_state.manual_course_keys:
                st.session_state.manual_course_keys.append(_code)
            st.session_state.manual_course_gen += 1
            st.rerun()
        else:
            st.error(f"{_code!r} isn't a course in the current snapshot -- check the code and try again.")
            # "Did you mean" suggestions for a likely typo -- difflib is
            # stdlib (no new dependency) and, benchmarked directly against
            # the real ~15,000-course snapshot, resolves in ~15-20ms per
            # lookup (e.g. "CHEM31C" -> CHEM31S/E/B/A), imperceptible for a
            # one-shot action that only runs on this rare not-found path
            # (never per-keystroke) -- confirmed fast enough to be worth
            # keeping rather than skipped as "clunky/laggy."  cutoff=0.6
            # means nothing shows at all for a query too dissimilar from
            # any real code to be a plausible typo, rather than forcing a
            # bad guess.
            _real_keys = [k for k in courses if not is_placeholder_key(k)]
            _suggestions = difflib.get_close_matches(_code, _real_keys, n=5, cutoff=0.6)
            if _suggestions:
                st.caption("Did you mean:")
                _cols = st.columns(len(_suggestions))
                for _col, _sugg in zip(_cols, _suggestions):
                    with _col:
                        if st.button(_sugg, key=f"manual_course_suggest_{_mgen}_{_sugg}", help=courses[_sugg].title):
                            if _sugg not in st.session_state.manual_course_keys:
                                st.session_state.manual_course_keys.append(_sugg)
                            st.session_state.manual_course_gen += 1
                            st.rerun()

    st.markdown("**Or upload a list:** a `.txt` file with one course code per line.")
    _uploaded_course_list = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
        key=f"manual_course_upload_{_mgen}",
        help="One course code per line -- case-insensitive, spaces ignored, "
             "same normalization as the box above (e.g. 'chem 131' and "
             "'CHEM131' both work). Blank lines are skipped. Lines that "
             "aren't a real course code in the current snapshot are "
             "reported below, not silently dropped.",
    )
    if _uploaded_course_list is not None:
        _file_text = _uploaded_course_list.getvalue().decode("utf-8", errors="replace")
        _file_added: list[str] = []
        _file_not_found: list[str] = []
        for _raw_line in _file_text.splitlines():
            _line_code = _raw_line.strip().upper().replace(" ", "")
            if not _line_code:
                continue  # skip blank lines
            if _line_code in courses and not is_placeholder_key(_line_code):
                if _line_code not in st.session_state.manual_course_keys:
                    st.session_state.manual_course_keys.append(_line_code)
                    _file_added.append(_line_code)
            else:
                _file_not_found.append(_line_code)
        if _file_added:
            st.success(f"Added {len(_file_added)} course(s) from the file: {', '.join(_file_added)}")
        if _file_not_found:
            st.error(
                f"{len(_file_not_found)} line(s) from the file weren't found in the "
                f"current snapshot -- check for typos: {', '.join(_file_not_found)}"
            )
            _real_keys = [k for k in courses if not is_placeholder_key(k)]
            for _bad in _file_not_found:
                _suggestions = difflib.get_close_matches(_bad, _real_keys, n=3, cutoff=0.6)
                if _suggestions:
                    st.caption(f"Did you mean, for {_bad!r}: {', '.join(_suggestions)}")
        if not _file_added and not _file_not_found:
            st.caption("That file didn't have any non-blank lines to read.")
        if _file_added and not _file_not_found:
            # Clean run (everything in the file resolved) -- bump the
            # generation counter so the uploader resets for a fresh upload
            # next time, same reset pattern as the single-code text box
            # above. Left attached (no rerun) whenever there's anything to
            # review, so the success/error/suggestion messages above stay
            # visible instead of vanishing the instant they'd be useful.
            st.session_state.manual_course_gen += 1
            st.rerun()

    manual_course_keys: list[str] = list(st.session_state.manual_course_keys)
    if manual_course_keys:
        st.subheader(f"Your course list ({len(manual_course_keys)})")
        for _i, _key in enumerate(manual_course_keys):
            _c = courses.get(_key)
            _title = _c.title if _c else "(not in current snapshot)"
            _cols = st.columns([6, 1])
            _cols[0].write(f"**{_key}** -- {_title}")
            if _cols[1].button("Remove", key=f"manual_course_remove_{_i}"):
                st.session_state.manual_course_keys.pop(_i)
                st.rerun()
        requirement_rules.append(RequirementRule(
            label=f"Your own course list -- {len(manual_course_keys)} course(s), all required",
            kind="all",
            courses=manual_course_keys,
            source="Manually added by you (not derived from any program sheet or university requirement)",
        ))
    else:
        st.caption("No courses added yet -- type a code above or upload a .txt file.")

# ---------------------------------------------------------------------------
# 3b. University-wide requirements (WAYS, COLLEGE, Writing, Language) -- see
# engine/university_requirements.py's module docstring for exactly what's
# covered and what's deliberately left out (units/GPA/residency/pass-fail,
# double-major/coterm interactions). Applies ON TOP of whatever program
# sheet(s) were selected above, same rule-list contract.
# ---------------------------------------------------------------------------
st.header("2b. University-wide requirements")
# Freely toggleable regardless of "2d. Your own course list" -- the two
# are additive, not mutually exclusive. With a manual list active, leaving
# this checked means the solver "fills in" whatever WAYS/COLLEGE/Writing/
# Language courses your own list doesn't already happen to satisfy (a
# course you added yourself that happens to carry a WAYS tag still counts,
# same double-counting-aware credit as anywhere else in this app); leaving
# it unchecked means your list is treated as fully self-contained and
# university requirements aren't checked at all. Previously this was
# force-disabled whenever "2d" was active, on the assumption that a custom
# list meant taking on full responsibility for every requirement yourself
# -- changed after user feedback that a custom list is often built to
# cover major-specific requirements only, with university-wide ones still
# meant to be auto-filled same as usual.
include_university_reqs = st.checkbox(
    "Include Stanford's university-wide requirements (WAYS, COLLEGE, Writing, Language)",
    value=True,
    help=(
        "On top of whatever program sheet(s) and/or your own course list (section 2d) "
        "you're using. Units/GPA/residency/pass-fail aren't modeled as solver "
        "constraints -- see the informational 180-unit check after generating a "
        "schedule instead."
        + (
            " With \"2d. Your own course list\" active, this \"fills in\" extra "
            "courses for whatever WAYS/COLLEGE/Writing/Language requirements your "
            "list doesn't already cover, rather than requiring your list to cover "
            "everything itself."
            if use_manual_courses else ""
        )
    ),
)
university_rules: list = []
if include_university_reqs:
    ucol1, ucol2 = st.columns(2)
    with ucol1:
        admit_fall_year = st.number_input(
            "Admit year (fall you started/will start)",
            min_value=2020, max_value=2035, value=2026,
            help="Selects the 2-quarter (through 2026-2027) vs. 3-quarter (Fall 2027+, "
                 "best-guess) COLLEGE policy.",
        )
    with ucol2:
        language_quarters_needed = st.number_input(
            "Quarters of language requirement still needed",
            min_value=0, max_value=6, value=0,
            help="Not tied to real course data -- depends on your placement test/AP "
                 "credit, which this tool has no data source for. Enter how many "
                 "quarters you still owe; it's modeled as a generic placeholder "
                 "course repeated that many times. Replace with your real language "
                 "course(s) before trusting the schedule.",
        )
    # Every course already required by a program sheet or your own manual
    # list (section 2d), so WAYS/COLLEGE/Writing can credit one of THOSE
    # too, not just courses already on your transcript -- a real bug
    # otherwise: a manually-added CHEM131 (real WAYS tags AQR + SMA) was
    # still forcing 11 separate, generic WAYS placeholder quarters on top,
    # since the calculation had no way to know CHEM131 was already
    # happening regardless. See models.all_referenced_course_keys() and
    # build_all_university_rules()'s own docstring for the full story.
    _other_required_keys = all_referenced_course_keys(requirement_rules)
    university_rules, university_placeholders = build_all_university_rules(
        courses,
        admit_fall_year=int(admit_fall_year),
        language_quarters_needed=int(language_quarters_needed),
        completed_keys=completed_keys,
        other_required_keys=_other_required_keys,
    )
    if university_placeholders:
        courses = add_placeholder_courses(courses, university_placeholders)
        placeholder_courses.update(university_placeholders)
    requirement_rules = requirement_rules + university_rules
    with st.expander(f"University-wide rules being applied ({len(university_rules)})"):
        st.caption(
            "\"<- ALREADY HAVE\" flags which of a rule's own course pool your transcript "
            "already covers (e.g. a WAYS category you've already got a tagged course for)."
        )
        for r in university_rules:
            for line in _rule_lines(r, completed_keys=completed_keys):
                st.text(line)
else:
    admit_fall_year = None
    language_quarters_needed = 0

if placeholder_courses:
    with st.expander(f"Placeholder courses in use ({len(placeholder_courses)}) -- need manual review", expanded=True):
        st.caption(
            "These aren't real Stanford courses -- they stand in for a requirement whose "
            "bulletin text named no specific course (free text, or an unenumerated pool like "
            "'any MATH course numbered 100 and above'). They'll be scheduled like a normal "
            "course so the plan stays complete, but you should replace each one with a real "
            "course that fits before treating the schedule as final. Units are usually a guess "
            "(shown as a 3-5 range below and in the schedule; the solver uses a fixed 4 "
            "internally for unit caps and load-balancing) -- a few (like the language "
            "requirement) have real, non-guessed units instead, shown as a single number."
        )
        for key, value in placeholder_courses.items():
            # Most placeholders are just a description string here (see
            # course_data.add_placeholder_courses()'s docstring); a few
            # (the language requirement) are the actual Course object,
            # since their units are real rather than the generic 3-5/4
            # guess -- pull out the matching description/units for each.
            if isinstance(value, str):
                desc, units_note = value, "3-5u, ~4 used internally"
            else:
                desc = value.description
                if value.units_min == value.units_max:
                    units_note = f"{value.units_min:.0f}u"
                else:
                    units_note = f"{value.units_min:.0f}-{value.units_max:.0f}u"
            st.text(f"  {key} ({units_note})\n    -> {desc}")

# ---------------------------------------------------------------------------
# 4. Weight table -- an add-one-at-a-time form instead of a spreadsheet-
# style editor with a freeform JSON "args" cell. The JSON-cell version was
# fragile in practice (a user typing "morning" into that cell, expecting it
# to just be the value, got a bare "couldn't parse args" error with no
# indication of the JSON-object shape it actually needed) -- every arg now
# gets its own properly-typed widget (a dropdown for an enum-valued arg
# like day/phase, a number input for a numeric one, ...), driven by
# WEIGHT_TYPE_SCHEMA's structured arg specs so adding a new weight type
# later doesn't need any new app.py code as long as its args fit the
# existing "kind"s. No rows are pre-seeded, unlike the old default two-row
# starting table -- an empty weight table is a completely valid choice
# (the solver just satisfies requirements with no scheduling preference).
# ---------------------------------------------------------------------------
st.header("3. Weight table (tweak these)")
st.caption(
    "Optional: nudge the solver's preferences among otherwise-equally-valid schedules. "
    "Add as many or as few as you want -- weight magnitudes well above 1 are recommended, "
    "since the solver has a small built-in bias against scheduling unnecessary courses that "
    "a tiny weight can get lost against."
)
with st.expander("Available weight types"):
    for wtype, info in WEIGHT_TYPE_SCHEMA.items():
        st.markdown(f"**{wtype}** -- {info['description']}")

if "weight_rows" not in st.session_state:
    st.session_state.weight_rows = []
if "weight_form_gen" not in st.session_state:
    st.session_state.weight_form_gen = 0

# Deliberately NOT wrapped in st.form: a form batches every widget inside
# it and only reruns the script when its submit button fires, so changing
# "Type" wouldn't immediately swap in the matching description/arg widgets
# below it -- confirmed as the actual cause of a real "clunky, the
# description doesn't match what I picked" complaint (a plain-widget
# rerun-on-every-change is exactly what's needed here, and the tradeoff --
# every keystroke in a text-based arg reruns the app -- doesn't apply to
# the enum/number widgets this schema actually uses). `_gen` is bumped
# after a successful add so every widget below (keyed on it) resets to a
# fresh default instead of holding onto whatever was just submitted --
# the same "clear the form after submit" behavior st.form's
# clear_on_submit gave up by not using a form.
_gen = st.session_state.weight_form_gen
st.subheader("Add a weight rule")
_new_type = st.selectbox("Type", options=list(WEIGHT_TYPE_SCHEMA.keys()), key=f"new_weight_type_{_gen}")
_type_info = WEIGHT_TYPE_SCHEMA[_new_type]
st.caption(_type_info["description"])
_new_args: dict = {}
for _arg_name, _arg_spec in _type_info["args"].items():
    _widget_key = f"new_weight_arg_{_gen}_{_new_type}_{_arg_name}"
    _kind = _arg_spec["kind"]
    if _kind == "enum":
        _new_args[_arg_name] = st.selectbox(
            _arg_name, options=_arg_spec["options"], help=_arg_spec.get("help", ""), key=_widget_key
        )
    elif _kind == "multi_enum":
        _new_args[_arg_name] = st.multiselect(
            _arg_name, options=_arg_spec["options"], help=_arg_spec.get("help", ""), key=_widget_key
        )
    elif _kind == "number":
        _new_args[_arg_name] = st.number_input(
            _arg_name,
            min_value=_arg_spec.get("min", 0),
            max_value=_arg_spec.get("max", 999),
            value=_arg_spec.get("default", 0),
            help=_arg_spec.get("help", ""),
            key=_widget_key,
        )
    elif _kind == "course_key":
        _new_args[_arg_name] = st.text_input(
            _arg_name, help=_arg_spec.get("help", ""), key=_widget_key
        ).strip().upper().replace(" ", "")
    elif _kind == "course_key_list":
        _raw_list = st.text_input(
            _arg_name, help=_arg_spec.get("help", ""), key=_widget_key
        )
        _new_args[_arg_name] = [
            c.strip().upper().replace(" ", "") for c in re.split(r"[,\n]+", _raw_list) if c.strip()
        ]
    elif _kind == "subject":
        _new_args[_arg_name] = st.text_input(
            _arg_name, help=_arg_spec.get("help", ""), key=_widget_key
        ).strip().upper()
    else:
        _new_args[_arg_name] = st.text_input(_arg_name, key=_widget_key)
_new_weight = st.number_input("Weight", value=3.0, step=1.0, key=f"new_weight_value_{_gen}")
_new_label = st.text_input("Label (optional, just for your own reference)", key=f"new_weight_label_{_gen}")
if st.button("Add weight rule", key=f"add_weight_btn_{_gen}"):
    _missing = [k for k, v in _new_args.items() if v in ("", [], None)]
    if _missing:
        st.warning(f"Fill in {', '.join(_missing)} before adding this rule.")
    else:
        st.session_state.weight_rows.append(
            {"type": _new_type, "args": _new_args, "weight": _new_weight, "label": _new_label}
        )
        st.session_state.weight_form_gen += 1
        st.rerun()

if not st.session_state.weight_rows:
    st.caption("No weight rules yet -- the solver will satisfy requirements with no scheduling preference.")
else:
    st.subheader("Current weight rules")
    for _i, _row in enumerate(st.session_state.weight_rows):
        _cols = st.columns([6, 1])
        _args_str = ", ".join(f"{k}={v}" for k, v in _row["args"].items())
        _label_part = f"  -- {_row['label']}" if _row.get("label") else ""
        _cols[0].write(f"**{_row['type']}**({_args_str}) weight={_row['weight']:g}{_label_part}")
        if _cols[1].button("Remove", key=f"remove_weight_{_i}"):
            st.session_state.weight_rows.pop(_i)
            st.rerun()

weight_rules: list[WeightRule] = [
    WeightRule(type=row["type"], args=row["args"], weight=float(row["weight"]), label=row.get("label", ""))
    for row in st.session_state.weight_rows
]

# ---------------------------------------------------------------------------
# 3b. Small changes (natural language) -- the "natural-language override
# box" the README's "Stubbed" section flagged as a natural next step. Each
# request is parsed IMMEDIATELY (so the log below fills in right away),
# but parsing alone never re-solves anything -- it only appends to
# st.session_state.session_overrides / .weight_rows, the same two lists
# the rest of this app already reads from once "Generate schedule" (below)
# is actually clicked. That button press is the only thing that ever
# triggers a solve, so any number of requests can queue up here first,
# exactly like editing the weight table above doesn't auto-generate either.
# ---------------------------------------------------------------------------
st.header("3b. Small changes (describe them in plain English)")
st.caption(
    "Examples: \"CS107 is also offered in Spring\", \"waive the CHEM101 prerequisite for "
    "me, I have AP credit\", \"I'd rather have morning classes\". Each request is parsed "
    "right away and logged below, but nothing re-solves until you click \"Generate "
    "schedule\" -- queue up as many as you want first. Requires an ANTHROPIC_API_KEY "
    "set in the environment this app was launched from."
)

if "nl_change_log" not in st.session_state:
    st.session_state.nl_change_log = []
if "nl_change_gen" not in st.session_state:
    st.session_state.nl_change_gen = 0

_nl_gen = st.session_state.nl_change_gen
_nl_sentence = st.text_input("Describe a change", key=f"nl_change_input_{_nl_gen}")
if st.button("Add change", key=f"nl_change_btn_{_nl_gen}"):
    if not _nl_sentence.strip():
        st.warning("Type something first.")
    else:
        with st.spinner("Interpreting..."):
            _result = parse_change_request(_nl_sentence, known_course_keys=set(courses.keys()))
        st.session_state.nl_change_log.append({
            "sentence": _nl_sentence,
            "change_type": _result.change_type,
            "summary": _result.summary,
            "error": _result.error,
        })
        if _result.change_type == "course_override" and _result.course_override is not None:
            st.session_state.session_overrides.append(_result.course_override)
        elif _result.change_type == "weight_rule" and _result.weight_rule is not None:
            wr = _result.weight_rule
            st.session_state.weight_rows.append(
                {"type": wr.type, "args": wr.args, "weight": wr.weight, "label": wr.label}
            )
        st.session_state.nl_change_gen += 1
        st.rerun()

if st.session_state.nl_change_log:
    st.subheader("Change log")
    for _entry in reversed(st.session_state.nl_change_log):
        if _entry["change_type"] == "course_override":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (course correction, this session only)")
        elif _entry["change_type"] == "weight_rule":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (added as a weight rule above)")
        else:
            st.warning(f"**\"{_entry['sentence']}\"** -- not applied: {_entry['error'] or 'unrecognized request'}")

# ---------------------------------------------------------------------------
# 5. Planning horizon + generate
# ---------------------------------------------------------------------------
st.header("4. Generate")
col1, col2, col3, col4 = st.columns(4)
with col1:
    start_year = st.text_input("Starting catalog year (next quarter to plan)", value="2026-2027")
with col2:
    num_quarters = st.number_input("Quarters to plan ahead", min_value=1, max_value=16, value=8)
with col3:
    unit_cap = st.number_input("Max units per quarter", min_value=1, max_value=30, value=18)
with col4:
    years_completed_at_start = st.number_input(
        "Full years already completed",
        min_value=0, max_value=6, value=0,
        help="How many full academic years you've already finished as of the quarter "
             "above -- 0 if you're starting fresh as a freshman, 1 if this plan starts "
             "sophomore year, etc. Only affects a small, hand-curated list of courses "
             "known to require a specific class standing (e.g. CHEM176 is junior/senior "
             "only; CS-BS's Capstone/WiM options -- CS191W/CS194W/CS210B -- are senior "
             "only) -- see engine/course_data.py's KNOWN_CLASS_YEAR_MIN.",
    )

# Default rotation skips Summer -- most students aren't enrolled then, and
# including it by default would let the solver quietly schedule required
# courses in a quarter the student doesn't actually plan to use. Extending
# this to optionally include Summer is easy later (make it a checkbox that
# swaps which tuple this rotation cycles through).
SEASON_ROTATION = ("Autumn", "Winter", "Spring")


def build_quarter_list(start_year: str, start_season: str, n: int) -> list[str]:
    y0 = int(start_year.split("-")[0])
    quarters = []
    si = SEASON_ROTATION.index(start_season)
    year = y0
    for _ in range(n):
        season = SEASON_ROTATION[si]
        quarters.append(f"{year}-{year+1} {season}")
        si += 1
        if si >= len(SEASON_ROTATION):
            si = 0
            year += 1
    return quarters


start_season = st.selectbox("Starting season", SEASON_ROTATION, index=1)
quarter_list = build_quarter_list(start_year, start_season, int(num_quarters))
st.caption("Planning horizon: " + ", ".join(quarter_list))

if st.button("Generate schedule", type="primary"):
    # Manually-added courses (section 2d) skip their OWN prerequisite
    # enforcement -- someone typing in "the entire CHEM-BS" already has a
    # specific set of quarters in mind for it, often on top of years
    # already completed (e.g. "I've finished freshman year, just plan the
    # next three"); auto-inserting an earlier course they didn't ask for
    # just to "prove" a prerequisite would work against that, not for it.
    # Only skips prereqs FOR these courses themselves -- one of them can
    # still be enforced as someone else's prerequisite as normal. Read from
    # session_state directly (not the local `manual_course_keys` variable
    # from section 2d, which only exists in that branch's scope) so this is
    # correct whether or not "2d" is the active section on this run.
    _manual_keys_for_prereq_skip = (
        set(st.session_state.get("manual_course_keys", [])) if use_manual_courses else set()
    )
    # Hard-forbid known class-year-restricted courses (CHEM176, CS-BS's
    # Capstone/WiM options, ...) from landing in a quarter before the
    # student would actually have that class standing -- see "Full years
    # already completed" above and engine/course_data.py's
    # KNOWN_CLASS_YEAR_MIN for the underlying list.
    _class_year_exclusions = class_year_quarter_exclusions(
        quarter_list, years_completed_at_horizon_start=int(years_completed_at_start)
    )
    with st.spinner("Solving..."):
        result = build_and_solve(
            courses=courses,
            completed_keys=completed_keys,
            requirement_rules=requirement_rules,
            weight_rules=weight_rules,
            quarter_list=quarter_list,
            unit_cap_per_quarter=float(unit_cap),
            time_limit_s=30.0,
            restricted_course_programs=restricted_course_programs,
            assume_prereqs_satisfied_for=_manual_keys_for_prereq_skip,
            course_quarter_exclusions=_class_year_exclusions,
        )

    if result.status not in ("OPTIMAL", "FEASIBLE"):
        if result.blocking_prerequisites:
            # A hard-enforced prerequisite (see engine/solver.py) is
            # specifically what's blocking this plan, not a general
            # requirement conflict -- CP-SAT identified exactly which
            # pair(s) via its own assumption-based infeasibility
            # diagnosis, so offer a concrete, one-click way forward
            # instead of a dead end: drop it and try again. This is a
            # PERSONAL exception (e.g. AP credit, a waiver, a transfer
            # course), not a fact about the course itself -- it's recorded
            # in st.session_state, applied only to this browser session,
            # and never written to the shared data/course_overrides.json
            # file that every user of this app loads (see the big comment
            # near where `courses` is first built).
            st.error(
                f"Solver status: {result.status} -- blocked by the following "
                f"prerequisite requirement(s), not a general scheduling "
                f"conflict. Drop one below to let the solver proceed for "
                f"THIS session only -- you're telling the tool this specific "
                f"prerequisite doesn't apply to you personally, e.g. because "
                f"of AP credit, a waiver, or a transfer course; it won't "
                f"change what anyone else sees."
            )
            for prereq_key, course_key in result.blocking_prerequisites:
                # prereq_key is usually a real course key, but for an
                # OR-group prerequisite (e.g. CS107E's real "CS106B or
                # CS106X") it's a " or "-joined display string instead --
                # build a matching " or "-joined title rather than a raw
                # dict lookup that would just fail and fall back to the key.
                if " or " in prereq_key:
                    prereq_title = " or ".join(
                        courses[p].title if p in courses else p
                        for p in prereq_key.split(" or ")
                    )
                else:
                    prereq_title = courses[prereq_key].title if prereq_key in courses else prereq_key
                course_title = courses[course_key].title if course_key in courses else course_key
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.text(f"{course_key} ({course_title}) requires {prereq_key} ({prereq_title}) first")
                with col_b:
                    if st.button("Drop for me", key=f"drop_prereq_{prereq_key}_{course_key}"):
                        st.session_state.session_overrides.append(
                            CourseOverride(
                                course_key=course_key,
                                field="prerequisite_courses",
                                value=[],
                                note=(
                                    f"Dropped via app (this session only): "
                                    f"blocked the schedule because {prereq_key} "
                                    f"couldn't be fit before {course_key} in "
                                    f"the chosen horizon."
                                ),
                            )
                        )
                        st.rerun()
        else:
            st.error(f"Solver status: {result.status} -- no valid schedule found. "
                     f"Check for unsatisfiable requirements or a too-short horizon.")
    else:
        badge = "proven optimal" if result.status == "OPTIMAL" else "best found in time limit (not proven optimal)"
        st.success(f"Schedule generated -- {badge}. Objective score: {result.objective_value:.1f} "
                   f"(solved in {result.solve_time_s:.2f}s)")

        if result.unplaced_required:
            # University-wide rules (WAYS/writing) draw from pools of hundreds
            # of alternative course sections; most "unplaced" entries there
            # are just pool members with no offering inside the chosen
            # horizon (e.g. Summer-only sections when Summer isn't in the
            # rotation) -- benign, since the rule itself is still satisfied
            # by other pool members whenever the overall status is OPTIMAL/
            # FEASIBLE. Truncate the display rather than dumping a list that
            # can run into the thousands for these pool-shaped rules.
            _unplaced_shown = result.unplaced_required[:30]
            _unplaced_more = len(result.unplaced_required) - len(_unplaced_shown)
            st.warning(
                f"{len(result.unplaced_required)} required course(s)/alternative(s) have NO "
                "valid quarter in this snapshot/horizon. For a specific single required "
                "course this usually means it's missing from the fetched departments or "
                "needs a manual override; for a large pool-based rule (WAYS/Writing) this "
                "commonly just means that particular alternative isn't offered in your "
                "chosen quarters -- the rule as a whole is still satisfied by other pool "
                "members since the schedule above is " + result.status + ". "
                "Examples: " + ", ".join(_unplaced_shown)
                + (f" ... and {_unplaced_more} more" if _unplaced_more > 0 else "")
            )

        if result.prerequisite_warnings:
            # CS department doesn't actually enforce its own prerequisites
            # in Axess -- see engine/solver.py's build_and_solve (section
            # 2b/4c) for the full reasoning -- so these were allowed into
            # the schedule as-is rather than blocking it or forcing a
            # different order. Purely informational: nothing to fix here
            # unless the student actually cares about taking them in the
            # "normal" order.
            st.info(
                f"{len(result.prerequisite_warnings)} CS course(s) landed out of their usual "
                "prerequisite order. The CS department doesn't actually enforce prerequisites "
                "at enrollment, so this wasn't treated as a blocker -- just flagging it:\n\n"
                + "\n".join(
                    f"- {course_key} ({courses[course_key].title if course_key in courses else '?'}) "
                    f"is in {quarter_label}, before its usual prerequisite {prereq_display}"
                    for prereq_display, course_key, quarter_label in result.prerequisite_warnings
                )
            )

        # graduation_unit_target defaults to 180 (single degree) and is only
        # overridden when 2 undergrad majors are stacked in "double degree"
        # mode above (225, per university_requirements.json's double_degree.
        # total_units_required) -- secondary major and double major are both
        # still 180. Always informational, never a solver constraint (see
        # engine/university_requirements.py's module docstring for why).
        single_degree_min = graduation_unit_target

        # A repeat-eligible course (e.g. the Language requirement placeholder,
        # scheduled across 2-3 quarters to satisfy an at_least_n(n>1) rule)
        # shows up once in result.assignment but ALL its quarters in
        # result.repeat_quarters -- iterate occurrences (one per scheduled
        # quarter, not one per course key) so units and the by-quarter table
        # below both count/show every enrollment, not just the first.
        def _scheduled_occurrences():
            for key in result.assignment:
                for q in result.repeat_quarters.get(key, [result.assignment[key]]):
                    yield key, q

        # Units are only actually EARNED once a course finishes with a
        # grade -- an in-progress course has attempted units on the
        # transcript but no posted grade yet, so it hasn't earned anything
        # regardless of whether treat_in_progress_as_done is letting it
        # satisfy a requirement pool above. Counting it here too was a real
        # bug (confirmed against a real transcript: this showed 102 units
        # when the true earned-so-far total, by hand, was 84 -- exactly the
        # gap created by including the 18 in-progress units on top of 56
        # earned + 28 AP). Keep it out of `completed_units` unconditionally,
        # and only fold it into the informational total when the user has
        # said to treat that quarter as done (mirroring completed_keys'
        # own toggle) -- if they've unchecked that because the quarter
        # hasn't really started, its units shouldn't count toward the
        # total either; extending the planning horizon to include that
        # quarter would let the solver's own scheduled_units pick them up
        # instead, via whatever it actually assigns there.
        _finished_rows = [c for c in completed_rows if not c.in_progress]
        _in_progress_rows_for_units = [c for c in completed_rows if c.in_progress]
        in_progress_units = sum(c.units_attempted for c in _in_progress_rows_for_units)
        completed_units = (
            sum(c.units_attempted for c in _finished_rows)
            + sum(a.units for a in ap_credits)
        )
        scheduled_units = sum(
            courses[key].units_typical for key, _q in _scheduled_occurrences() if key in courses
        )
        _counted_in_progress_units = in_progress_units if treat_in_progress_as_done else 0.0
        total_units = completed_units + _counted_in_progress_units + scheduled_units
        _breakdown = (
            f"{completed_units:.0f} earned/AP"
            + (f" + {_counted_in_progress_units:.0f} in progress" if _counted_in_progress_units else "")
            + f" + {scheduled_units:.0f} scheduled"
        )
        if total_units < single_degree_min:
            st.warning(
                f"Informational only (not enforced by the solver): this plan totals "
                f"{total_units:.0f} units ({_breakdown}), short of Stanford's "
                f"{single_degree_min}-unit graduation minimum by "
                f"{single_degree_min - total_units:.0f}. Add more courses or extend "
                f"the planning horizon."
            )
        else:
            st.caption(
                f"Units check (informational): {total_units:.0f} of {single_degree_min} "
                f"required for graduation ({_breakdown})."
            )
        if not treat_in_progress_as_done and in_progress_units:
            st.caption(
                f"({in_progress_units:.0f} in-progress unit(s) not counted above, since "
                "you're not treating that quarter as done -- extend the planning horizon "
                "to include it if you want the solver to plan it instead.)"
            )

        by_quarter: dict[str, list[str]] = {}
        for key, q in _scheduled_occurrences():
            by_quarter.setdefault(q, []).append(key)

        for q in quarter_list:
            if q not in by_quarter:
                continue
            st.subheader(q)
            for key in sorted(by_quarter[q]):
                c = courses[key]
                if is_placeholder_key(key):
                    # Most placeholders show a 3-5 unit RANGE here (a
                    # single "4u" would look like a firm number for a
                    # course that doesn't actually exist yet) -- but a few
                    # (the language requirement) have a genuinely fixed
                    # unit count despite not being a real course either
                    # (Stanford's language sequence is consistently 5
                    # units/quarter), so units_min == units_max collapses
                    # to a plain number instead of a same-to-same "5-5u".
                    if c.units_min == c.units_max:
                        units_label = f"{c.units_min:.0f}u"
                    else:
                        units_label = f"{c.units_min:.0f}-{c.units_max:.0f}u"
                else:
                    units_label = f"{c.units_typical:.0f}u"
                if key in result.repeat_quarters:
                    _rep = result.repeat_quarters[key]
                    units_label += f"  (repeat {_rep.index(q) + 1}/{len(_rep)})"
                st.text(f"  {key:10s} {c.title[:60]:60s} {units_label}")
