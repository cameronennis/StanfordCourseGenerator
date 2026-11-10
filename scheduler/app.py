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
    post_sophomore_quarter_indices, exact_quarter_match_exclusions,
)
from engine.models import CourseOverride, RequirementRule, WeightRule, all_referenced_course_keys, is_placeholder_key
from engine.requirements import (
    PROGRAM_SHEET_BUILDERS, BROKEN_PROGRAMS, build_requirements_grouped,
    list_variant_labels_by_code, placeholderize_large_elective_pools,
    apply_requirement_substitutions, apply_requirement_removals, all_rule_labels,
)
from engine.multi_program import (
    STACK_MODE_SECONDARY, STACK_MODE_DOUBLE_MAJOR, STACK_MODE_DOUBLE_DEGREE,
    build_restricted_course_programs, degree_type_of, is_undergrad_major_sheet,
    load_stack_modes, sheet_code, tag_program_id,
)
from engine.ap_credit import (
    guess_ap_equivalencies, flatten_course_items, EXAM_BOARDS, EXTERNAL_TEST_BY_ID, EXTERNAL_TEST_CHART,
    EXTERNAL_TEST_CHART_URL, EXTERNAL_TEST_CHART_YEAR, EXTERNAL_TEST_GENERAL_RULES,
)
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

# Streamlit throws away a widget's value whenever a run ends without that
# widget being drawn -- and every st.rerun() (adding a course, adding a
# rule, removing something...) ends the run early, before the widgets below
# it are drawn. Without this, adding a rule in section 4 silently reset
# "Quarters to plan ahead", "Full years already completed", even the
# program picked in 2a. `sticky()` keeps each value in its own
# session-state copy and restores it before the widget is drawn.
if "_sticky" not in st.session_state:
    st.session_state._sticky = {}


def sticky(widget_fn, label, *, key: str, default_arg: str | None = None, default=None, options=None, **kw):
    """Draw `widget_fn(label, key=key, ...)` with its last value restored.
    `default_arg`/`default`: the widget's own default parameter (e.g.
    "value"/2025, "index"/0, "default"/[]) -- only passed on first draw.
    `options`: passed through for choice widgets; a remembered value no
    longer in the options is dropped."""
    store = st.session_state._sticky
    if options is not None:
        kw["options"] = options
    remembered = store.get(key, None)
    if key in st.session_state:
        # Widget state still alive (normal case, including a fresh user edit) -- leave it.
        value = widget_fn(label, key=key, **kw)
        store[key] = value
        return value
    ok = key in store and (
        options is None
        or (isinstance(remembered, list) and all(v in options for v in remembered))
        or (not isinstance(remembered, list) and remembered in options)
    )
    if ok:
        st.session_state[key] = remembered
    elif default_arg is not None:
        kw[default_arg] = default
    value = widget_fn(label, key=key, **kw)
    store[key] = value
    return value


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
    interaction -- every checkbox, every "Remove" click in section 2b --
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
# instead (see "Your session-only corrections" (sidebar) below) and are applied
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
if "requirement_substitutions" not in st.session_state:
    st.session_state.requirement_substitutions = []
if "quarter_restrictions" not in st.session_state:
    st.session_state.quarter_restrictions = []
if "requirement_removals" not in st.session_state:
    st.session_state.requirement_removals = []
if "course_groupings" not in st.session_state:
    st.session_state.course_groupings = []

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
    interaction (e.g. removing a manually-added course in section 2b) was
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




def _normalize_code(raw: str) -> str:
    return raw.strip().upper().replace(" ", "")


def course_list_widget(state_key: str, label: str, empty_caption: str) -> list[str]:
    """Type-a-code-and-Enter box + "Did you mean" suggestions + .txt upload
    (one code per line) + removable list. Used by both 1b (courses you'll
    have completed) and 2b (your own course list); each gets its own
    session-state list via `state_key`. Returns the current list."""
    gen_key = f"{state_key}__gen"
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    if gen_key not in st.session_state:
        st.session_state[gen_key] = 0
    _gen = st.session_state[gen_key]
    _keys: list[str] = st.session_state[state_key]
    _real_keys = None

    def _real():
        nonlocal _real_keys
        if _real_keys is None:
            _real_keys = [k for k in courses if not is_placeholder_key(k)]
        return _real_keys

    _code_input = st.text_input(
        label,
        key=f"{state_key}_code_{_gen}",
        help="Case-insensitive, spaces ignored -- 'chem 131' and 'CHEM131' both work.",
    )
    if _code_input.strip():
        _code = _normalize_code(_code_input)
        if _code in courses and not is_placeholder_key(_code):
            if _code not in _keys:
                _keys.append(_code)
            st.session_state[gen_key] += 1
            st.rerun()
        else:
            st.error(f"{_code!r} isn't a course in the current snapshot -- check the code and try again.")
            # difflib "did you mean" -- ~15-20ms against the full snapshot, only on this not-found path.
            _suggestions = difflib.get_close_matches(_code, _real(), n=5, cutoff=0.6)
            if _suggestions:
                st.caption("Did you mean:")
                for _col, _sugg in zip(st.columns(len(_suggestions)), _suggestions):
                    with _col:
                        if st.button(_sugg, key=f"{state_key}_suggest_{_gen}_{_sugg}", help=courses[_sugg].title):
                            if _sugg not in _keys:
                                _keys.append(_sugg)
                            st.session_state[gen_key] += 1
                            st.rerun()

    _uploaded_list = st.file_uploader(
        "Or upload a .txt file (one course code per line)",
        type=["txt"],
        key=f"{state_key}_upload_{_gen}",
        help="Same normalization as the box above. Blank lines are skipped; codes that aren't "
             "in the current snapshot are reported below, not silently dropped.",
    )
    if _uploaded_list is not None:
        _added, _not_found = [], []
        for _raw_line in _uploaded_list.getvalue().decode("utf-8", errors="replace").splitlines():
            _line_code = _normalize_code(_raw_line)
            if not _line_code:
                continue
            if _line_code in courses and not is_placeholder_key(_line_code):
                if _line_code not in _keys:
                    _keys.append(_line_code)
                    _added.append(_line_code)
            else:
                _not_found.append(_line_code)
        if _added:
            st.success(f"Added {len(_added)} course(s) from the file: {', '.join(_added)}")
        if _not_found:
            st.error(
                f"{len(_not_found)} line(s) weren't found in the current snapshot -- check for "
                f"typos: {', '.join(_not_found)}"
            )
            for _bad in _not_found:
                _suggestions = difflib.get_close_matches(_bad, _real(), n=3, cutoff=0.6)
                if _suggestions:
                    st.caption(f"Did you mean, for {_bad!r}: {', '.join(_suggestions)}")
        if not _added and not _not_found:
            st.caption("That file didn't have any non-blank lines to read.")
        if _added and not _not_found:
            # Clean run -- reset the uploader. Left attached otherwise so the
            # messages above stay visible.
            st.session_state[gen_key] += 1
            st.rerun()

    if _keys:
        for _i, _key in enumerate(list(_keys)):
            _c = courses.get(_key)
            _cols = st.columns([6, 1])
            _cols[0].write(f"**{_key}** -- {_c.title if _c else '(not in current snapshot)'}")
            if _cols[1].button("Remove", key=f"{state_key}_remove_{_i}"):
                _keys.pop(_i)
                st.rerun()
    else:
        st.caption(empty_caption)
    return list(_keys)
# ---------------------------------------------------------------------------
# 1) Your starting point -- 1a (transcript) and 1b (manual credit entry) are
# merged as SETS, so a course or exam that shows up in both counts once.
# ---------------------------------------------------------------------------
st.header("1) Your starting point")
st.caption("Everything from 1a and 1b is combined -- a course that appears in both is only counted once.")
st.subheader("1a. Upload transcript (optional)")
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
        treat_in_progress_as_done = sticky(
            st.checkbox,
            f"Treat your {len(_in_progress_rows)} in-progress course(s) as already completed",
            key="treat_in_progress", default_arg="value", default=True,
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
                    if sticky(st.checkbox, _label, key=f"ap_auto_{_i}", default_arg="value", default=True):
                        ap_substituted_keys.update(_keys)
                elif len(_matches) > 1:
                    st.caption(
                        f"**{a.title}** ({a.units:.0f}u) -- {len(_matches)} possible matches; "
                        "the transcript's units alone can't tell which score applies:"
                    )
                    _options = ["Not sure / none of these"] + [c.exam_label for c in _matches]
                    _choice = sticky(
                        st.radio, "Which is yours?", options=_options, key=f"ap_choice_{_i}",
                        label_visibility="collapsed",
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
                _manual = sticky(
                    st.text_input,
                    f"Additional/override course code(s) for {a.title}",
                    key=f"ap_manual_{_i}", default_arg="value", default="",
                    help="Comma-separated. Use this if the built-in chart got it wrong for your "
                         "case, or for an exam it doesn't cover at all.",
                )
                for _code in _manual.split(","):
                    _code = _code.strip().upper().replace(" ", "")
                    if _code:
                        ap_substituted_keys.add(_code)
else:
    st.info("No transcript uploaded -- use 1b below, or plan from zero completed courses.")

# ---------------------------------------------------------------------------
# 1b. Manual course credit entry -- (1) courses that will already be done
# by the first planned quarter, (2) external test credit picked straight
# from Stanford's equivalency chart (engine/ap_credit.py's
# EXTERNAL_TEST_CHART, every exam board on that page).
# ---------------------------------------------------------------------------
st.subheader("1b. Manual course credit entry (optional)")
st.markdown("**Courses you'll have completed by the first planned quarter**")
st.caption(
    "Anything not on (or not yet on) your transcript -- e.g. this quarter's courses before grades "
    "post, or your whole history if you skipped the upload."
)
manual_completed_keys = course_list_widget(
    "manual_completed_keys",
    "Enter a completed course code and press Enter (e.g. \"math51\")",
    "No manually entered completed courses.",
)

st.markdown("**External test credit (AP, IB, A-Levels, ...)**")
st.caption(
    f"Matches Stanford's [External Test Credit Equivalency Charts]({EXTERNAL_TEST_CHART_URL}) "
    f"(effective for students matriculating {EXTERNAL_TEST_CHART_YEAR}; your own admit year's "
    f"bulletin governs, so double-check if you matriculated earlier). {EXTERNAL_TEST_GENERAL_RULES}"
)
if "test_credit_ids" not in st.session_state:
    st.session_state.test_credit_ids = []
_tc_cols = st.columns([2, 3, 3, 1])
with _tc_cols[0]:
    _tc_board = st.selectbox(
        "Exam", options=list(EXAM_BOARDS), key="tc_board",
        format_func=lambda b: f"{b} -- {EXAM_BOARDS[b]}",
    )
_tc_board_rows = [r for r in EXTERNAL_TEST_CHART if r.board == _tc_board]
with _tc_cols[1]:
    _tc_subject = st.selectbox(
        "Subject", options=list(dict.fromkeys(r.subject for r in _tc_board_rows)), key=f"tc_subject_{_tc_board}",
    )
_tc_subject_rows = [r for r in _tc_board_rows if r.subject == _tc_subject]
with _tc_cols[2]:
    _tc_row_id = st.selectbox(
        "Score", options=[r.row_id for r in _tc_subject_rows], key=f"tc_score_{_tc_board}_{_tc_subject}",
        format_func=lambda rid: (lambda r: f"{r.score}  ({r.units:g}u -> {r.courses_text})")(EXTERNAL_TEST_BY_ID[rid]),
    )
with _tc_cols[3]:
    st.write("")
    st.write("")
    if st.button("Add", key="tc_add"):
        if _tc_row_id not in st.session_state.test_credit_ids:
            st.session_state.test_credit_ids.append(_tc_row_id)
        st.rerun()

test_credit_rows = [EXTERNAL_TEST_BY_ID[i] for i in st.session_state.test_credit_ids if i in EXTERNAL_TEST_BY_ID]
for _i, _r in enumerate(test_credit_rows):
    _cols = st.columns([6, 1])
    _cols[0].write(f"**{_r.board} {_r.subject}**, score {_r.score} -- {_r.units:g} units; satisfies {_r.courses_text}"
                   + (f"  \n_{_r.note}_" if _r.note else ""))
    if _cols[1].button("Remove", key=f"tc_remove_{_i}"):
        st.session_state.test_credit_ids.pop(_i)
        st.rerun()
# The chart's own one-exam-per-subject rule (language and physics excepted).
_tc_sections = [r.section for r in test_credit_rows if r.section not in ("Language", "Physics")]
for _sec in sorted({s for s in _tc_sections if _tc_sections.count(s) > 1}):
    st.warning(
        f"More than one {_sec} exam entered -- Stanford only awards units for one exam per subject "
        "(the highest-satisfying one). All of them are counted below; remove the ones that won't post."
    )
test_credit_keys: set[str] = set()
for _r in test_credit_rows:
    test_credit_keys |= flatten_course_items(_r.satisfies)
test_credit_fulfills_language = [r for r in test_credit_rows if r.fulfills_language]

transcript_keys = {
    c.key for c in completed_rows if not c.in_progress or treat_in_progress_as_done
} | ap_substituted_keys
completed_keys = transcript_keys | set(manual_completed_keys) | test_credit_keys
# Units for the informational graduation-unit check: manual courses not
# already on the transcript, plus test credit whose courses the transcript
# doesn't already cover (so the same AP exam isn't counted from both places).
_transcript_row_keys = {c.key for c in completed_rows}
manual_completed_units = sum(
    courses[k].units_typical for k in set(manual_completed_keys) - _transcript_row_keys if k in courses
)
manual_test_units = sum(
    r.units for r in test_credit_rows
    if not (flatten_course_items(r.satisfies) and flatten_course_items(r.satisfies) <= transcript_keys)
)
# ---------------------------------------------------------------------------
# 2a) Program sheets -- two-step picker: pick a program by CODE, then pick
# which specific subplan/path/version of it applies to you. Deliberately no
# default selection anywhere in here (see list_variant_labels_by_code()'s
# docstring in engine/requirements.py for why a silent default -- the old
# picker pre-selected "CS-BS (core subset, v1, hand-encoded)", a tiny v1
# placeholder nowhere near the real major -- is actively dangerous: a real
# user's generated schedule quietly used that stub instead of the real
# CS-BS requirements).
# ---------------------------------------------------------------------------
st.header("2a) Requirements to plan for (optional)")
_variants_by_code = list_variant_labels_by_code()
_selectable_codes = sorted(
    code for code, variants in _variants_by_code.items()
    if any(full_name not in BROKEN_PROGRAMS for full_name in variants.values())
)
selected_codes = sticky(
    st.multiselect,
    "Which program(s) are you working toward?",
    options=_selectable_codes,
    key="selected_codes", default_arg="default", default=[],
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
    _choice = sticky(
        st.selectbox,
        f"{_code}: which version/subplan?",
        options=[_placeholder] + sorted(_working.keys()),
        key=f"variant_choice_{_code}", default_arg="index", default=0,
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
# 2a (cont.) Stacking two undergrad majors (secondary major / double major / double
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
    st.subheader("Combining your two majors")
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
    stack_mode = sticky(
        st.selectbox,
        "How do these two majors combine?",
        options=_available_modes,
        format_func=lambda m: _stack_modes_info[m]["label"],
        key="stack_mode", default_arg="index", default=0,
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

# ---------------------------------------------------------------------------
# 2b) Your own course list (optional) -- courses you want sequenced, on top
# of (or instead of) 2a. Hard-required (kind="all"). A listed course that
# also belongs to a 2a requirement counts toward it once -- it's never
# scheduled twice, and a collapsed elective pool uses it as one of its
# slots instead of adding a generic placeholder on top (see
# placeholderize_large_elective_pools' credit_keys).
# ---------------------------------------------------------------------------
st.header("2b) Your own course list (optional)")
st.caption(
    "Courses you want in the plan, on top of (or instead of) 2a. Every course here is required. "
    "If one also counts toward a 2a requirement, it fills that requirement instead of being added twice. "
    "2c's university-wide requirements still \"fill in\" around this list unless you turn them off."
)
manual_course_keys = course_list_widget(
    "manual_course_keys",
    "Enter a course code and press Enter (e.g. \"chem131\")",
    "No courses added yet -- type a code above or upload a .txt file.",
)
use_manual_courses = bool(manual_course_keys)

# Build the program-sheet rule list now that 1 (completed) and 2b (listed)
# are both known. Order matters:
#  1. personal requirement substitutions from section 4 (before pools are
#     collapsed, so a substitution inside a big elective pool still lands);
#  2. collapse big elective pools, letting completed + listed courses fill
#     slots first (each course fills at most one pool -- shared across both
#     majors only when double-counting is restricted);
#  3. append the 2b list itself.
_subs = st.session_state.get("requirement_substitutions") or []
_removals = st.session_state.get("requirement_removals") or []
# A removal counts as "used" if it matched in ANY rule list (sheets, 2b list, university-wide).
_removals_matched: set[int] = set()


def _apply_removals(rule_list):
    new, unmatched = apply_requirement_removals(rule_list, _removals)
    _unmatched_ids = {id(r) for r in unmatched}
    _removals_matched.update(id(r) for r in _removals if id(r) not in _unmatched_ids)
    return new

_credit_order = sorted(completed_keys) + [k for k in manual_course_keys if k not in completed_keys]
_shared_claims: set[str] = set()
requirement_rules: list = []
for _name in selected_sheets:
    _sheet_rules = _apply_removals(apply_requirement_substitutions(_grouped_rules.get(_name, []), _subs))
    _sheet_rules, _sheet_elective_placeholders = placeholderize_large_elective_pools(
        _sheet_rules,
        credit_keys=_credit_order,
        claimed=_shared_claims if restricted_course_programs else set(),
    )
    if _sheet_elective_placeholders:
        courses = add_placeholder_courses(courses, _sheet_elective_placeholders)
        placeholder_courses.update(_sheet_elective_placeholders)
    requirement_rules.extend(_sheet_rules)
if manual_course_keys:
    requirement_rules.extend(_apply_removals(apply_requirement_substitutions([RequirementRule(
        label=f"Your own course list -- {len(manual_course_keys)} course(s), all required",
        kind="all",
        courses=manual_course_keys,
        source="Manually added by you (not derived from any program sheet or university requirement)",
    )], _subs)))

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
# 2c) University-wide requirements (WAYS, COLLEGE, Writing, Language) -- see
# engine/university_requirements.py's module docstring for exactly what's
# covered and what's deliberately left out (units/GPA/residency/pass-fail,
# double-major/coterm interactions). Applies ON TOP of whatever program
# sheet(s) were selected above, same rule-list contract.
# ---------------------------------------------------------------------------
st.header("2c) University-wide requirements")
# Freely toggleable regardless of "2b) Your own course list" -- the two
# are additive, not mutually exclusive. With a manual list active, leaving
# this checked means the solver "fills in" whatever WAYS/COLLEGE/Writing/
# Language courses your own list doesn't already happen to satisfy (a
# course you added yourself that happens to carry a WAYS tag still counts,
# same double-counting-aware credit as anywhere else in this app); leaving
# it unchecked means your list is treated as fully self-contained and
# university requirements aren't checked at all. Previously this was
# force-disabled whenever the custom list was active, on the assumption that a custom
# list meant taking on full responsibility for every requirement yourself
# -- changed after user feedback that a custom list is often built to
# cover major-specific requirements only, with university-wide ones still
# meant to be auto-filled same as usual.
include_university_reqs = sticky(
    st.checkbox,
    "Include Stanford's university-wide requirements (WAYS, COLLEGE, Writing, Language)",
    key="include_university_reqs", default_arg="value", default=True,
    help=(
        "On top of whatever program sheet(s) and/or your own course list (section 2b) "
        "you're using. Units/GPA/residency/pass-fail aren't modeled as solver "
        "constraints -- see the informational 180-unit check after generating a "
        "schedule instead."
        + (
            " With \"2b) Your own course list\" active, this \"fills in\" extra "
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
        admit_fall_year = sticky(
            st.number_input,
            "Admit year (fall you started/will start)",
            min_value=2020, max_value=2035, key="admit_fall_year", default_arg="value", default=2025,
            help="Selects the 2-quarter (through 2026-2027) vs. 3-quarter (Fall 2027+, "
                 "best-guess) COLLEGE policy.",
        )
    with ucol2:
        if test_credit_fulfills_language:
            # The equivalency chart says these exams fulfill the university
            # language requirement outright (1b's test credit entry).
            language_quarters_needed = 0
            st.success(
                "Language requirement fulfilled by your test credit ("
                + ", ".join(f"{r.board} {r.subject} {r.score}" for r in test_credit_fulfills_language)
                + ")."
            )
        else:
            language_quarters_needed = sticky(
                st.number_input,
                "Quarters of language requirement still needed",
                min_value=0, max_value=6, key="language_quarters_needed", default_arg="value", default=0,
                help="Not tied to real course data -- depends on your placement test, which "
                     "this tool has no data source for (a qualifying language exam entered in "
                     "1b fulfills it automatically). Enter how many quarters you still owe; "
                     "it's modeled as a generic placeholder course repeated that many times. "
                     "Replace with your real language course(s) before trusting the schedule.",
            )
    # Every course already required by a program sheet or your own manual
    # list (section 2b), so WAYS/COLLEGE/Writing can credit one of THOSE
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
    university_rules = _apply_removals(university_rules)
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
st.header("3) Weight table (optional)")
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
# 4) Add rules (natural language) -- the "natural-language override
# box" the README's "Stubbed" section flagged as a natural next step. Each
# request is parsed IMMEDIATELY (so the log below fills in right away),
# but parsing alone never re-solves anything -- it only appends to
# st.session_state.session_overrides / .weight_rows, the same two lists
# the rest of this app already reads from once "Generate schedule" (below)
# is actually clicked. That button press is the only thing that ever
# triggers a solve, so any number of requests can queue up here first,
# exactly like editing the weight table above doesn't auto-generate either.
# ---------------------------------------------------------------------------
st.header("4) Add rules (describe them in plain English)")
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
_nl_sentence = st.text_input("Describe a rule", key=f"nl_change_input_{_nl_gen}")
if st.button("Add rule", key=f"nl_change_btn_{_nl_gen}"):
    if not _nl_sentence.strip():
        st.warning("Type something first.")
    else:
        with st.spinner("Interpreting..."):
            _result = parse_change_request(
                _nl_sentence, known_course_keys=set(courses.keys()), courses=courses,
                rule_labels=all_rule_labels(requirement_rules),
            )
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
        elif _result.change_type == "requirement_substitution" and _result.requirement_substitution is not None:
            st.session_state.requirement_substitutions.append(_result.requirement_substitution)
        elif _result.change_type == "quarter_restriction" and _result.quarter_restriction is not None:
            st.session_state.quarter_restrictions.append(_result.quarter_restriction)
        elif _result.change_type == "requirement_removal" and _result.requirement_removal is not None:
            st.session_state.requirement_removals.append(_result.requirement_removal)
        elif _result.change_type == "course_grouping" and _result.course_grouping is not None:
            st.session_state.course_groupings.append(_result.course_grouping)
        st.session_state.nl_change_gen += 1
        st.rerun()

for _rm in _removals:
    if id(_rm) not in _removals_matched:
        st.warning(
            f"Waiver for {_rm.course_key or repr(_rm.rule_label)} doesn't match anything required in your "
            "current plan (2a/2b/2c), so it has no effect right now. A course is only waived where it's "
            "required outright -- not where it's one option in an elective list."
        )

if st.session_state.nl_change_log:
    st.subheader("Rules added")
    for _entry in reversed(st.session_state.nl_change_log):
        if _entry["change_type"] == "course_override":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (course correction, this session only)")
        elif _entry["change_type"] == "weight_rule":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (added as a weight rule above)")
        elif _entry["change_type"] == "requirement_substitution":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (requirement substitution, this session only)")
        elif _entry["change_type"] == "quarter_restriction":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (hard quarter restriction, this session only)")
        elif _entry["change_type"] == "requirement_removal":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (requirement waived, this session only)")
        elif _entry["change_type"] == "course_grouping":
            st.success(f"**\"{_entry['sentence']}\"** -- {_entry['summary']} (hard same-period rule, this session only)")
        else:
            st.warning(f"**\"{_entry['sentence']}\"** -- not applied: {_entry['error'] or 'unrecognized request'}")

# ---------------------------------------------------------------------------
# 5) Planning horizon + generate
# ---------------------------------------------------------------------------
st.header("5) Generate")
col1, col2, col3, col4 = st.columns(4)
with col1:
    start_year = sticky(st.text_input, "Starting catalog year (next quarter to plan)", key="start_year",
                        default_arg="value", default="2026-2027")
with col2:
    num_quarters = sticky(st.number_input, "Quarters to plan ahead", min_value=1, max_value=16,
                          key="num_quarters", default_arg="value", default=8)
with col3:
    unit_cap = sticky(st.number_input, "Max units per quarter", min_value=1, max_value=30,
                      key="unit_cap", default_arg="value", default=18)
with col4:
    years_completed_at_start = sticky(
        st.number_input,
        "Full years already completed",
        min_value=0, max_value=6, key="years_completed_at_start", default_arg="value", default=0,
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


start_season = sticky(st.selectbox, "Starting season", options=list(SEASON_ROTATION), key="start_season",
                      default_arg="index", default=1)
quarter_list = build_quarter_list(start_year, start_season, int(num_quarters))
st.caption("Planning horizon: " + ", ".join(quarter_list))

def _settings_fingerprint() -> str:
    """Everything that feeds the solver, hashed -- used to tell whether the
    schedule on screen is still current."""
    import hashlib
    blob = repr((
        requirement_rules, weight_rules, sorted(completed_keys), list(quarter_list), float(unit_cap),
        int(years_completed_at_start), st.session_state.session_overrides,
        st.session_state.quarter_restrictions, sorted(restricted_course_programs.items()),
        st.session_state.course_groupings,
    ))
    return hashlib.md5(blob.encode()).hexdigest()


if st.button("Generate schedule", type="primary"):
    # Manually-added courses (section 2b) skip their OWN prerequisite
    # enforcement -- someone typing in "the entire CHEM-BS" already has a
    # specific set of quarters in mind for it, often on top of years
    # already completed (e.g. "I've finished freshman year, just plan the
    # next three"); auto-inserting an earlier course they didn't ask for
    # just to "prove" a prerequisite would work against that, not for it.
    # Only skips prereqs FOR these courses themselves -- one of them can
    # still be enforced as someone else's prerequisite as normal. Read from
    # session_state directly (not the local `manual_course_keys` variable
    # from section 2b, which only exists in that branch's scope) so this is
    # correct whether or not 2b has anything in it on this run.
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
    # WR2 (Writing and Rhetoric 2) is HARD-forbidden junior year or later --
    # direct user correction: "you cannot take WR2 as a junior or senior."
    # Merged into the same hard-exclusion dict as CHEM176/Capstone above,
    # rather than a separate soft preference. A no-op if "PLACEHOLDER:WR2"
    # isn't one of this run's requirement keys at all (already satisfied by
    # a completed course, or this run built no university-wide rules).
    _class_year_exclusions["PLACEHOLDER:WR2"] = post_sophomore_quarter_indices(
        quarter_list, years_completed_at_horizon_start=int(years_completed_at_start)
    )
    # Personal, this-session-only HARD quarter pins from the NL change box
    # (e.g. "enforce CHEM171 in sophomore spring, no compromises") -- see
    # models.QuarterRestriction's own docstring for the real bug this
    # fixes (the NL box previously had no way to express an absolute
    # class-year+season target at all, and silently misused terms_offered
    # instead, which can't pin a specific year). Each restriction excludes
    # every quarter except its one real target -- if the horizon never
    # actually reaches that class year/season, this correctly surfaces as
    # INFEASIBLE rather than quietly picking the closest quarter instead.
    for _qr in st.session_state.quarter_restrictions:
        _class_year_exclusions[_qr.course_key] = exact_quarter_match_exclusions(
            quarter_list,
            years_completed_at_horizon_start=int(years_completed_at_start),
            target_class_year=_qr.class_year,
            target_season=_qr.season,
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
            same_period_groups=[(g.course_keys, g.scope) for g in st.session_state.course_groupings],
        )

    st.session_state.last_schedule = {
        "result": result,
        "fingerprint": _settings_fingerprint(),
        "quarter_list": list(quarter_list),
        "courses": courses,
        "graduation_unit_target": graduation_unit_target,
        "completed_rows": list(completed_rows),
        "ap_credits": list(ap_credits),
        "treat_in_progress_as_done": treat_in_progress_as_done,
        "extra_completed_units": manual_completed_units + manual_test_units,
    }

# The latest schedule is kept in session state and re-drawn on every rerun,
# so adding a rule in section 4 (or anything else that reruns the page)
# doesn't wipe it. If settings changed since it was made, say so instead of
# silently showing a stale plan as if it were current.
_last = st.session_state.get("last_schedule")
if _last is not None:
    result = _last["result"]
    quarter_list = _last["quarter_list"]
    courses = _last["courses"]
    graduation_unit_target = _last["graduation_unit_target"]
    completed_rows = _last["completed_rows"]
    ap_credits = _last["ap_credits"]
    treat_in_progress_as_done = _last["treat_in_progress_as_done"]
    _extra_completed_units = _last["extra_completed_units"]
    if _last["fingerprint"] != _settings_fingerprint():
        st.info(
            "Your settings or rules changed since this schedule was generated -- it's the previous "
            "result. Click \"Generate schedule\" to apply the changes."
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
            + _extra_completed_units  # 1b: manual completed courses + test credit, de-duplicated
        )
        scheduled_units = sum(
            courses[key].units_typical for key, _q in _scheduled_occurrences() if key in courses
        )
        _counted_in_progress_units = in_progress_units if treat_in_progress_as_done else 0.0
        total_units = completed_units + _counted_in_progress_units + scheduled_units
        _breakdown = (
            f"{completed_units:.0f} earned/test credit"
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
