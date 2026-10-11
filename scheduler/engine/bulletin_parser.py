"""
General parser for Stanford Bulletin program pages (bulletin.stanford.edu/
programs/{CODE}), turning the page's rendered text into RequirementRule
trees -- the scale-up path for engine/requirements.py, which so far only
has one *hand-encoded* subset (CS-BS). Rather than hand-encoding every
major/minor/coterm, this reads the same nested "complete ALL / at least N /
ANY, joined by AND/OR" structure that every program page uses and builds
the tree automatically.

Source of the text: fetch via a browser (Claude in Chrome's get_page_text
on the article element), since the page is a client-rendered Nuxt app and
this org's web-fetch is disabled. IMPORTANT: the page sometimes returns
before Vue has hydrated course links, showing raw internal IDs (e.g.
"1056451") instead of "CS106A - Programming Abstractions" -- always fetch
the text twice (or wait a beat) and use the second read.

This is a heuristic parser, not a guaranteed-correct one -- see
`ParseWarning` below. Validate its output against a hand-checked source
before trusting a given program's tree, the same way the transcript parser
and prerequisite extraction ask for review rather than blind trust.

A requirement that names no specific courses at all -- either pure free
text ("six math courses at or above Math 51") or an unenumerated pool
("any MATH course numbered 100 and above") -- gets a PLACEHOLDER course
instead of being silently dropped or (worse) having a neighboring rule's
real course list misattributed to it. See models.make_placeholder_course
and ProgramRequirements.placeholder_courses. Every placeholder creation
also emits a ParseWarning, so it's never silent.

Known limitations (deliberately not handled -- flagged instead):
  - Cross-block prose constraints ("31 units must be completed between the
    Required Course, WIM/Design/Capstone, and Electives") aren't turned
    into a constraint; they're kept as unparsed notes on the section.
  - "Earn at least N credits" is a UNIT-based rule (kind="unit_min");
    "Complete at least N of the following Courses" is a COURSE-COUNT rule
    (kind="at_least_n"). Mixing these up would silently produce a wrong
    constraint, so the header text is matched exactly.
  - A course-list "slot" can itself require an AND-group as a single
    alternative (e.g. "(MATH51 AND MATH53) OR (CME100 AND CME102)") --
    represented as a tuple-of-tuples; make sure solver.py's alternative
    handling (currently single-course tuples only) gets extended before
    relying on this for a program that uses it, or it'll under-constrain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import RequirementRule, PLACEHOLDER_PREFIX

# A course reference that fetch_bulletin.py couldn't resolve to a real
# "SUBJ### - Title" line -- either a transient miss or (per its
# UNRESOLVED_MARKER caching) a confirmed 404 on bulletin.stanford.edu's own
# /courses/{id} page -- gets left in the sample text as this bare internal
# numeric id (see fetch_bulletin.py's resolve_course_ids_in_lines). It is
# NOT a course line (no letters), so it used to fall through to the catch-
# all "text" classification, which had two bad effects: inside a course
# list, _parse_course_slots() treats any non-course-list line as "list
# over" and stops -- silently truncating every real course after the dead
# id (this is what made ARCHA-MIN's "Self-Designed Path" and 10 other
# programs' sub-rules solver-INFEASIBLE: a whole elective list collapsed
# down to whatever few courses happened to precede the id). And outside a
# course list, parse_one_item()'s label-scanning loop would pick the bare
# id up AS a rule's label, then misattribute the next real header line to
# it -- producing a phantom zero-course rule and shifting every following
# label/header pairing in that section by one.
#
# A missing single option out of an OR-style list ("earn N credits from
# the following: A, B, [dead], C, D...") doesn't need a placeholder the
# way an unenumerable free-text requirement does (see PLACEHOLDER_PREFIX
# below) -- the other real alternatives already make the requirement
# satisfiable, and a placeholder for a course nobody can actually take
# would just be dead weight. So this is classified distinctly and simply
# skipped wherever it's found (see _parse_course_slots and
# parse_one_item), with a warning so it's never silent.
_BARE_ID_RE = re.compile(r"^\d{5,8}$")

COURSE_LINE_RE = re.compile(r"^([A-Z][A-Z&]{1,9}\d[A-Z0-9]*)\s*-\s*(.+)$")
OR_ALT_RE = re.compile(r"^OR\s+([A-Z][A-Z&]{1,9}\d[A-Z0-9]*)\s*-\s*(.+)$")
AND_ALT_RE = re.compile(r"^AND\s+([A-Z][A-Z&]{1,9}\d[A-Z0-9]*)\s*-\s*(.+)$")
CONNECTOR_RE = re.compile(r"^(AND|OR)$")

# Some course-list "slots" name a whole unenumerated pool instead of a
# specific course, e.g. "MATH Courses Numbered 100 and Above" or "CS
# Courses Numbered 110 and Above" (seen in DATSC-BS's Technical Electives).
# These don't match COURSE_LINE_RE (no course number directly on the
# subject, no " - Title"), and previously fell through to "text", silently
# breaking the course-slot scan and getting swallowed as if it were the
# NEXT item's label -- corrupting that next item's course list. Recognize
# them explicitly and turn each into its own placeholder course (see
# models.make_placeholder_course) instead.
COURSE_SET_RE = re.compile(
    r"^([A-Z][A-Z&]{1,9})\s+Courses?\s+Numbered\s+(\d+)\s*(and above|and up|or above|and below|or below)?\s*$",
    re.IGNORECASE,
)
OR_COURSE_SET_RE = re.compile(
    r"^OR\s+([A-Z][A-Z&]{1,9})\s+Courses?\s+Numbered\s+(\d+)\s*(and above|and up|or above|and below|or below)?\s*$",
    re.IGNORECASE,
)

# A requirement whose body is pure prose with no enumerated course list at
# all, e.g. MATH-MIN's "The Math minor consists of six math courses at or
# above Math 51." -- look for a leading count so the placeholder can stand
# in for the RIGHT NUMBER of courses, not just one.
_NUMBER_WORDS_EXT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_LEAD_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:[A-Za-z]+\s+){0,2}courses?\b",
    re.IGNORECASE,
)

#  "...following Courses:" is the usual noun; "...following Course Sets:"
# shows up when the list is made of unenumerated pools (e.g. DATSC-BS's
# "Complete ANY of the following Course Sets: / MATH Courses Numbered 100
# and Above") rather than specific courses -- same kind/n semantics either
# way, so accept both.
HEADER_ALL_RE = re.compile(r"^Complete ALL of the following (?:Courses|Course Sets):$", re.IGNORECASE)
HEADER_AT_LEAST_RE = re.compile(r"^Complete at least (\d+) of the following (?:Courses|Course Sets):$", re.IGNORECASE)
HEADER_ANY_RE = re.compile(r"^Complete ANY of the following (?:Courses|Course Sets):$", re.IGNORECASE)
HEADER_EARN_CREDITS_RE = re.compile(r"^Earn at least (\d+) credits? from the following:$", re.IGNORECASE)
# A course-count header phrased as "Complete exactly N course(s) and earn M
# or more credit(s) from the following:" or "Complete N or more course(s)
# and earn M or more credit(s) from the following:" (seen in MATSC-BS's
# Technology in Society and BIOE-BS's Depth Electives). Semantically this is
# still just an at-least-N-courses requirement -- the "...and earn M
# credit(s)" clause is an additional per-requirement unit floor this parser
# doesn't separately enforce (same documented limitation as other
# cross-block prose constraints), but the course count and course list are
# real and fully enumerated, so they must not be dropped or placeholder-ed.
HEADER_COUNT_AND_CREDITS_RE = re.compile(
    r"^Complete (?:exactly (\d+)|(\d+) or more) course\(s\) and earn \d+ or more credit\(s\) from the following:$",
    re.IGNORECASE,
)
WRAPPER_ALL_RE = re.compile(r"^Fulfill ALL of the following requirements:$", re.IGNORECASE)
WRAPPER_ANY_RE = re.compile(r"^Fulfill ANY of the following requirements:$", re.IGNORECASE)
# "Complete at least N of the following:" (no trailing "Courses") is a
# *different* construct from HEADER_AT_LEAST above -- empirically, when
# "Courses" is omitted, what follows is a set of NAMED sub-groups (e.g.
# "Area I: AI Methods", "Area II: ...") joined by OR, not raw course
# lines. Treat it like WRAPPER_ANY but with an explicit N instead of 1.
WRAPPER_AT_LEAST_N_RE = re.compile(r"^Complete at least (\d+) of the following:$", re.IGNORECASE)
SECTION_MARKER = "Completion requirement"

# "Fulfill ANY of the following requirements:" always defaults to n=1 in
# this parser (satisfy any *one* named alternative) -- but some programs
# phrase a genuinely different count (e.g. "choose two courses from
# different areas, but no more than one course per area") only in the
# free-text label next to the wrapper, not in a machine-parseable header.
# CS-BS phrases the same *kind* of requirement with an explicit
# "Complete at least 2 of the following:" header (parsed correctly, see
# WRAPPER_AT_LEAST_N_RE) -- CS-MIN does not, which is exactly the
# real-world inconsistency this check exists to catch.
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_COUNT_HINT_RE = re.compile(
    r"\b(?:choose|select|pick)\s+(\d+|one|two|three|four|five|six)\b", re.IGNORECASE
)


@dataclass
class ParseWarning:
    where: str
    message: str


@dataclass
class ProgramSection:
    title: str
    rules: list[RequirementRule] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ProgramRequirements:
    code: str
    title: str
    units_min: float | None
    university_units_min: float | None
    sections: list[ProgramSection] = field(default_factory=list)
    warnings: list[ParseWarning] = field(default_factory=list)
    # key -> human-readable description, for every placeholder course this
    # parse created (see models.make_placeholder_course). Empty for a
    # program whose bulletin page was fully machine-parseable.
    placeholder_courses: dict[str, str] = field(default_factory=dict)


def _to_key(code: str) -> str:
    return code.upper().replace(" ", "")


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    return s[:50] or "ITEM"


def _looks_like_a_label(line: str) -> bool:
    """Heuristic to tell a short block label (e.g. "Core", "Track C
    Requirement", "Science Elective") apart from a descriptive prose
    sentence (e.g. "MATH 19, MATH 20, and MATH 21, or AP Calculus Credit
    may be used..."). Labels are short and don't end in sentence-final
    punctuation; prose does. Used to stop swallowing trailing notes right
    before the next item's label instead of eating it."""
    if line.endswith(":"):
        return True  # e.g. "Electives (choose two ... per area):" -- always a label/lead-in
    if line.endswith((".", "!", "?")):
        return False  # sentence-final punctuation -- prose
    return len(line) <= 70


def _classify(line: str) -> str:
    if _BARE_ID_RE.match(line):
        return "unresolved_id"
    if WRAPPER_ALL_RE.match(line):
        return "wrapper_all"
    if WRAPPER_ANY_RE.match(line):
        return "wrapper_any"
    if WRAPPER_AT_LEAST_N_RE.match(line):
        return "wrapper_at_least_n"
    if HEADER_EARN_CREDITS_RE.match(line):
        return "header_earn_credits"
    if HEADER_ALL_RE.match(line):
        return "header_all"
    if HEADER_AT_LEAST_RE.match(line):
        return "header_at_least"
    if HEADER_ANY_RE.match(line):
        return "header_any"
    if HEADER_COUNT_AND_CREDITS_RE.match(line):
        return "header_count_credits"
    if CONNECTOR_RE.match(line):
        return "connector"
    if AND_ALT_RE.match(line):
        return "and_alt"
    if OR_ALT_RE.match(line):
        return "or_alt"
    if COURSE_LINE_RE.match(line):
        return "course"
    if OR_COURSE_SET_RE.match(line):
        return "or_course_set"
    if COURSE_SET_RE.match(line):
        return "course_set"
    return "text"


class _Parser:
    """Line-based recursive-descent parser over one section's body lines.

    Lines are pre-split (blank lines removed, each line stripped). A small
    pending-line buffer holds text seen since the last recognized
    header/wrapper/connector; when a header or wrapper is hit, the *first*
    buffered line becomes the label for the item being started (labels are
    always a short line right after the previous boundary, e.g. "Required
    Math Courses", "Track C Requirement"), and any further buffered lines
    are descriptive prose, kept as notes rather than guessed at (there's no
    strict guarantee they belong to this item vs. the previous one -- keep
    them as human-reviewable text either way).
    """

    def __init__(
        self,
        lines: list[str],
        warn_ctx: str,
        warnings: list[ParseWarning],
        placeholders: dict[str, str],
        code: str = "",
    ):
        self.lines = lines
        self.pos = 0
        self.n = len(lines)
        self.warn_ctx = warn_ctx
        self.warnings = warnings
        self.pending_notes: list[str] = []
        self.placeholders = placeholders
        self.code = code

    def peek(self) -> str | None:
        return self.lines[self.pos] if self.pos < self.n else None

    def advance(self) -> str:
        line = self.lines[self.pos]
        self.pos += 1
        return line

    def warn(self, msg: str):
        self.warnings.append(ParseWarning(self.warn_ctx, msg))

    def _warn_if_label_implies_different_count(self, label: str | None, n: int):
        if not label:
            return
        m = _COUNT_HINT_RE.search(label)
        if not m:
            return
        word = m.group(1).lower()
        implied = int(word) if word.isdigit() else _NUMBER_WORDS.get(word)
        if implied is not None and implied != n:
            self.warn(
                f"label {label!r} implies choosing {implied}, but this wrapper defaults to "
                f"n={n} (no machine-parseable count header) -- verify and correct manually"
            )

    # -- top-level: a section body is a sequence of independent items --
    def parse_section_body(self) -> tuple[list[RequirementRule], list[str]]:
        """Parse the section's top-level items one at a time.

        Each call to parse_one_item() either returns a rule, or -- having
        found no header/wrapper before running out of lines -- returns None
        and leaves whatever it scanned sitting in self.pending_notes. That
        None case means THIS item's text was pure prose with no enumerated
        course list, e.g. an open-ended guideline paragraph after the real
        requirements (CE-MIN's "Requirements Guidelines"), or an
        unrecognized header phrasing before HEADER_COUNT_AND_CREDITS_RE and
        friends existed to catch it.

        Only the whole-section-is-prose case used to get a placeholder
        (parse_program_text checked "not rules and notes" once, after this
        entire method returned) -- so if even one earlier top-level item in
        the same section had already parsed successfully, this item's
        leftover prose would sit invisibly in the section's notes forever,
        with no warning and no placeholder. Checking per item, right when
        each one finishes, means a single bad item can't hide behind its
        successfully-parsed siblings.
        """
        rules: list[RequirementRule] = []
        while self.pos < self.n:
            notes_before = len(self.pending_notes)
            rule = self.parse_one_item()
            if rule is not None:
                rules.append(rule)
                continue
            # No header/wrapper ever found for this item -- everything
            # buffered since the previous item (or section start) is prose.
            leftover = self.pending_notes[notes_before:]
            if leftover:
                placeholder_rule = _make_placeholder_rule_from_prose(
                    self.warn_ctx, leftover, self.code, self.warnings, self.placeholders
                )
                if placeholder_rule is not None:
                    rules.append(placeholder_rule)
            break
        return rules, self.pending_notes

    def parse_one_item(self) -> RequirementRule | None:
        """Consume lines up to and including one top-level item (a leaf
        rule, or a Fulfill-wrapped tree), buffering any preceding
        unrecognized lines as notes. Returns None if we run out of
        recognizable structure (rest is trailing prose)."""
        label = None
        while self.pos < self.n:
            line = self.peek()
            kind = _classify(line)
            if kind in ("wrapper_all", "wrapper_any", "wrapper_at_least_n"):
                header_line = self.advance()
                if kind == "wrapper_all":
                    combine, n = "all", None
                elif kind == "wrapper_any":
                    combine, n = "at_least_n", 1
                else:
                    combine, n = "at_least_n", int(WRAPPER_AT_LEAST_N_RE.match(header_line).group(1))
                if kind == "wrapper_any":
                    self._warn_if_label_implies_different_count(label, n=1)
                sub_rules = self._parse_wrapped_list(combine)
                return RequirementRule(
                    label=label or "",
                    kind=combine,
                    n=n,
                    sub_rules=sub_rules,
                )
            if kind in (
                "header_all", "header_at_least", "header_any",
                "header_earn_credits", "header_count_credits",
            ):
                return self._parse_leaf(label)
            if kind == "unresolved_id":
                # A dead/unresolvable course reference that landed outside
                # a recognized course list (e.g. between one item's course
                # list and the next item's label) -- see _BARE_ID_RE's
                # docstring. Skip it outright: NOT as a label (that's what
                # produced phantom rules like a bare id as its own
                # zero-course "requirement", which then stole the
                # following real header line meant for the next real
                # item -- corrupting every item after it in the section)
                # and NOT as a note (nothing useful to say about it beyond
                # the warning already emitted where it was actually
                # dropped from a course list, if it was ever in one).
                self.advance()
                continue
            # Not a structural line yet -- treat as label/notes and keep scanning.
            self.advance()
            if label is None:
                label = line
            else:
                self.pending_notes.append(line)
        # Ran out of lines without ever finding a header/wrapper -- this
        # whole item attempt was just prose (e.g. a free-text requirement
        # with no enumerated course list). Keep `label` (the first line)
        # in pending_notes too, instead of discarding it, so the caller
        # (parse_program_text) has the FULL prose available to build a
        # placeholder requirement from -- see _make_placeholder_rule_from_prose.
        if label is not None:
            self.pending_notes.insert(0, label)
        return None

    def _parse_wrapped_list(self, combine: str) -> list[RequirementRule]:
        """Parse named sub-items joined by AND (if combine=='all') or OR
        (if combine=='at_least_n'), stopping when the expected connector
        no longer appears."""
        sub_rules = []
        expect_connector = "AND" if combine == "all" else "OR"
        while True:
            item = self.parse_one_item()
            if item is None:
                break
            sub_rules.append(item)
            if self.peek() is not None and CONNECTOR_RE.match(self.peek()):
                conn = self.peek()
                if conn != expect_connector:
                    self.warn(f"expected {expect_connector} connector, saw {conn} after {item.label!r}")
                self.advance()
                continue
            break
        return sub_rules

    def _parse_leaf(self, label: str | None) -> RequirementRule:
        header_line = self.advance()
        kind, n, unit_min = self._interpret_header(header_line)
        courses = self._parse_course_slots()
        if not courses:
            self.warn(f"leaf {label!r} had header but no parsed course slots (line: {header_line!r})")
        # Swallow trailing prose (e.g. "MATH 19, MATH 20, and MATH 21, or AP
        # Calculus Credit may be used...") up to the next structural marker.
        # Without this, that prose is left sitting in front of an "AND"/"OR"
        # connector or the next item's label, and whoever reads next
        # (_parse_wrapped_list looking for a connector, or the section loop
        # looking for the next item's label) misreads it.
        _STRUCTURAL_KINDS = (
            "wrapper_all", "wrapper_any", "wrapper_at_least_n",
            "header_all", "header_at_least", "header_any", "header_earn_credits",
            "header_count_credits",
        )
        while (
            self.pos < self.n
            and _classify(self.peek()) == "text"
            and not _looks_like_a_label(self.peek())
            # Even if this line reads like prose (e.g. ends with a period),
            # don't swallow it if the VERY NEXT line is itself a header/
            # wrapper -- that's strong evidence this line is actually the
            # next item's label (labels are sometimes written as full
            # sentences, e.g. "Thesis Requirement. Completion of an honors
            # thesis is required."), not trailing prose for this one.
            and not (
                self.pos + 1 < self.n
                and _classify(self.lines[self.pos + 1]) in _STRUCTURAL_KINDS
            )
        ):
            self.pending_notes.append(self.advance())
        return RequirementRule(
            label=label or "",
            kind=kind,
            courses=courses,
            n=n,
            unit_min=unit_min,
        )

    def _interpret_header(self, header_line: str) -> tuple[str, int | None, float | None]:
        if HEADER_ALL_RE.match(header_line):
            return "all", None, None
        m = HEADER_AT_LEAST_RE.match(header_line)
        if m:
            return "at_least_n", int(m.group(1)), None
        if HEADER_ANY_RE.match(header_line):
            return "at_least_n", 1, None
        m = HEADER_EARN_CREDITS_RE.match(header_line)
        if m:
            return "unit_min", None, float(m.group(1))
        m = HEADER_COUNT_AND_CREDITS_RE.match(header_line)
        if m:
            return "at_least_n", int(m.group(1) or m.group(2)), None
        self.warn(f"unrecognized header treated as at_least_n=1: {header_line!r}")
        return "at_least_n", 1, None

    def _parse_course_slots(self) -> list[object]:
        """Consume consecutive course/OR-alt/AND-alt lines into slots.
        A slot is a single course key, a tuple of OR-alternative keys, or
        (rare) a tuple-of-tuples if an OR-alternative is itself an
        AND-compound. Stops at the first non-course-list line."""
        slots: list[object] = []
        current_alts: list[object] = []

        def close_slot():
            if not current_alts:
                return
            slots.append(current_alts[0] if len(current_alts) == 1 else tuple(current_alts))
            current_alts.clear()

        while self.pos < self.n:
            line = self.peek()
            kind = _classify(line)
            if kind == "course":
                close_slot()
                self.advance()
                m = COURSE_LINE_RE.match(line)
                current_alts.append(_to_key(m.group(1)))
            elif kind == "or_alt":
                self.advance()
                m = OR_ALT_RE.match(line)
                current_alts.append(_to_key(m.group(1)))
            elif kind == "and_alt":
                self.advance()
                m = AND_ALT_RE.match(line)
                # Compound alternative: merge with the most recently added
                # alt into an AND-group (tuple with a leading "__and__"
                # marker so the solver layer can distinguish it later).
                if current_alts:
                    prev = current_alts.pop()
                    prev_group = prev if isinstance(prev, tuple) and prev and prev[0] == "__and__" else ("__and__", prev)
                    current_alts.append(prev_group + (_to_key(m.group(1)),))
                else:
                    current_alts.append(("__and__", _to_key(m.group(1))))
            elif kind == "course_set":
                close_slot()
                self.advance()
                current_alts.append(self._placeholder_for_course_set(COURSE_SET_RE.match(line), line))
            elif kind == "or_course_set":
                self.advance()
                current_alts.append(self._placeholder_for_course_set(OR_COURSE_SET_RE.match(line), line))
            elif kind == "unresolved_id":
                # A dead/unresolvable course reference sitting mid-list --
                # close out whatever slot was in progress (matching the
                # "course" branch's behavior) and just skip it, rather than
                # treating it as "list over" (see _BARE_ID_RE's docstring).
                # No slot gets added for it -- one fewer option among the
                # rest, not a placeholder standing in for nothing.
                close_slot()
                self.advance()
                self.warn(f"skipped unresolved course reference (bulletin id {line!r}) mid-list -- excluded from this requirement's options")
            elif kind == "connector" and self.pos + 1 < self.n and COURSE_LINE_RE.match(self.lines[self.pos + 1]):
                # A bare "AND"/"OR" line with its alternative's course on
                # its OWN next line, rather than "OR CODE - Title" all on
                # one line the way OR_ALT_RE/AND_ALT_RE expect -- a DOM
                # line-splitting artifact on some rendered bulletin pages,
                # the same class of thing html_to_lines() already rejoins
                # for split header phrases. Confirmed happening for real on
                # CS-BS's own "CS-BS General Electives" list ("CS157 -
                # Computational Logic" / "OR" / "PHIL151 - Metalogic" as
                # three separate lines): left unhandled, this fell through
                # to the plain `else: break` below, which ended the course
                # list right there -- silently dropping every real course
                # after it (~80 of CS-BS's ~100 real general electives) and
                # leaving "PHIL151 - Metalogic" to be misread as a brand
                # new item's LABEL one level up, which then went on to
                # swallow real course lines as if they were prose notes
                # until it stumbled onto a real header somewhere much later
                # in the document and misattributed THAT list to itself.
                # Once the line-split is detected, treat it exactly like
                # the single-line "or_alt"/"and_alt" forms already do.
                conn = line
                self.advance()  # the bare "AND"/"OR" line
                course_line = self.advance()
                key = _to_key(COURSE_LINE_RE.match(course_line).group(1))
                if conn == "AND":
                    if current_alts:
                        prev = current_alts.pop()
                        prev_group = prev if isinstance(prev, tuple) and prev and prev[0] == "__and__" else ("__and__", prev)
                        current_alts.append(prev_group + (key,))
                    else:
                        current_alts.append(("__and__", key))
                else:
                    current_alts.append(key)
            else:
                break
        close_slot()
        return slots

    def _placeholder_for_course_set(self, m: "re.Match", raw_line: str) -> str:
        """A course-list slot naming a whole unenumerated pool (e.g. "MATH
        Courses Numbered 100 and Above") instead of a specific course.
        Registers (deduplicated by pool) and returns a placeholder course
        key -- see models.make_placeholder_course. Warns once per distinct
        pool so this is never a silent substitution."""
        subject, num, direction = m.group(1), m.group(2), (m.group(3) or "").lower()
        if "above" in direction or "up" in direction:
            suffix, phrase = "PLUS", "and above"
        elif "below" in direction:
            suffix, phrase = "MINUS", "and below"
        else:
            suffix, phrase = "", ""
        key = f"{PLACEHOLDER_PREFIX}{_to_key(subject)}{num}{suffix}"
        description = f"Any {subject.upper()} course numbered {num}{(' ' + phrase) if phrase else ''}"
        is_new = key not in self.placeholders
        self.placeholders[key] = description
        if is_new:
            self.warn(
                f"course slot names an unenumerated pool ({raw_line!r}), not a specific course -- "
                f"created placeholder {key!r}; resolve with a real course before trusting the schedule"
            )
        return key


def _make_placeholder_rule_from_prose(
    section_title: str,
    notes: list[str],
    code: str,
    warnings: list[ParseWarning],
    placeholders: dict[str, str],
) -> RequirementRule | None:
    """A section produced zero rules -- every line in it was prose, no
    "Complete ALL/ANY/at least N of the following Courses:" list ever
    appeared (e.g. MATH-MIN's "six math courses at or above Math 51").
    Rather than silently leaving this requirement absent (the previous,
    buggy behavior), create N placeholder courses standing in for it --
    N taken from a leading count in the prose if one is found, else 1
    with an explicit warning that the count also needs manual review."""
    if not notes:
        return None
    prose = " ".join(notes)
    m = _LEAD_COUNT_RE.search(prose)
    if m:
        word = m.group(1).lower()
        n = int(word) if word.isdigit() else _NUMBER_WORDS_EXT.get(word, 1)
    else:
        n = 1
        warnings.append(ParseWarning(
            section_title,
            f"free-text requirement has no extractable course count either -- defaulted to "
            f"1 placeholder; verify the real count manually. Text: {prose[:200]!r}",
        ))
    # Prefer whichever note line actually contains the extracted count --
    # much more informative than notes[0], which is often just a generic
    # sub-heading like "Minor Requirements" rather than the real sentence
    # ("The Math minor consists of six math courses at or above Math 51.").
    description = None
    if m:
        for note in notes:
            if _LEAD_COUNT_RE.search(note):
                description = note.rstrip(".")
                break
    if not description:
        description = notes[0].rstrip(".") or prose[:80]
    slug = _slugify(f"{code}_{section_title}_{description}")
    courses: list[object] = []
    for i in range(1, n + 1):
        key = f"{PLACEHOLDER_PREFIX}{slug}" + (f"_{i}" if n > 1 else "")
        placeholders[key] = description if n == 1 else f"{description} (course {i} of {n})"
        courses.append(key)
    warnings.append(ParseWarning(
        section_title,
        f"free-text requirement with no enumerated course list -- created {n} placeholder "
        f"course(s) for {description!r}; resolve with real courses before trusting the schedule",
    ))
    return RequirementRule(label=description[:80], kind="all", courses=courses,
                            source=f"{code} program page: {section_title} (placeholder, free text)")


def parse_program_text(text: str, code: str) -> ProgramRequirements:
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l]  # drop blanks -- blank-line structure isn't load-bearing here

    # Title: first line, usually "CODE - Title"
    title = lines[0] if lines else code
    if " - " in title:
        title = title.split(" - ", 1)[1]

    units_min = None
    university_units_min = None
    for i, l in enumerate(lines):
        if l == "Minimum Units in the Program" and i + 1 < len(lines):
            try:
                units_min = float(lines[i + 1])
            except ValueError:
                pass
        if l == "Minimum University Units" and i + 1 < len(lines):
            try:
                university_units_min = float(lines[i + 1])
            except ValueError:
                pass

    # Find section boundaries: a line immediately followed by "Completion requirement".
    section_starts = [i for i in range(len(lines) - 1) if lines[i + 1] == SECTION_MARKER]
    warnings: list[ParseWarning] = []
    sections: list[ProgramSection] = []
    placeholders: dict[str, str] = {}

    if not section_starts:
        warnings.append(ParseWarning("document", "no 'Completion requirement' section markers found at all"))

    for idx, start in enumerate(section_starts):
        title_line = lines[start]
        body_start = start + 2  # skip title line + "Completion requirement"
        body_end = section_starts[idx + 1] if idx + 1 < len(section_starts) else len(lines)
        body_lines = lines[body_start:body_end]
        parser = _Parser(body_lines, warn_ctx=title_line, warnings=warnings, placeholders=placeholders, code=code)
        # parse_section_body() now creates a placeholder for any top-level
        # item that turns out to be pure prose -- whether that's the WHOLE
        # section (previously the only case handled here) or just one item
        # among several that otherwise parsed fine (see its docstring).
        rules, notes = parser.parse_section_body()
        sections.append(ProgramSection(title=title_line, rules=rules, notes=notes))

    return ProgramRequirements(
        code=code,
        title=title,
        units_min=units_min,
        university_units_min=university_units_min,
        sections=sections,
        warnings=warnings,
        placeholder_courses=placeholders,
    )


def format_tree(rule: RequirementRule, indent: int = 0) -> str:
    pad = "  " * indent
    head = f"{pad}[{rule.kind}"
    if rule.n is not None:
        head += f" n={rule.n}"
    if rule.unit_min is not None:
        head += f" unit_min={rule.unit_min}"
    head += f"] {rule.label}"
    out = [head]
    for c in rule.courses:
        out.append(f"{pad}  - {c}")
    for sr in rule.sub_rules:
        out.append(format_tree(sr, indent + 1))
    return "\n".join(out)
