"""
Parse a Stanford unofficial undergraduate transcript PDF into a generic,
student-agnostic list of completed courses.

Nobody's transcript is hardcoded here -- this parses whatever PDF path you
give it. Stanford's transcript PDF lays terms out in a 2-column grid (e.g.
Autumn in the left column, Winter in the right column, side by side), which
breaks naive top-to-bottom text extraction. This parser splits each page at
the horizontal midpoint, reconstructs each column's lines independently by
grouping words with similar y-position, then runs a small line-based state
machine down each column to pull out course rows.

Format quirks this handles:
  - The Attempted/Earned/Grade numbers always sit on the *first* line of a
    course entry, even when the course title wraps onto following lines
    (the wrapped title lines visually render below/after the numbers).
  - In-progress courses (current quarter) show Attempted/Earned but no
    grade letter yet.
  - After the numbers, zero or more ALL-CAPS lines are wrapped title text;
    the first line containing a lowercase letter is the instructor name
    (or a "Previous Grade(s): ..." note) and ends the entry.

This is a first-pass, best-effort parser (v1). Real transcripts vary in
formatting across catalog years and situations (e.g. transfer credit,
withdrawals); anything it gets wrong should be fixable by hand in the app's
review step before a schedule is generated -- don't trust this blindly for
edge cases you haven't seen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pdfplumber

TERM_RE = re.compile(r"^(\d{4})-(\d{4})\s+(Autumn|Winter|Spring|Summer)$")
COURSE_START_RE = re.compile(r"^([A-Z][A-Z&]*)\s+(\d+[A-Z]*)\s+([A-Z]{2,5})\s+(.*)$")
TRAILING_UNITS_RE = re.compile(
    r"^(.*?)\s+(\d+\.\d{2})\s+(\d+\.\d{2})(?:\s+([A-Za-z+\-]{1,3}))?\s*$"
)
# "Advanced Placement Chemistry                 10.00" -- a single trailing
# units number (no attempted/earned pair, no grade). These live inside the
# "--------- Advanced Placement Test Credit ---------" block and otherwise
# look enough like a stop-line ("Advanced Placement" prefix) that they'd be
# silently skipped; matched explicitly (and checked before the stop-prefix
# check below) so this credit isn't lost from the unit total.
AP_CREDIT_RE = re.compile(r"^(Advanced Placement .+?)\s+(\d+\.\d{2})$")
STOP_PREFIXES = (
    "UG Term GPA", "UG Cum GPA", "Term Totals", "Cum Totals",
    "Undergraduate Career Totals", "UNITS COMPLETED", "STANFORD TOTAL",
    "Max Allowed", "UG Activity Units", "UG Pass/Fail", "Advanced Placement",
    "Total Quarter Units", "Course Cmpt Title", "*", "END OF TRANSCRIPT",
    "Information must be kept", "Page ", "Allowable Test Credit",
    "Applied Toward", "---------",
)


@dataclass
class CompletedCourse:
    subject: str
    code: str
    component: str
    title: str
    units_attempted: float
    units_earned: float
    grade: Optional[str]
    term: str  # e.g. "2025-2026 Autumn"

    @property
    def key(self) -> str:
        """Normalized course key, e.g. 'CS106A', matching explorecourses-style codes."""
        return f"{self.subject}{self.code}".upper().replace(" ", "")

    @property
    def in_progress(self) -> bool:
        return self.grade is None


@dataclass
class APCredit:
    """One line from the transcript's 'Advanced Placement Test Credit' block,
    e.g. 'Advanced Placement Chemistry', 10.00 units.

    This is NOT resolved to a specific course key -- Stanford's AP-to-course
    equivalency varies by exam score and department, and the raw transcript
    line alone doesn't say which course (if any) it substitutes for. It's
    surfaced separately so its units can be counted toward the informational
    180-unit graduation check without the parser guessing a course match.
    If AP credit actually exempts you from a specific requirement, note that
    yourself in the review step -- this tool won't infer it for you.
    """
    title: str
    units: float


def _cluster_rows(words: list[dict], y_tol: float = 3.0) -> list[list[dict]]:
    """Group words into rows by similar vertical position, sorted top-to-bottom,
    each row sorted left-to-right."""
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["top"]):
        placed = False
        for row in rows:
            if abs(row[0]["top"] - w["top"]) <= y_tol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    rows.sort(key=lambda row: row[0]["top"])
    for row in rows:
        row.sort(key=lambda w: w["x0"])
    return rows


def _rows_to_lines(rows: list[list[dict]]) -> list[str]:
    return [" ".join(w["text"] for w in row) for row in rows]


def _is_stop_line(line: str) -> bool:
    return any(line.startswith(p) for p in STOP_PREFIXES)


def _parse_lines(lines: list[str]) -> tuple[list[CompletedCourse], list[APCredit]]:
    completed: list[CompletedCourse] = []
    ap_credits: list[APCredit] = []
    current_term: Optional[str] = None
    pending: Optional[dict] = None

    def flush():
        nonlocal pending
        if pending is not None and pending.get("units_attempted") is not None:
            completed.append(CompletedCourse(
                subject=pending["subject"],
                code=pending["code"],
                component=pending["component"],
                title=" ".join(pending["title_parts"]).strip(" ,:"),
                units_attempted=pending["units_attempted"],
                units_earned=pending["units_earned"],
                grade=pending["grade"],
                term=pending["term"],
            ))
        pending = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        m = TERM_RE.match(line)
        if m:
            flush()
            current_term = line
            continue

        # Checked before the stop-prefix test below: AP credit lines also
        # start with "Advanced Placement" (a stop prefix, since most of that
        # block is boilerplate), so they'd otherwise be silently dropped.
        ap_m = AP_CREDIT_RE.match(line)
        if ap_m:
            flush()
            ap_title, ap_units = ap_m.groups()
            ap_credits.append(APCredit(title=ap_title.strip(), units=float(ap_units)))
            continue

        if _is_stop_line(line):
            flush()
            continue

        m = COURSE_START_RE.match(line)
        if m and current_term:
            flush()
            subject, code, component, rest = m.groups()
            um = TRAILING_UNITS_RE.match(rest)
            if um:
                title_part, attempted, earned, grade = um.groups()
                pending = {
                    "subject": subject, "code": code, "component": component,
                    "title_parts": [title_part],
                    "units_attempted": float(attempted),
                    "units_earned": float(earned),
                    "grade": grade,
                    "term": current_term,
                }
            else:
                pending = {
                    "subject": subject, "code": code, "component": component,
                    "title_parts": [rest],
                    "units_attempted": None, "units_earned": None, "grade": None,
                    "term": current_term,
                }
            continue

        if pending is not None:
            if pending["units_attempted"] is None:
                um = TRAILING_UNITS_RE.match(line)
                if um:
                    title_part, attempted, earned, grade = um.groups()
                    pending["title_parts"].append(title_part)
                    pending["units_attempted"] = float(attempted)
                    pending["units_earned"] = float(earned)
                    pending["grade"] = grade
                else:
                    pending["title_parts"].append(line)
                continue

            if line == line.upper():
                pending["title_parts"].append(line)
            else:
                flush()
            continue

    flush()
    return completed, ap_credits


def parse_transcript_pdf(path: str) -> tuple[list[CompletedCourse], list[APCredit]]:
    """Parse a Stanford unofficial transcript PDF (2-column term layout).

    Returns (completed_courses, ap_credits) in chronological (term) order.
    ap_credits is usually empty -- most transcripts have no AP Test Credit
    block -- and its units are informational only (see APCredit's docstring);
    they don't get folded into any CompletedCourse or count toward
    completed_keys."""
    all_completed: list[CompletedCourse] = []
    all_ap_credits: list[APCredit] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=False)
            mid_x = page.width / 2
            left = [w for w in words if w["x0"] < mid_x]
            right = [w for w in words if w["x0"] >= mid_x]
            for column_words in (left, right):
                rows = _cluster_rows(column_words)
                lines = _rows_to_lines(rows)
                page_completed, page_ap = _parse_lines(lines)
                all_completed.extend(page_completed)
                all_ap_credits.extend(page_ap)
    return all_completed, all_ap_credits


def summarize(completed: list[CompletedCourse], ap_credits: list[APCredit] | None = None) -> str:
    lines = []
    for c in completed:
        status = "IN PROGRESS" if c.in_progress else f"grade {c.grade}"
        lines.append(f"{c.key:10s} {c.title[:48]:48s} {c.units_attempted:5.2f}u  {status:14s} [{c.term}]")
    if ap_credits:
        lines.append("")
        lines.append("AP credit (informational units only, not attributed to any course):")
        for a in ap_credits:
            lines.append(f"  {a.title:48s} {a.units:5.2f}u")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    completed, ap_credits = parse_transcript_pdf(sys.argv[1])
    print(summarize(completed, ap_credits))
    print(f"\nTotal parsed rows: {len(completed)}")
    if ap_credits:
        print(f"Total AP credit units: {sum(a.units for a in ap_credits):.0f}")
