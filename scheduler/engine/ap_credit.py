"""
AP credit -> specific Stanford course exemptions, built directly from
Stanford's own "External Test Credit Equivalency Charts" page
(studentservices.stanford.edu/my-academics/earn-my-degree/undergraduate-
degree-progress/test-transfer-credit/external-test-0). This environment's
web fetch is blocked by org policy, so the live page couldn't be pulled
directly -- the user supplied an HTML export of the page instead, and the
tables below are hand-transcribed from that export's "Chemistry",
"Computer Science", "Economics", "Mathematics", and "Physics" equivalency
tables. Treat this as a snapshot (dated by this comment, not auto-synced)
to confirm against the live chart, not a permanent source of truth --
Stanford revises it periodically.

Scope, deliberately narrow (same "don't guess, flag it" philosophy as the
rest of this project):
  - Only AP (College Board) exam rows are modeled. The same chart also has
    IBACC/GCEUK/GCEEU/CAPE/GMABT/IRELC/AUSHC/NEWL rows for the same
    subjects, but engine/transcript.py's AP_CREDIT_RE only ever extracts
    lines under the transcript's "Advanced Placement Test Credit" header,
    so a transcript can never hand this module anything but an AP exam
    name -- the other exam boards are out of scope until/unless this tool
    parses a transcript that can actually produce them.
  - Only the subjects whose "Courses Satisfied" column names concrete
    Stanford course codes directly are modeled here: Chemistry, Calculus,
    Physics, Computer Science, Economics. The chart's Language/Latin table
    uses a generic "(LANG*) 1A/2A or 1/2/3" placeholder that requires
    knowing each specific language's *LANG subject-code prefix (e.g. is it
    "JAPANLANG" or "JAPANLNG"? this codebase doesn't have a reliable
    per-language lookup) to resolve into real course keys -- left unmapped
    rather than guessed at.
  - Where the chart lists TWO rows for the same exam with the SAME "Unit
    Award" (the number that actually lands on a transcript) -- e.g. AP
    Physics C Mechanics shows 4 units awarded whether the score was a 4 or
    a 5 -- a transcript's units_attempted value alone can't tell you which
    row applies. Both rows are returned as candidates in that case; the
    caller (app.py) is expected to ask the user which one is theirs rather
    than silently picking one.
  - "Courses Satisfied" cells written as "X/Y" (no "or") are read as an
    AND -- both courses satisfied together -- cross-checked against a
    program bulletin's own words for the AP-Calc-BC-score-5 case
    ("MATH 19, MATH 20, and MATH 21, or AP Calculus Credit may be used",
    from CS-BS's own program page) and against a real transcript (a
    student with 10.00 units of "Advanced Placement Mathematics: Calculus
    BC" -- the chart's score-5/10-unit row -- who did NOT separately take
    MATH19/20/21 as real courses). Cells written as "X or Y" (chart's own
    "or") are read as real alternatives -- only one needs to be satisfied
    -- matching how CHEM-BS's own hand-transcribed rule tree already
    represents "CHEM31A AND CHEM31B, OR CHEM31E" as one pool.
  - The score-5 Physics C rows additionally include the paired lab course
    (PHYSICS42 alongside PHYSICS41; PHYSICS44 alongside PHYSICS43) per
    direct user confirmation, NOT because the scraped chart table lists
    it -- Stanford's own "Courses Satisfied" column only names PHYSICS41/
    PHYSICS43 for these rows. Flagged with its own `note` on those two
    rows so this one deviates visibly from "straight off the chart."
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class APEquivalencyRow:
    exam_label: str
    units: float
    # Course items in the same shape RequirementRule.courses already uses:
    # a plain course-key string (independently satisfied), or an
    # ("__and__", ...) tuple (all members satisfied together as one unit),
    # mixed freely in the same list when the chart itself mixes AND/OR
    # (e.g. Chemistry's "31A/31B or 31E").
    satisfies: list
    note: str = ""


def flatten_course_items(items) -> set[str]:
    """Every course key referenced by a `satisfies` list, regardless of
    AND/OR structure -- safe to add to `completed_keys` wholesale, since
    marking extra course keys "done" can only help other pool-shaped rules
    resolve (nothing in this codebase requires a course to be *absent* from
    completed_keys), the same reasoning app.py's free-text AP substitution
    field already relied on before this module existed."""
    keys: set[str] = set()
    for item in items:
        if isinstance(item, tuple) and item and item[0] == "__and__":
            keys.update(c for c in item[1:] if isinstance(c, str))
        elif isinstance(item, (tuple, list)):
            keys.update(c for c in item if isinstance(c, str))
        elif isinstance(item, str):
            keys.add(item)
    return keys


# Source: studentservices.stanford.edu's External Test Credit Equivalency
# Charts, AP-board rows only (see module docstring). Order matters: for
# Calculus, the plain "AB" pattern uses a negative lookahead so an "AB
# Subscore" title falls through to its own, more specific pattern instead.
_AP_CHART: list[tuple[re.Pattern, list[APEquivalencyRow]]] = [
    (re.compile(r"chemistry", re.I), [
        APEquivalencyRow(
            "AP Chemistry, score 5", 10.0,
            [("__and__", "CHEM31A", "CHEM31B"), "CHEM31E"],
            "Only a score of 5 earns any credit on this exam. Satisfies the "
            "'CHEM31A + CHEM31B, or CHEM31E' pool either way (you don't need "
            "to actually take whichever half you're not using).",
        ),
    ]),
    (re.compile(r"calculus\s*bc", re.I), [
        APEquivalencyRow("AP Calculus BC, score 4", 6.0, ["MATH19", "MATH20"]),
        APEquivalencyRow("AP Calculus BC, score 5", 10.0, ["MATH19", "MATH20", "MATH21"]),
    ]),
    (re.compile(r"calculus\s*ab\s*subscore", re.I), [
        APEquivalencyRow("AP Calculus AB Subscore, score 5", 6.0, ["MATH19", "MATH20"]),
    ]),
    (re.compile(r"calculus\s*ab(?!\s*subscore)", re.I), [
        APEquivalencyRow("AP Calculus AB, score 5", 6.0, ["MATH19", "MATH20"]),
    ]),
    (re.compile(r"physics\s*c.*mechan|mechan.*physics\s*c", re.I), [
        APEquivalencyRow("AP Physics C Mechanics, score 4", 4.0, ["PHYSICS21"]),
        APEquivalencyRow(
            "AP Physics C Mechanics, score 5", 4.0,
            ["PHYSICS21", "PHYSICS41", "PHYSICS42"],
            "PHYSICS42 (the paired lab) included per user confirmation -- "
            "Stanford's own chart table only lists PHYSICS41 for this row, "
            "not the lab that normally follows it.",
        ),
    ]),
    (re.compile(r"physics\s*c.*(elect|e\s*&\s*m|magn)", re.I), [
        APEquivalencyRow("AP Physics C E&M, score 4", 4.0, ["PHYSICS23"]),
        APEquivalencyRow(
            "AP Physics C E&M, score 5", 4.0,
            ["PHYSICS23", "PHYSICS43", "PHYSICS44"],
            "PHYSICS44 (the paired lab) included per user confirmation -- "
            "Stanford's own chart table only lists PHYSICS43 for this row, "
            "not the lab that normally follows it.",
        ),
    ]),
    (re.compile(r"physics\s*1\s*(&|and)\s*2", re.I), [
        APEquivalencyRow("AP Physics 1&2 combined, score 8", 4.0, ["PHYSICS21"]),
        APEquivalencyRow("AP Physics 1&2 combined, score 9 or 10", 8.0, ["PHYSICS21", "PHYSICS23"]),
    ]),
    (re.compile(r"computer science a\b", re.I), [
        APEquivalencyRow("AP Computer Science A, score 4+", 5.0, ["CS105", "CS106A"]),
    ]),
    (re.compile(r"computer science principles|cs principles", re.I), [
        APEquivalencyRow("AP CS Principles, score 4+", 5.0, ["CS105"]),
    ]),
    (re.compile(r"micro\s*economics", re.I), [
        APEquivalencyRow(
            "AP Micro + Macro Economics, both score 5", 0.0, ["ECON1"],
            "Only counts if you also have a qualifying AP Macroeconomics "
            "score -- Stanford's chart requires both exams together for this "
            "one, and awards 0 transcript units for it either way.",
        ),
    ]),
    (re.compile(r"macro\s*economics", re.I), [
        APEquivalencyRow(
            "AP Micro + Macro Economics, both score 5", 0.0, ["ECON1"],
            "Only counts if you also have a qualifying AP Microeconomics "
            "score -- Stanford's chart requires both exams together for this "
            "one, and awards 0 transcript units for it either way.",
        ),
    ]),
]


# ---------------------------------------------------------------------------
# Full chart, every exam board, for MANUAL entry (app.py section 1b).
#
# Transcribed from the live page (read in Chrome, Nov 2026): "Examinations
# currently accepted by Stanford: Effective for Undergraduates Matriculating
# 2026-27". Rows are in the page's own order. `courses_text` is the
# page's "Courses Satisfied" cell verbatim; `satisfies` is the same thing as
# course keys ("X/Y" = both, "X or Y" = either -- same reading as above).
# One deliberate deviation, same as the AP-only table above: the AP Physics C
# score-5 rows also mark the paired lab (PHYSICS42 / PHYSICS44) done, per
# earlier user confirmation. Language rows satisfy no specific course (the
# page's "(LANG*) 1A/2A or 1/2/3" depends on which language) -- they're
# flagged `fulfills_language` instead.
# ---------------------------------------------------------------------------

EXTERNAL_TEST_CHART_YEAR = "2026-27"
EXTERNAL_TEST_CHART_URL = (
    "https://studentservices.stanford.edu/my-academics/earn-my-degree/"
    "undergraduate-degree-progress/test-transfer-credit/external-test-0"
)

EXAM_BOARDS = {
    "AP": "Advanced Placement (College Board)",
    "AUSHC": "Australian Higher Certificate",
    "CAPE": "Caribbean Advanced Placement Exam",
    "GCEUK": "GCE/A-Levels - British model",
    "GCEEU": "Higher (H2) A-Levels - European/International model",
    "GMABT": "German Abitur",
    "IBACC": "International Baccalaureate",
    "IRELC": "Irish Leaving Certificate",
    "NEWL": "National Examination in World Languages",
}

# Footnotes from the page, keyed by chart section.
EXTERNAL_TEST_FOOTNOTES = {
    "Computer Science": "Credit may only be awarded for one AP computer science subject test; if both "
                        "scores are submitted, the higher-satisfying exam will be awarded.",
    "Economics": "Qualifying scores on both subject tests required for credit award.",
    "Mathematics": "Students may receive credit for only one subject test in a given exam; if multiple "
                   "scores are submitted, the higher-satisfying exam will be awarded.",
    "Physics": "Students may receive a maximum of 8 units for AP Physics; credit can be given for "
               "Physics 1/2 combined or one/both of Physics C.",
    "Language": "Subject tests satisfy specific courses in the indicated language. Scores of 4 on the "
                "starred (AP / NEWL) exams fulfill the university language requirement but earn no "
                "unit credit. Credit by exam that fulfills the language requirement does not "
                "constitute placement in further study; placement tests are still required.",
}
EXTERNAL_TEST_GENERAL_RULES = (
    "Max 45 quarter units of test + transfer credit toward the degree. You may not be awarded units "
    "for more than one exam in the same subject type (excluding language and physics). Stanford may "
    "remove test credit if later coursework duplicates it."
)


@dataclass
class ExternalTestRow:
    section: str          # chart section: Chemistry, Computer Science, ...
    board: str            # key of EXAM_BOARDS
    subject: str
    score: str
    units: float
    courses_text: str     # the page's "Courses Satisfied" cell
    satisfies: list
    fulfills_language: bool = False
    note: str = ""

    @property
    def row_id(self) -> str:
        return f"{self.board}|{self.subject}|{self.score}"

    @property
    def label(self) -> str:
        units = f"{self.units:g} units" if self.units else "0 units"
        return f"{self.subject}, score {self.score} -> {units}; satisfies {self.courses_text}"


def _row(section, board, subject, score, units, courses_text, satisfies, **kw) -> ExternalTestRow:
    return ExternalTestRow(section, board, subject, score, float(units), courses_text, satisfies, **kw)


_CHEM = [("__and__", "CHEM31A", "CHEM31B"), "CHEM31E"]
_CHEM_TXT = "CHEM 31A/31B or 31E"
_M1920, _M1920T = ["MATH19", "MATH20"], "MATH 19/20"
_M192021, _M192021T = ["MATH19", "MATH20", "MATH21"], "MATH 19/20/21"
_P2123, _P2123T = ["PHYSICS21", "PHYSICS23"], "PHYSICS 21/23"
_LANG_TXT = "(LANG) 1A/2A or 1/2/3"
_LATIN = ["CLASSICS1L", "CLASSICS2L", "CLASSICS3L"]
_LATIN_TXT = "CLASSICS 1L/2L/3L"
_SCORE4_NOTE = "Fulfills the university language requirement but earns no unit credit."


def _lang_rows(board, languages, score, units=10, score4=False, prefix=""):
    rows = []
    for lang in languages:
        rows.append(_row("Language", board, f"{prefix}{lang}", score, units, _LANG_TXT, [], fulfills_language=True))
        if score4:
            rows.append(_row("Language", board, f"{prefix}{lang}", "4", 0, "university language requirement only",
                             [], fulfills_language=True, note=_SCORE4_NOTE))
    return rows


EXTERNAL_TEST_CHART: list[ExternalTestRow] = [
    # Chemistry
    _row("Chemistry", "AP", "Chemistry", "5", 10, _CHEM_TXT, _CHEM),
    _row("Chemistry", "IBACC", "HL Chemistry", "5+", 10, _CHEM_TXT, _CHEM),
    _row("Chemistry", "GCEUK", "Chemistry", "A or B", 10, _CHEM_TXT, _CHEM),
    _row("Chemistry", "GCEEU", "Chemistry", "A or B", 10, _CHEM_TXT, _CHEM),
    _row("Chemistry", "CAPE", "Chem 1 & 2", "1-2", 10, _CHEM_TXT, _CHEM),
    # Computer Science
    _row("Computer Science", "AP", "Computer Science A", "4+", 5, "CS 105, 106A", ["CS105", "CS106A"],
         note=EXTERNAL_TEST_FOOTNOTES["Computer Science"]),
    _row("Computer Science", "AP", "CS Principles", "4+", 5, "CS 105", ["CS105"],
         note=EXTERNAL_TEST_FOOTNOTES["Computer Science"]),
    _row("Computer Science", "IBACC", "HL Computer Science", "5+", 5, "CS 105, 106A", ["CS105", "CS106A"]),
    _row("Computer Science", "GCEUK", "Computer Science", "A-D", 5, "CS 105, 106A", ["CS105", "CS106A"]),
    _row("Computer Science", "GCEEU", "Computer Science", "A", 5, "CS 105, 106A", ["CS105", "CS106A"]),
    _row("Computer Science", "IRELC", "Computing", "H1", 5, "CS 101, 105, 106A", ["CS101", "CS105", "CS106A"]),
    # Economics
    _row("Economics", "AP", "Microeconomics & Macroeconomics", "5 & 5", 0, "ECON 1", ["ECON1"],
         note=EXTERNAL_TEST_FOOTNOTES["Economics"]),
    _row("Economics", "IBACC", "HL Economics", "7", 0, "ECON 1", ["ECON1"]),
    # Mathematics
    _row("Mathematics", "AP", "Calculus AB", "5", 6, _M1920T, _M1920),
    _row("Mathematics", "AP", "Calculus AB Subscore", "5", 6, _M1920T, _M1920),
    _row("Mathematics", "AP", "Calculus BC", "4", 6, _M1920T, _M1920),
    _row("Mathematics", "AP", "Calculus BC", "5", 10, _M192021T, _M192021),
    _row("Mathematics", "IBACC", "HL Math Analysis/Approaches", "6+", 6, _M1920T, _M1920),
    _row("Mathematics", "GCEUK", "Mathematics", "A-B", 6, _M1920T, _M1920),
    _row("Mathematics", "GCEUK", "Further Math", "A", 6, _M1920T, _M1920),
    _row("Mathematics", "GCEEU", "Mathematics", "A-B", 6, _M1920T, _M1920),
    _row("Mathematics", "CAPE", "Mathematics", "1", 10, _M192021T, _M192021),
    _row("Mathematics", "GMABT", "Mathematics", "1-1.3", 10, _M192021T, _M192021),
    _row("Mathematics", "GMABT", "Mathematics", "1.7-2.3", 6, _M1920T, _M1920),
    _row("Mathematics", "IRELC", "Mathematics", "H1", 3, "MATH 19", ["MATH19"]),
    _row("Mathematics", "AUSHC", "Math Ext 1", "E4", 3, "MATH 19", ["MATH19"]),
    _row("Mathematics", "AUSHC", "Math Ext 1 & 2", "E4", 6, _M1920T, _M1920,
         note="Qualifying scores on both subject tests required for credit award."),
    # Physics
    _row("Physics", "AP", "Physics 1 & 2", "8 combined", 4, "PHYSICS 21", ["PHYSICS21"]),
    _row("Physics", "AP", "Physics 1 & 2", "9 or 10 combined", 8, _P2123T, _P2123),
    _row("Physics", "AP", "Physics C Mechanics", "4", 4, "PHYSICS 21", ["PHYSICS21"]),
    _row("Physics", "AP", "Physics C Mechanics", "5", 4, "PHYSICS 21/41", ["PHYSICS21", "PHYSICS41", "PHYSICS42"],
         note="PHYSICS42 (the paired lab) also marked done, per earlier user confirmation -- the chart lists 21/41."),
    _row("Physics", "AP", "Physics C Elect/Mag", "4", 4, "PHYSICS 23", ["PHYSICS23"]),
    _row("Physics", "AP", "Physics C Elect/Mag", "5", 4, "PHYSICS 23/43", ["PHYSICS23", "PHYSICS43", "PHYSICS44"],
         note="PHYSICS44 (the paired lab) also marked done, per earlier user confirmation -- the chart lists 23/43."),
    _row("Physics", "IBACC", "HL Physics", "5", 4, "PHYSICS 21", ["PHYSICS21"]),
    _row("Physics", "IBACC", "HL Physics", "6+", 8, _P2123T, _P2123),
    _row("Physics", "GCEUK", "Physics", "A-B", 8, _P2123T, _P2123),
    _row("Physics", "GCEEU", "Physics H2", "A-B", 8, _P2123T, _P2123),
    # Language
    *_lang_rows("AP", ["Chinese", "French", "German", "Italian", "Japanese", "Spanish"], "5", score4=True),
    _row("Language", "AP", "Latin (Lit. or Vergil)", "5", 10, _LATIN_TXT, _LATIN, fulfills_language=True),
    _row("Language", "AP", "Latin (Lit. or Vergil)", "4", 0, "university language requirement only", [],
         fulfills_language=True, note=_SCORE4_NOTE),
    *_lang_rows("IBACC", ["French", "German", "Italian", "Japanese", "Korean", "Chinese", "Spanish"], "5+",
                prefix="HL "),
    *_lang_rows("GCEUK", ["Arabic", "Chinese", "French", "German", "Hindi", "Italian", "Japanese",
                          "Portuguese", "Spanish", "Tamil", "Urdu"], "A-B"),
    _row("Language", "GCEUK", "Latin", "A-B", 10, _LATIN_TXT, _LATIN, fulfills_language=True),
    *_lang_rows("NEWL", ["Arabic", "Korean", "Portuguese", "Russian"], "5", score4=True),
    *_lang_rows("IRELC", ["Chinese", "French", "German", "Irish", "Italian", "Lithuanian", "Polish",
                          "Portuguese", "Spanish"], "H1"),
]

EXTERNAL_TEST_BY_ID = {r.row_id: r for r in EXTERNAL_TEST_CHART}


def guess_ap_equivalencies(exam_title: str) -> list[APEquivalencyRow]:
    """All chart rows whose subject pattern matches this AP exam's title
    (as it appears in the "Advanced Placement Test Credit" block of a
    parsed transcript, e.g. "Advanced Placement Physics C - Mechanics").
    May be empty (subject not modeled -- see module docstring), have one
    entry (unambiguous), or have more than one (same transcript unit award
    covers more than one possible score; let the caller ask which applies)."""
    for pattern, rows in _AP_CHART:
        if pattern.search(exam_title):
            return rows
    return []
