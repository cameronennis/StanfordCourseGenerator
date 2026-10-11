"""
Batch A hand-fixes: BIO-BS, CME-MS, ENVRES-MS.

(AFRAM-MIN, the fourth code assigned to this batch, is NOT included here --
see the note below the last function. Its generic bulletin_parser.py output
turned out to already be structurally correct; the real problem is a
course-data gap, not a rule-tree bug, so it needs a CourseOverride, not a
hand-written function.)

All three programs below share the same root cause: bulletin_parser.py's
_parse_course_slots() only recognizes "AND"/"OR" as continuing a course list
when they're glued to the next course on the SAME line (AND_ALT_RE / OR_ALT_RE
= r"^(AND|OR)\s+CODE - Title$"). Both BIO-BS's and ENVRES-MS's bulletin pages
render plenty of course alternatives as a bare "OR" (or "AND") on its OWN
line, between two "CODE - Title" lines. A standalone connector line matches
none of _parse_course_slots' recognized kinds, so the `else: break` branch
fires -- the parser thinks the course list is over. Every real course after
that point silently falls through to parse_one_item()'s generic label/notes
scanning instead, which (a) drops it from the rule it actually belongs to,
and (b) starts using bare "CODE - Title" lines themselves as bogus item
labels, shifting/corrupting every later label-header pairing in the same
block. Confirmed directly with `bulletin_parser.parse_program_text` +
`format_tree` on all three raw samples (see per-function notes for exact
before/after).

CME-MS has a second, independent bug: bulletin.stanford.edu renders its four
program "Tracks" (General CME / Data Science / Imaging Science / Math &
Comp Finance) as items WITHIN one "Depth in Discipline" `Completion
requirement` block, not as separate top-level sections. requirements.py's
is_subplan_section()/rules_from_parsed_program() split only operates on
`ProgramRequirements.sections` (i.e. at the granularity of a "Completion
requirement" marker) -- it has no way to see that "General CME Track" and
"Data Science Track" are meant to be alternative, pick-exactly-one children
of the SAME section, so both parsers today put every track's requirements
in one single "all" bucket, needing every track's courses simultaneously.
Same root issue independently affects ENVRES-MS's six "MS Course Pathway"
blocks (Climate Change / Food and Water Systems / ... / Sustainable Energy
Systems), which live inside one "Depth in Discipline" section the same way.

Per the project owner's direction, none of this is being fixed by
generalizing bulletin_parser.py (too much blast radius across 200+ other
programs) -- instead each affected program gets its OWN hand-transcribed
RequirementRule tree here, read directly off samples/{CODE}.txt.
"""
from __future__ import annotations

from .models import RequirementRule, make_placeholder_course  # noqa: F401  (placeholder helper referenced in docstrings)


def bio_bs_manual_rules() -> list[RequirementRule]:
    """Biology (BS) -- General Biology Subplan (samples/BIO-BS.txt).

    bulletin_parser.py's default build (no subplan selected) already scopes
    to "Core Program Requirements" + "Electives" + "Writing in the Major
    (WIM)" + "Capstone" -- correctly matching the bulletin's own note ('The
    "General Biology Subplan" requirements... are listed in the "Core
    Program Requirements" of this Bulletin'), since none of those four
    section titles match is_subplan_section()'s regex. The other 6
    "...Subplan" sections (Biochemistry and Biophysics, Cellular Molecular
    and Organismal, Computational and Systems, Ecology Evolution and
    Environment, Microbial Sciences, Neurosciences) are correctly excluded
    the same way CS-BS's non-default subplans are -- this hand-fix keeps
    that same General-Biology-Subplan scope rather than trying to encode
    all 7 tracks, consistent with existing project convention
    (PROGRAM_SHEET_BUILDERS only special-cases a track default when a
    program's *default* scope needs one; General Biology Subplan already
    doesn't).

    Bug found (see module docstring for the general mechanism): inside
    "Core Program Requirements" > "STEM Requirements", the raw text is

        Math Requirement
        Complete ANY of the following Courses:
        MATH19 - Calculus
        AND
        MATH20 - Calculus
        AND
        MATH21 - Calculus
        MATH51 - Linear Algebra, Multivariable Calculus, and Modern Applications
        CME100 - Vector Calculus for Engineers
        ...
        AND
        Physics Requirements
        ...

    i.e. the real Math Requirement is "(MATH19 AND MATH20 AND MATH21) OR
    MATH51 OR CME100" -- one alternative is itself a 3-course AND-sequence.
    The standalone "AND" right after "MATH19 - Calculus" stops
    _parse_course_slots() dead: the parsed rule ends up as just
    `at_least_n(n=1, courses=["MATH19"])`, and MATH20/MATH21/MATH51/CME100
    all get swallowed into parse_one_item()'s generic label/notes scan --
    "MATH20 - Calculus" (a real course line!) becomes the *label* of the
    next parsed rule, misattributing all of "Physics Requirements"'s real
    content under a bogus "MATH20 - Calculus" label three levels deep
    (confirmed via `format_tree` on the live parse: `[all] MATH20 -
    Calculus` wrapping the PHYSICS-series sub_rules). The same standalone-
    "OR" pattern breaks "Foundational Laboratory Requirement" (drops the
    BIO45/BIO44 half of two OR-pairs) and the "Electives" section's
    ~80-course out-of-department list (drops everything after
    "OSPAUSTL10 - Coral Reef Ecosystems", including PSYC124, STATS191,
    CHEM121, and ~20 more real options -- confirmed by diffing the parsed
    tree's course count against the raw text).

    Fix: every AND/OR-joined alternative below is hand-transcribed as an
    OR-tuple (or an AND-group tuple for the MATH19/20/21 sequence) directly
    in one leaf rule's `courses` list, instead of relying on
    _parse_course_slots()'s line-by-line scan.

    Also folds "General Biology Electives" (23 units, described in the raw
    text only as two "Untitled Course Set" placeholders -- the bulletin's
    own course-list widget didn't render real content there, a genuine
    scrape gap, not a parser bug) and the separately-rendered "Electives"
    section's ~80-course out-of-department list into ONE unit_min=23 pool,
    since the bulletin text says students may draw the 23 units from either
    ("Students may choose courses from the General Biology Subplan
    Electives and from the Approved Out-of-Department Elective Courses").
    A single small placeholder stands in for the undocumented in-department
    course-set half; the real, corrected 78-course out-of-department list
    (2 duplicates removed) does the rest of the work and is large enough
    (74 of 78 courses have real terms_offered, summing to ~250 potential
    units) that the placeholder is a safety net, not load-bearing.
    """
    out_of_dept_electives = [
        "BIOC241", "BMDS202", "BMDS221", "BMDS273", "BIOE101", "BIOE103",
        "BIOE214", "BIOE217", "BIOE220", "BIOE231", "BIOE279", "BIOE450",
        "BMDS173A", "BMDS210", "CBIO101", "CBIO240", "CEE161I", "CEE177",
        "CEE274D", "CHEM141", "CHEM143", "CHEM171", "CHEM183", "CHEM184",
        "CHEM185", "CHEM283", "CHEMENG110A", "CS106A", "CS106B", "CS107",
        "CS109", "DBIO210", "EARTHSYS114", "EARTHSYS123A", "EARTHSYS141",
        "EARTHSYS142", "EARTHSYS144", "EARTHSYS146B", "EARTHSYS151",
        "EARTHSYS152", "EARTHSYS155", "EARTHSYS158", "EARTHSYS240",
        "EPS139", "GENE202", "GENE211", "GENE235", "HUMBIO113", "HUMBIO130",
        "HUMBIO131", "HUMBIO135", "HUMBIO154C", "IMMUNOL201", "IMMUNOL202",
        "IMMUNOL205", "IMMUNOL206", "IMMUNOL209", "MCP156", "OCEANS173H",
        "OSPAUSTL10", "OSPAUSTL28", "OSPAUSTL32", "OSPSANTG85", "PSYC124",
        "PSYCH202", "PSYCH221", "STATS191", "STATS200", "STATS202",
        "SURG101", "SUSTAIN103", "EARTHSYS205A", "EPI231", "NBIO227",
        "OSPSANTG58", "PSYC152", "PSYC183", "PSYCH169", "CHEM121",
    ]
    return [
        RequirementRule(
            label="BioFoundations Requirement",
            kind="at_least_n",
            n=4,
            courses=["BIO81", "BIO82", "BIO83", "BIO84", "BIO85", "BIO86"],
            source="BIO-BS program sheet: Core Program Requirements > BioFoundations Requirement",
        ),
        RequirementRule(
            label="Foundational Laboratory Requirement",
            kind="all",
            courses=[("BIO43", "BIO45"), ("BIO47", "BIO44")],
            source=(
                "BIO-BS program sheet: Core Program Requirements > Foundational "
                "Laboratory Requirement -- 'Complete ALL: BIO43 OR BIO45, BIO47 OR "
                "BIO44'; hand-fixed from a parser bug where the standalone 'OR' "
                "lines dropped BIO45 and BIO44 entirely."
            ),
        ),
        RequirementRule(
            label="Math Requirement",
            kind="at_least_n",
            n=1,
            courses=[("__and__", "MATH19", "MATH20", "MATH21"), "MATH51", "CME100"],
            source=(
                "BIO-BS program sheet: STEM Requirements > Math Requirement -- "
                "'Complete ANY: (MATH19 AND MATH20 AND MATH21) OR MATH51 OR "
                "CME100'; hand-fixed from the standalone-AND parser bug described "
                "in this function's docstring."
            ),
        ),
        RequirementRule(
            label="Physics Requirement",
            kind="at_least_n",
            n=1,
            sub_rules=[
                RequirementRule(
                    label="PHYSICS 20 Series",
                    kind="all",
                    courses=["PHYSICS21", "PHYSICS22", "PHYSICS23", "PHYSICS24"],
                ),
                RequirementRule(
                    label="PHYSICS 40 Series",
                    kind="all",
                    courses=[
                        ("PHYSICS41", "PHYSICS41E", "PHYSICS61"),
                        ("PHYSICS43", "PHYSICS81"),
                        ("PHYSICS45", "PHYSICS44"),
                    ],
                ),
                RequirementRule(
                    label="PHYSICS 40/20 Series",
                    kind="all",
                    courses=[("PHYSICS41", "PHYSICS41E", "PHYSICS61"), "PHYSICS23", "PHYSICS24"],
                ),
            ],
            source="BIO-BS program sheet: STEM Requirements > Physics Requirements (one of 3 sequences)",
        ),
        RequirementRule(
            label="Chemistry Requirement",
            kind="all",
            courses=["CHEM31A", "CHEM31B", "CHEM33", "CHEM121"],
            source="BIO-BS program sheet: STEM Requirements > Chemistry Requirements",
        ),
        RequirementRule(
            label="Statistics Requirement",
            kind="at_least_n",
            n=1,
            courses=["BIO131", "BIO141", "STATS60", "CME106", "CS109", "OCEANS174H", "STATS110"],
            source="BIO-BS program sheet: STEM Requirements > Statistics Requirement",
        ),
        RequirementRule(
            label="General Biology Electives (23 units)",
            kind="unit_min",
            unit_min=23.0,
            courses=[*out_of_dept_electives, "PLACEHOLDER:bio_bs_general_biology_dept_electives"],
            source=(
                "BIO-BS program sheet: 'General Biology Electives' (23 units, from "
                "General Biology Subplan Electives or Approved Out-of-Department "
                "Elective Courses) + 'Electives' section's out-of-department list -- "
                "combined into one pool per the bulletin's own 'may choose from "
                "either' language. The in-department half ('BIO 80-series (limit "
                "1), lab courses (limit 1), 100-level Bio courses...') rendered as "
                "an empty 'Untitled Course Set' in the scrape -- a genuine content "
                "gap, not enumerated here -- represented by one placeholder. The "
                "out-of-department list is hand-fixed from the standalone-'AND' "
                "parser bug that dropped everything after 'OSPAUSTL10' (~20 real "
                "courses, including CHEM121 and STATS191)."
            ),
        ),
        RequirementRule(
            label="Writing in the Major (WIM)",
            kind="at_least_n",
            n=1,
            courses=["BIO47", "BIO127", "BIO133", "BIO168", "BIO199A", "BIO199W"],
            source="BIO-BS program sheet: Writing in the Major (WIM)",
        ),
        RequirementRule(
            label="Capstone Options (choose 1)",
            kind="at_least_n",
            n=1,
            sub_rules=[
                RequirementRule(
                    label="Capstone Option #1 - Honors",
                    kind="unit_min",
                    unit_min=13.0,
                    courses=["BIO199", "BIO199X", "BIO199W"],
                    source=(
                        "BIO-BS program sheet: Capstone Option #1- Honors -- '10 "
                        "units of BIO199 or BIO199X from your Honors lab... 3 units "
                        "of BIO199W. Fulfills WIM.'"
                    ),
                ),
                RequirementRule(
                    label="Capstone Option #2 - Independent Capstone",
                    kind="all",
                    courses=["BIO199A", "BIO199B"],
                    source="BIO-BS program sheet: Capstone Option #2- Independent Capstone",
                ),
            ],
            source=(
                "BIO-BS program sheet: 'Capstone Options -- All students must "
                "choose one of the following... options'. Only 2 of the raw text's "
                "4 named options are included: Option #3 (Notation in Science "
                "Communication) requires an off-list application/email-based "
                "approval process with no fixed course list, and Option #4 "
                "(Interdisciplinary Honors) names other Bulletin PROGRAMS "
                "(ARTS-IHN/STS-IHN/ED-IHN), not courses -- neither fits this "
                "engine's course-based RequirementRule model. Options #1 and #2 "
                "are both fully real, concrete, and independently schedulable, so "
                "the at_least_n=1 choice is still genuinely satisfiable."
            ),
        ),
    ]


def cme_ms_manual_rules() -> list[RequirementRule]:
    """Computational and Mathematical Engineering (MS) -- General CME Track
    (samples/CME-MS.txt).

    Bug found (see module docstring): CME-MS's bulletin page renders its
    four tracks (General CME / Data Science / Imaging Science / Mathematical
    and Computational Finance) as items inside ONE "Depth in Discipline"
    section, not as separate sections -- so requirements.py's subplan-
    exclusion logic (which operates on whole ProgramSections) can't isolate
    them, and the generic parse ANDs all four tracks' requirements together
    (needing e.g. the Data Science track's Requirement 1 in addition to the
    General CME track's Requirement 1, simultaneously). On top of that, the
    same standalone-AND/OR-line bug documented in bio_bs_manual_rules()
    corrupts several individual requirement lists within "Depth in
    Discipline" (confirmed via format_tree: e.g. "MS&E220 - Probabilistic
    Analysis" ends up as a bogus rule *label* wrapping "CME500" alone, and
    "Requirement 1: Mathematical and Statistical Foundations" for the Data
    Science track gets truncated to just CME298 before the OR-alternatives
    collapse into notes).

    Fix: hand-transcribe just the General CME Track (the program's default/
    general track, named first and requiring no specialized background,
    unlike the other three) -- same pattern as CS-BS's and DATSC-BS's
    existing single-subplan-default builders in requirements.py.

    Also excludes: all "MS&E..." (Management Science & Engineering) course
    references. Confirmed the MS&E department is entirely absent from
    data/courses_snapshot.json's fetched department list (not just a few
    missing courses) -- almost certainly because the raw bulletin HTML
    entity "MS&amp;E" was never decoded to "MS&E" by the fetch pipeline, so
    subjects_referenced_by_programs() never recognized "MS&E" as a subject
    needing a fetch. This is a genuine course-data/fetch-pipeline gap (out
    of scope for a per-program RequirementRule hand-fix -- see this batch's
    final report), not something to guess course data for here; General CME
    Track's Breadth Electives pool has ~40 other real options with MS&E's
    ~13 removed, so omitting them doesn't affect feasibility.

    "Requirement 4: Specialized Electives (9 units)" is genuinely
    unenumerated prose ("focused graduate application electives, approved
    by the program advisor... in engineering, mathematics, physical,
    biological, information, and other quantitative sciences") with no
    course list anywhere in the raw text -- represented with a placeholder.
    "Requirement 5: Seminar (3 units)" similarly says "2 units are up to
    the student's choice of ICME graduate seminars or other approved
    seminars" with no enumerated list; modeled as a repeatable unit_min
    pool of the two named-and-real ICME seminar courses (CME500, CME300)
    instead of inventing a placeholder for something two real courses
    already cover.
    """
    breadth_electives = [
        "BMDS222", "CEE260G", "CEE281", "CME218", "CME241", "CME262",
        "CME263", "CME279", "CME290", "CME291", "CME295", "CME309",
        "CME364A", "CS204", "CS221", "CS224N", "CS224W", "CS228", "CS229",
        "CS230", "CS231N", "CS233", "CS234", "CS236", "CS238", "CS239",
        "CS244B", "CS246", "CS255", "CS261", "ECON293", "EE223", "ENGR209A",
        "MATH136", "MATH171", "MATH235", "MATH236", "MATH238", "MATH275A",
        "ME335A", "ME408", "ME469",
    ]
    return [
        RequirementRule(
            label="Requirement 1: Foundational (12 units)",
            kind="unit_min",
            unit_min=12.0,
            courses=[
                "CME302", "CME303", "CME306",
                ("CME307", "CME364A"),
                "CME310",
                ("MATH236", "CME298", "CME308"),
            ],
            source=(
                "CME-MS program sheet: General CME Track > Requirement 1: "
                "Foundational -- 'completing four of the six core courses... "
                "Earn at least 12 credits from: CME302, CME303, CME306, (CME307 "
                "OR CME364A), CME310, (MATH236 OR CME298 OR CME308)'."
            ),
        ),
        RequirementRule(
            label="Requirement 2a: Basic Programming and Computing Algorithms (3 units)",
            kind="unit_min",
            unit_min=3.0,
            courses=["CME212", "CME310", "CS106B", "CS161", "CS166", "CS168", "CS190"],
            source="CME-MS program sheet: General CME Track > Requirement 2a",
        ),
        RequirementRule(
            label="Requirement 2b: Advanced Programming and Computing Algorithms (3 units)",
            kind="unit_min",
            unit_min=3.0,
            courses=["CME213", "CS149", "CS194", "CS246", "CS295", "CS315B"],
            source="CME-MS program sheet: General CME Track > Requirement 2b",
        ),
        RequirementRule(
            label="Requirement 3: Breadth Electives (15 units)",
            kind="unit_min",
            unit_min=15.0,
            courses=breadth_electives,
            source=(
                "CME-MS program sheet: General CME Track > Requirement 3: Breadth "
                "Electives -- MS&E courses excluded (see function docstring: whole "
                "department missing from the course snapshot)."
            ),
        ),
        RequirementRule(
            label="Requirement 4: Specialized Electives (9 units)",
            kind="unit_min",
            unit_min=9.0,
            courses=["PLACEHOLDER:cme_ms_general_specialized_electives"],
            source=(
                "CME-MS program sheet: General CME Track > Requirement 4: "
                "'Nine units of focused graduate application electives, approved "
                "by the program advisor... no enumerated list in the raw text."
            ),
        ),
        RequirementRule(
            label="Requirement 5: Seminar (3 units)",
            kind="unit_min",
            unit_min=3.0,
            courses=["CME500", "CME300"],
            source=(
                "CME-MS program sheet: General CME Track > Requirement 5: Seminar "
                "-- '1 seminar unit must come from CME500... 2 units are up to the "
                "student's choice of ICME graduate seminars'. CME500 (Departmental "
                "Seminar, 1 unit) and CME300 (First Year Seminar Series, 1 unit) "
                "are both real, repeatable-in-practice ICME seminar courses; "
                "modeled as a 3-unit pool over the two rather than inventing a "
                "placeholder for the unenumerated 'other approved seminars' half."
            ),
        ),
    ]


def envres_ms_manual_rules() -> list[RequirementRule]:
    """Environment and Resources (MS) (samples/ENVRES-MS.txt).

    Bug found (see module docstring): ENVRES-MS's six "MS Course Pathway"
    blocks (Climate Change / Food and Water Systems / Global Environmental
    Systems / Human and Planetary Health / Sustainable Built Environment /
    Sustainable Energy Systems) each say "Complete at least 4 of the
    following Courses:" and live as items inside ONE "Depth in Discipline"
    section -- meant as SIX ALTERNATIVES a student picks exactly one from
    ("A minimum of four letter-graded courses... from ONE course pathway"),
    but requirements.py's subplan-exclusion only operates at the
    ProgramSection level (see cme_ms_manual_rules()'s docstring for the
    identical structural issue), so the generic parse instead produces six
    INDEPENDENT top-level at_least_n=4 rules, ANDed together -- i.e.
    "complete 4 courses in EVERY pathway simultaneously" (24 courses total)
    instead of "complete 4 courses in ONE pathway you choose." Confirmed via
    format_tree: the parsed "Depth in Discipline" section has 5 separate
    `[at_least_n n=4]` top-level rules (the 6th pathway, Sustainable Energy
    Systems, didn't even get its own rule -- see below).

    On top of that, virtually every pathway's course list is riddled with
    the standalone-"OR"-line bug (see bio_bs_manual_rules()'s docstring for
    the mechanism) -- Stanford's cross-listing convention here is
    especially heavy ("CEE207A - Understand Energy / OR / CEE107A - ... /
    OR / EARTHSYS103 - ... / OR / ENERGY107A - ... / OR / ENERGY207A - ...",
    all five names for the SAME class), so each pathway's course list gets
    truncated down to just its first 1-4 entries. This is also what silently
    ate the 6th pathway entirely: "Sustainable Energy Systems"'s label and
    header got consumed as trailing "notes" off the corrupted 5th pathway's
    parse instead of starting a new rule.

    Fix: hand-transcribe ONE pathway (Climate Change, E-IPER's most central)
    in full, with real course keys and proper OR-groups for every
    cross-listed equivalent, and use it directly as the "Pathway Courses"
    requirement -- same "pick one, hand-encode that one" pattern as
    CS-BS's/DATSC-BS's single-subplan-default builders. (An at_least_n=1
    wrapper offering only this one sub_rule would be functionally identical
    to using it directly -- there's no ambiguity to preserve since the
    other 5 pathways aren't encoded here.)

    Also folds together the "Joint Master's Degree" and "Dual Master's
    Degree" top-level items: both raw-text blocks require the *exact* same
    E-IPER-side Core Courses / Human Environmental Decision-Making Course /
    Pathway Courses (only their OUTER professional-degree context --
    MBA/JD vs. another Stanford MA/MS/MD/PhD -- differs, which isn't
    something this engine's course-based rules model anyway). One shared
    tree covers both.

    Excludes "MS&E243"/"MS&E394" from the Human-Env-Decision and Climate
    Change course lists -- see cme_ms_manual_rules()'s docstring: the whole
    MS&E department is missing from the course snapshot (a fetch-pipeline
    gap, not something to fabricate data for). Also excludes the "Elective
    Courses" sub-block under each degree-path overview: it's pure prose
    about unit CAPS on already-counted coursework (guest-lecture-heavy
    courses, independent study, professional-school courses), not a new
    enumerable requirement, and this codebase doesn't enforce a total
    45-unit "Minimum Units in the Program" constraint for any program (see
    PHILO-MIN's manual fix in manual_fix_batchD.py for the same precedent).
    """
    return [
        RequirementRule(
            label="Core Courses",
            kind="all",
            courses=["ENVRES280", "ENVRES290"],
            source="ENVRES-MS program sheet: Core Courses",
        ),
        RequirementRule(
            label="Human Environmental Decision-Making Course",
            kind="at_least_n",
            n=1,
            courses=[
                ("BIO203", "BIO103", "SOC103"),
                "CEE226",
                ("EARTHSYS279", "EARTHSYS179", "BIO279"),
                "EBS220",
                "ECON155",
                ("ENERGY263", "CEE263H"),
                ("ENVRES250", "EARTHSYS254"),
                ("ESS166", "ESS266"),
                ("ESS227", "EARTHSYS227"),
                "GSBGEN386",
                "SUSTAIN103",
            ],
            source=(
                "ENVRES-MS program sheet: Human Environmental Decision-Making "
                "Course -- 'Complete at least 1'; hand-fixed from the "
                "standalone-'OR' parser bug that dropped BIO103/SOC103 (kept only "
                "BIO203) and every alternative after it. MS&E243 excluded (whole "
                "department missing from the course snapshot, see "
                "cme_ms_manual_rules() docstring)."
            ),
        ),
        RequirementRule(
            label="Pathway Courses: Climate Change (at least 4)",
            kind="at_least_n",
            n=4,
            courses=[
                "CEE172",
                ("CEE207A", "CEE107A", "EARTHSYS103", "ENERGY107A", "ENERGY207A"),
                ("CEE261I", "CEE161I", "EARTHSYS146A", "ESS246A"),
                "CEE263C",
                "CEE263D",
                "CEE265E",
                ("CEE265F", "POLISCI227B", "PUBLPOL265F"),
                ("CEE265I", "ESS264"),
                ("CEE276B", "CEE176B"),
                "CEE278A",
                ("CHEMENG274", "BIO273A", "CEE274A", "CHEMENG174"),
                ("EARTHSYS114", "EARTHSYS214", "HUMBIO114"),
                ("EARTHSYS183", "EBS183"),
                ("EARTHSYS212", "EARTHSYS112", "ESS112", "HISTORY103D"),
                ("EARTHSYS217", "EARTHSYS111", "BIO117", "ESS111"),
                ("EARTHSYS223", "EARTHSYS123A", "ESS223", "ESS123"),
                ("EARTHSYS251", "EARTHSYS151", "ESS251", "ESS151"),
                ("ENERGY131", "EARTHSYS101"),
                "ENERGY203",
                ("ENERGY253", "ENERGY153"),
                "EPS226",
                ("ESS202", "ESS102"),
                ("ESS246B", "CEE162I", "CEE262I", "EARTHSYS146B"),
                ("ESS271", "ESS171"),
                "ESS348",
                ("HUMBIO116", "PUBLPOL116"),
                "PHYSICS240",
                "SUSTAIN101C",
                ("SUSTAIN231", "CSRE161", "CSRE261", "ENGLISH131D", "SUSTAIN131"),
                ("SUSTAIN352", "ANTHRO302D", "HISTORY302D"),
                "SUSTAIN370A",
                "SUSTAIN370B",
                "SUSTAIN370C",
            ],
            source=(
                "ENVRES-MS program sheet: Depth in Discipline > Climate Change -- "
                "'Complete at least 4'. One pathway hand-encoded as the concrete "
                "default (see function docstring for why); MS&E394 excluded (see "
                "cme_ms_manual_rules() docstring). One duplicate slot removed "
                "('ESS246A OR CEE161I OR CEE261I OR EARTHSYS146A' is listed twice "
                "verbatim in the raw text, naming the same course set both times)."
            ),
        ),
    ]


# AFRAM-MIN is deliberately NOT in this file / MANUAL_FIXES: independent
# verification (parse_program_text + format_tree on samples/AFRAM-MIN.txt,
# then build_and_solve()) found the generic parser's output already
# structurally correct -- every course named in the raw text ends up in the
# tree, correctly grouped ("Complete ALL: AFRICAAM101A" + "Earn >=5 credits
# from AFRICAAM145B/AFRICAAM104" as two separate top-level rules is
# semantically identical, after requirements.py's AND-flattening, to one
# combined rule; the "Unnamed Rule" / "Foundations of ... OR Foundations of
# ..." labels are literally what bulletin.stanford.edu itself renders, not a
# parser artifact).
#
# CORRECTION (found on a second pass, after this batch's own agent-run
# report): the real bug isn't a course-data gap -- it's the SAME class of
# "a bare requirement-header line got mistaken for a real required course"
# bug documented elsewhere in this project (e.g. the old bare-numeric-id
# corruption in ARCHA-MIN). Line-by-line, samples/AFRAM-MIN.txt reads:
#   AFRICAAM101A - African American Studies
#   Foundations of African Studies OR Foundations of African American Studies
#   Earn at least 5 credits from the following:
#   AFRICAAM145B - Foundations of African Studies
#   AFRICAAM104 - Foundations of African American Studies: ...
# "AFRICAAM101A" here is the section's OWN generic requirement-widget label
# (its title, "African American Studies," is the section name, not a course
# description) -- the actual, real satisfying courses are named two lines
# later. explorecourses does have a real, generically-titled catalog record
# for "AFRICAAM 101A" (confirmed: it's in data/courses_snapshot.json, 5.0
# units), but a 4-year explorecourses.stanford.edu history check turned up
# ZERO offered sections for it in 2022-23 through 2026-27 -- unlike
# ARCHLGY145's real biennial rotation, there's no confirmed year it's ever
# actually run, so a CourseOverride here would be fabricating an offering
# season with no evidence, not correcting a real one-year snapshot gap.
# The correct fix doesn't need one anyway: the bulletin's own text already
# names the two real courses that satisfy this requirement (AFRICAAM145B,
# offered Autumn per the snapshot, and AFRICAAM104), so the rule is a plain
# OR-alternative "all" slot over those two -- AFRICAAM101A never needs to
# appear in the tree at all. Verdict updated from this batch's original
# report: **FIXED**, not NEEDS_COURSE_OVERRIDE.
def afram_min_manual_rules() -> list[RequirementRule]:
    """African and African American Studies (Minor) -- samples/AFRAM-MIN.txt."""
    return [
        RequirementRule(
            label="African American Studies",
            kind="all",
            courses=[("AFRICAAM145B", "AFRICAAM104")],
            source="AFRAM-MIN program page: African American Studies (hand-fixed: "
                   "AFRICAAM101A is the section's own requirement-widget label, "
                   "not a real required course -- the bulletin text itself names "
                   "the two real satisfying courses one line later)",
        ),
        RequirementRule(
            label="Breadth",
            kind="all",
            sub_rules=[
                RequirementRule(
                    label="Creative Practice",
                    kind="at_least_n",
                    n=1,
                    courses=[
                        "AFRICAAM136", "AFRICAAM162", "AFRICAAM135",
                        "AFRICAAM114", "AFRICAAM42",
                    ],
                    source="AFRAM-MIN program page: Breadth -- Creative Practice",
                ),
                RequirementRule(
                    label="Africa-focused OR Americas-focused Courses",
                    kind="at_least_n",
                    n=1,
                    sub_rules=[
                        RequirementRule(
                            label="Africa-focused Courses",
                            kind="at_least_n",
                            n=1,
                            courses=["AFRICAAM161", "AFRICAAM198", "AFRICAAM263"],
                            source="AFRAM-MIN program page: Breadth -- Africa-focused",
                        ),
                        RequirementRule(
                            label="Americas-focused Courses",
                            kind="at_least_n",
                            n=1,
                            courses=[
                                "AFRICAAM214", "AFRICAAM274", "AFRICAAM136",
                                "AFRICAAM42", "AFRICAAM212",
                            ],
                            source="AFRAM-MIN program page: Breadth -- Americas-focused",
                        ),
                    ],
                    source="AFRAM-MIN program page: Breadth -- Unnamed Rule (ANY)",
                ),
            ],
            source="AFRAM-MIN program page: Breadth",
        ),
        RequirementRule(
            label="Depth",
            kind="at_least_n",
            n=1,
            sub_rules=[
                RequirementRule(
                    label="Cultures and Arts",
                    kind="at_least_n",
                    n=2,
                    courses=[
                        "AFRICAAM180", "AFRICAAM120", "AFRICAAM135", "AFRICAAM113N",
                        "AFRICAAM42", "AFRICAAM262", "AFRICAAM162", "AFRICAAM114",
                        "AFRICAAM279", "AFRICAAM145B", "AFRICAAM104",
                    ],
                    source="AFRAM-MIN program page: Depth -- Cultures and Arts",
                ),
                RequirementRule(
                    label="Science, Technology & Society",
                    kind="at_least_n",
                    n=2,
                    courses=["AFRICAAM198", "AFRICAAM263", "AFRICAAM274", "AFRICAAM151"],
                    source="AFRAM-MIN program page: Depth -- Science, Technology & Society",
                ),
                RequirementRule(
                    label="Ethical, Social, and Political Thought",
                    kind="at_least_n",
                    n=2,
                    courses=[
                        "AFRICAAM161", "AFRICAAM210", "AFRICAAM214",
                        "AFRICAAM212", "AFRICAAM139",
                    ],
                    source="AFRAM-MIN program page: Depth -- Ethical, Social, and Political Thought",
                ),
            ],
            source="AFRAM-MIN program page: Depth",
        ),
    ]


MANUAL_FIXES = {
    "BIO-BS": bio_bs_manual_rules,
    "CME-MS": cme_ms_manual_rules,
    "ENVRES-MS": envres_ms_manual_rules,
    "AFRAM-MIN": afram_min_manual_rules,
}
