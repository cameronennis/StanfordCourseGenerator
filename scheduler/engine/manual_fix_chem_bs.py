"""
Hand-fix for CHEM-BS (samples/CHEM-BS.txt) -- found while validating the
double-major/double-degree stacking feature (a user asked for a real
CHEM-BS + CS-BS double-major test and the generic parse turned out wrong).

Same root cause as the batch-A/B/C/D hand-fixes documented in
manual_fix_batchA.py's module docstring: bulletin_parser.py's
_parse_course_slots() only recognizes "AND"/"OR" as continuing a course
list when it's glued to the next course on the SAME line (AND_ALT_RE /
OR_ALT_RE = r"^(AND|OR)\\s+CODE - Title$"). CHEM-BS's "Breadth in
Discipline" section renders its OR-alternatives as bare "OR"/"AND" lines
between two "CODE - Title" lines:

    Foundational Courses
    Complete ALL of the following Courses:
    CHEM121 - ...
    ...
    MATH51 - Linear Algebra, Multivariable Calculus, and Modern Applications
    OR
    MATH61CM - Modern Mathematics: Continuous Methods
    OR
    MATH61DM - Modern Mathematics: Discrete Methods
    PHYSICS41 - Mechanics
    OR
    ...

A standalone connector line matches none of _parse_course_slots' recognized
kinds, so the leaf's course list stops dead at "MATH51" -- every course
after that (both PHYSICS OR-alternatives AND the entire "Advanced Courses"
sub-item that should follow) falls through to parse_one_item()'s generic
label/notes scanning instead, which starts using bare "CODE - Title" lines
as bogus item labels. Confirmed directly against the raw parse: the real
"Advanced Courses" sub-item (CHEM123/124/126/153/173/174/175/176/185) ends
up mislabeled as `[all] MATH61CM - Modern Mathematics: Continuous Methods`,
and the *actual* MATH61CM/MATH61DM/PHYSICS41-44/PHYSICS81 alternatives are
silently dropped from "Foundational Courses" entirely. The same thing
happens to "Additional Courses" (CHEM33/CS106A/MATH19/20/21, mislabeled as
`CHEM31B - Chemical Principles II`) and to the Biological Chemistry Path's
own Foundational/Advanced pair (mislabeled `BIO84 - Physiology`) -- visible
directly in bulletin_parser.py's own emitted warnings for this program
("expected AND connector, saw OR after ...", x3).

This mislabeling does NOT corrupt which courses end up required overall
(the course codes are still present somewhere in the tree, just attached
to the wrong label/rule boundary) -- but it DOES silently drop the real
OR-alternatives (MATH61CM/MATH61DM, PHYSICS41E/61, PHYSICS81) entirely,
since they get scanned as unrecognized "text" and only survive as inert
notes, never as real course-list entries the solver can pick. That's a
real correctness bug, not just a cosmetic labeling one: a plan that could
legitimately substitute MATH61CM for MATH51, or PHYSICS81 for PHYSICS43,
was invisible to the solver.

Per the same project-wide decision as the other hand-fixes (too much blast
radius to generalize bulletin_parser.py's connector-scanning for this one
program's benefit), this is hand-transcribed directly from
samples/CHEM-BS.txt below instead.
"""
from __future__ import annotations

from .models import RequirementRule


def chem_bs_manual_rules() -> list[RequirementRule]:
    """Chemistry (BS) -- samples/CHEM-BS.txt. "Honors" is intentionally
    excluded (opt-in only, not a graduation requirement -- same treatment
    rules_from_parsed_program() already gives every other program's Honors
    section). "Minimum Units in the Program: 89" is metadata, not a
    Complete-ALL/ANY block -- not modeled here, consistent with how no
    other program's own per-major unit floor is modeled (only the
    university-wide 180/225 total is, and only as an informational
    post-solve check -- see engine/university_requirements.py)."""
    return [
        # --- Core Program Requirements ---
        RequirementRule(
            label="Lab Course Requirement",
            kind="all",
            courses=["CHEM100"],
            source="CHEM-BS program page: Core Program Requirements",
        ),
        RequirementRule(
            label="Core Coursework",
            kind="all",
            sub_rules=[
                RequirementRule(
                    label="Introductory Courses",
                    kind="all",
                    sub_rules=[
                        RequirementRule(
                            label="Chem 30 Series",
                            kind="at_least_n",
                            n=1,
                            # "CHEM31A AND CHEM31B" is one alternative (both
                            # required together); "CHEM31E" alone is the
                            # other -- an AND-group nested inside an OR
                            # pick-one, exactly the shape
                            # solver.py's "__and__" tuple exists for.
                            courses=[("__and__", "CHEM31A", "CHEM31B"), "CHEM31E"],
                        ),
                        RequirementRule(
                            label="Additional Courses",
                            kind="all",
                            courses=["CHEM33", "CS106A", "MATH19", "MATH20", "MATH21"],
                        ),
                    ],
                ),
            ],
            source="CHEM-BS program page: Core Program Requirements",
        ),
        # --- Breadth in Discipline ---
        RequirementRule(
            label="Chemistry Pathway",
            kind="at_least_n",
            n=1,
            sub_rules=[
                RequirementRule(
                    label="Traditional Chemistry Path",
                    kind="all",
                    sub_rules=[
                        RequirementRule(
                            label="Foundational Courses",
                            kind="all",
                            courses=[
                                "CHEM121", "CHEM131", "CHEM151", "CHEM171", "CHEM181",
                                ("MATH51", "MATH61CM", "MATH61DM"),
                                ("PHYSICS41", "PHYSICS41E", "PHYSICS61"),
                                "PHYSICS42",
                                ("PHYSICS43", "PHYSICS81"),
                                "PHYSICS44",
                            ],
                        ),
                        RequirementRule(
                            label="Advanced Courses",
                            kind="all",
                            courses=[
                                "CHEM123", "CHEM124", "CHEM126", "CHEM153",
                                "CHEM173", "CHEM174", "CHEM175", "CHEM176", "CHEM185",
                            ],
                        ),
                    ],
                ),
                RequirementRule(
                    label="Biological Chemistry Path",
                    kind="all",
                    sub_rules=[
                        RequirementRule(
                            label="Foundational Courses",
                            kind="all",
                            courses=[
                                ("BIO86", "BIO84", "BIO82"),
                                "CHEM121", "CHEM131", "CHEM151", "CHEM171", "CHEM181",
                                ("MATH51", "MATH61CM", "MATH61DM"),
                                ("PHYSICS41", "PHYSICS41E", "PHYSICS61"),
                                "PHYSICS42",
                                ("PHYSICS43", "PHYSICS81"),
                                "PHYSICS44",
                            ],
                        ),
                        RequirementRule(
                            label="Advanced Courses",
                            kind="all",
                            courses=[
                                "CHEM123", "CHEM124", "CHEM126",
                                ("CHEM173", "CHEM175"),
                                "CHEM176", "CHEM183", "CHEM184", "CHEM185",
                            ],
                        ),
                    ],
                ),
            ],
            source="CHEM-BS program page: Breadth in Discipline",
        ),
        # --- Writing in the Major (WIM) ---
        RequirementRule(
            label="WIM Course",
            kind="all",
            courses=["CHEM131"],
            source="CHEM-BS program page: Writing in the Major (WIM)",
        ),
        # --- Capstone ---
        RequirementRule(
            label="Course Requirement",
            kind="all",
            courses=["CHEM185"],
            source="CHEM-BS program page: Capstone",
        ),
    ]


def _chem_bs_shared_rules() -> list[RequirementRule]:
    """The Core Program Requirements / WIM / Capstone rules, shared by both
    path-specific variants below -- everything in chem_bs_manual_rules()
    except the "Chemistry Pathway" choice-of-two-paths rule itself."""
    all_rules = chem_bs_manual_rules()
    return [r for r in all_rules if r.label != "Chemistry Pathway"]


def chem_bs_traditional_path_rules() -> list[RequirementRule]:
    """Chemistry (BS), Traditional Chemistry Path only -- same as
    chem_bs_manual_rules() but with the "Chemistry Pathway" pick-one-of-two
    rule replaced by just the Traditional path's own sub_rules (no
    Biological-path alternative to pick, since the user has already chosen
    which path applies to them). Mirrors the ENVSE-BS/SLAV-BA convention of
    registering named track/path variants as separate PROGRAM_SHEET_BUILDERS
    entries sharing one base code."""
    all_rules = chem_bs_manual_rules()
    pathway_rule = next(r for r in all_rules if r.label == "Chemistry Pathway")
    traditional = next(r for r in pathway_rule.sub_rules if r.label == "Traditional Chemistry Path")
    # Flatten: drop the "Traditional Chemistry Path" wrapper label too, since
    # there's no longer a sibling path to distinguish it from -- just surface
    # its own sub_rules ("Foundational Courses" / "Advanced Courses")
    # directly alongside the shared rules.
    return _chem_bs_shared_rules() + list(traditional.sub_rules)


def chem_bs_biological_path_rules() -> list[RequirementRule]:
    """Chemistry (BS), Biological Chemistry Path only -- see
    chem_bs_traditional_path_rules() docstring; same idea, other path."""
    all_rules = chem_bs_manual_rules()
    pathway_rule = next(r for r in all_rules if r.label == "Chemistry Pathway")
    biological = next(r for r in pathway_rule.sub_rules if r.label == "Biological Chemistry Path")
    return _chem_bs_shared_rules() + list(biological.sub_rules)
