"""
Hand-encoded requirement trees for four programs whose bulletin_parser.py
parse comes back wrong in ways that a targeted parser fix would be risky to
generalize (see README's parser-bug sweep). Each function below is a
from-scratch, by-hand transcription of the raw samples/{CODE}.txt text --
NOT a patched version of what bulletin_parser.py produces -- following the
build_cs_bs_core_requirements() pattern in requirements.py.

Verdicts (see each function's docstring for detail):
  ME-MS     -- FIXED. Depth/breadth/math "OR"-joined sibling-area blocks
               hand-encoded as proper "pick N of these areas" trees.
  MGTSC-MS  -- NOT fixable here. The entire MS&E (Management Science and
               Engineering) subject is absent from data/courses_snapshot.json
               -- every single course this program requires (both Breadth
               and Depth) is MS&E-prefixed. Rule tree is included below,
               correctly encoded, for whenever that data gap is closed, but
               it is NOT in MANUAL_FIXES since build_and_solve() cannot
               possibly place any of its requirements today.
  STATS-MS  -- FIXED. Breadth section's missing top-level connector (which
               had made every elective category mandatory) replaced with a
               flattened "at least 3 of the union" pool; Depth's unenumerated
               "Untitled Course Set" replaced with a repeatable placeholder.
  URBST-BA  -- FIXED. "One of the following pathways" (six pathways, no
               literal OR token) hand-encoded as an at_least_n(n=1) over six
               sub-rules.
"""
from __future__ import annotations

from .models import RequirementRule, make_placeholder_course


# ---------------------------------------------------------------------------
# ME-MS -- Mechanical Engineering (MS)
# ---------------------------------------------------------------------------
# Bug confirmed against samples/ME-MS.txt: three separate sections
# (Mathematical Fundamentals, Depth in Mechanical Engineering, Breadth in
# Mechanical Engineering) each list several sibling sub-areas (e.g. "Linear
# Algebra" / "Partial Differential Equations" / ... under Math; "Automatic
# Controls" / "Biomechanical Engineering" / ... under Depth) whose raw text
# joins them with a bare "OR" line sitting right at the boundary between one
# area's last course and the next area's heading (e.g. line 101: "...ME327
# ... / OR / Biomechanical Engineering"). bulletin_parser.py doesn't
# recognize an "OR" line immediately before a sub-heading as grouping the
# *sections* -- so instead of "pick N of these M areas", every area became
# its own independent mandatory rule, requiring all 4 math areas AND all 12
# depth areas AND all 14 breadth areas simultaneously (clearly INFEASIBLE:
# the depth areas alone are mutually-exclusive-by-design specializations).
# Each section already has its own explicit "Complete at least N of the
# following:" header giving the real N, confirmed independently against the
# raw text below.
#
# Within an area, a bare "OR" *between two consecutive courses* (not at an
# area boundary) means those two/three/four course listings are the same
# course cross-listed under different department codes (e.g. "ME274A OR
# AA274A OR CS237A OR EE260A" are one course, four subject codes) -- encoded
# as an OR-alternatives tuple so a student can't be asked to take the same
# course twice under two different codes.
def me_ms_manual_rules() -> list[RequirementRule]:
    SRC = "ME-MS program page (hand-fixed depth/breadth OR-grouping)"

    # -- Mathematical Fundamentals: "Complete at least 2 of the following:"
    # 4 areas (Linear Algebra, PDE, Numerical Analysis, Statistics), each
    # itself "Complete ANY of the following Courses" (pick >=1 within the
    # area to satisfy that area).
    linear_algebra = RequirementRule(
        label="Math Fundamentals: Linear Algebra",
        kind="at_least_n", n=1,
        courses=[("ME300A", "CME200"), "CME302", "CS205L", ("EE263", "CME263"), "EE363"],
        source=SRC,
    )
    pde = RequirementRule(
        label="Math Fundamentals: Partial Differential Equations",
        kind="at_least_n", n=1,
        courses=[("CME303", "MATH220A"), ("ME300B", "CME204")],
        source=SRC,
    )
    numerical_analysis = RequirementRule(
        label="Math Fundamentals: Numerical Analysis",
        kind="at_least_n", n=1,
        courses=["CME306", ("ME300C", "CME206"), ("ME408", "CME322")],
        source=SRC,
    )
    statistics = RequirementRule(
        label="Math Fundamentals: Statistics",
        kind="at_least_n", n=1,
        courses=[("CME106", "ENGR155C"), "STATS117", "STATS118", "STATS200"],
        source=SRC,
    )
    math_fundamentals = RequirementRule(
        label="Mathematical Fundamentals (at least 2 of 4 areas)",
        kind="at_least_n", n=2,
        sub_rules=[linear_algebra, pde, numerical_analysis, statistics],
        source=SRC,
    )

    # -- Depth in Mechanical Engineering: "Complete at least 1 of the
    # following:" 12 depth areas.
    automatic_controls = RequirementRule(
        label="Depth: Automatic Controls", kind="at_least_n", n=3,
        courses=[
            "AA203", "AA212", ("AA228", "CS238"), "AA277", "ENGR105", "ENGR205",
            ("ME274A", "AA274A", "CS237A", "EE260A"),
            ("ME274B", "AA274B", "CS237B", "EE260B"),
            "ME286", "ME327",
        ], source=SRC,
    )
    biomechanical = RequirementRule(
        label="Depth: Biomechanical Engineering", kind="all",
        sub_rules=[
            RequirementRule(
                label="Foundational Courses", kind="at_least_n", n=2,
                courses=["ME235", "ME244", "ME249", "ME281", "ME283", "ME287", ("ME305", "BIOE305")],
                source=SRC,
            ),
            RequirementRule(
                label="System-Specific Courses", kind="at_least_n", n=1,
                courses=["ME234", "ME285", ("ME303", "MATSCI333"), "ME381", "ME485"],
                source=SRC,
            ),
        ], source=SRC,
    )
    design_methodology = RequirementRule(
        label="Depth: Design Methodology", kind="all",
        courses=["ME310A", "ME310B", "ME310C"], source=SRC,
    )
    dynamics = RequirementRule(
        label="Depth: Dynamics", kind="at_least_n", n=3,
        courses=["AA242A", "AA279A", "CS225A", ("CS327A", "ME323"), "ME331A", "ME331B", "ME334"],
        source=SRC,
    )
    energy_transport = RequirementRule(
        label="Depth: Energy and Transport Sciences", kind="all",
        sub_rules=[
            RequirementRule(label="Required Courses", kind="all", courses=["ME352B", "ME370A"], source=SRC),
            RequirementRule(
                label="Additional Course", kind="at_least_n", n=1,
                courses=["ME352C", "ME352D", "ME362A", "ME371", "ME457"], source=SRC,
            ),
        ], source=SRC,
    )
    fluid_mechanics = RequirementRule(
        label="Depth: Fluid Mechanics", kind="all",
        sub_rules=[
            RequirementRule(label="Required Courses", kind="all", courses=["ME351A", "ME351B"], source=SRC),
            RequirementRule(
                label="Additional Course", kind="at_least_n", n=1,
                courses=["ME354", "ME355", "ME361", "ME451A", "ME451B", "ME451C", "ME461"], source=SRC,
            ),
        ], source=SRC,
    )
    manufacturing = RequirementRule(
        label="Depth: Manufacturing and Product Realization", kind="all",
        sub_rules=[
            RequirementRule(label="Required Course", kind="all", courses=["ME203"], source=SRC),
            RequirementRule(
                label="Additional Courses", kind="at_least_n", n=2,
                courses=["ME204", "ME217", "ME219", "ME225", "ME227", "ME318",
                         ("ME321", "MATSCI351"), "ME324", "ME325"], source=SRC,
            ),
        ], source=SRC,
    )
    materials_stress = RequirementRule(
        label="Depth: Materials and Stress Analysis", kind="at_least_n", n=3,
        courses=["ME219", ("ME258", "MATSCI358"), ("ME303", "MATSCI333"), "ME329", "ME340"],
        source=SRC,
    )
    mechatronics = RequirementRule(
        label="Depth: Mechatronics", kind="at_least_n", n=2,
        courses=["ME218A", "ME218B", "ME218C"], source=SRC,
    )
    reactive_gas = RequirementRule(
        label="Depth: Reactive and Ionized Gas Dynamics", kind="all",
        sub_rules=[
            RequirementRule(label="Required Course", kind="all", courses=["ME362A"], source=SRC),
            RequirementRule(
                label="Additional Courses", kind="at_least_n", n=2,
                courses=["ME362B", "ME363", "ME364", "ME366", "ME371", "ME372", "ME374", "ME463"], source=SRC,
            ),
        ], source=SRC,
    )
    robotics = RequirementRule(
        label="Depth: Robotics and Kinematics", kind="at_least_n", n=3,
        courses=[
            "AA277", "CS225A", ("CS229", "STATS229"), "CS326",
            ("ME274A", "AA274A", "CS237A", "EE260A"),
            ("ME274B", "AA274B", "CS237B", "EE260B"),
            ("ME303", "MATSCI333"), ("ME320", "CS223A"), ("ME323", "CS327A"),
            ("ME326", "CS339R"), "ME327",
        ], source=SRC,
    )
    solid_mechanics = RequirementRule(
        label="Depth: Solid Mechanics", kind="at_least_n", n=3,
        courses=["ME335A", "ME338", "ME340", "ME346A"], source=SRC,
    )
    depth = RequirementRule(
        label="Depth in Mechanical Engineering (at least 1 of 12 areas)",
        kind="at_least_n", n=1,
        sub_rules=[
            automatic_controls, biomechanical, design_methodology, dynamics,
            energy_transport, fluid_mechanics, manufacturing, materials_stress,
            mechatronics, reactive_gas, robotics, solid_mechanics,
        ], source=SRC,
    )

    # -- Breadth in Mechanical Engineering: "Two courses are required from
    # the list of eligible breadth courses ... They can come from the same
    # area or two different areas" -- i.e. genuinely "at least 2 of the
    # union", not "one from each of two areas". Flattened accordingly
    # (de-duplicated -- several courses/equivalence-groups repeat across
    # areas, e.g. ME327 appears in both Automatic Controls and Robotics).
    def _dedup(items):
        seen = set()
        out = []
        for it in items:
            key = it if isinstance(it, str) else tuple(sorted(it)) if isinstance(it, tuple) else it
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    breadth_pool = _dedup([
        "AA203", "AA212", ("AA228", "CS238"), "AA277", "ENGR105", "ENGR205",
        ("ME274A", "AA274A", "CS237A", "EE260A"), ("ME274B", "AA274B", "CS237B", "EE260B"),
        "ME286", "ME327",
        "ME234", "ME235", "ME244", "ME249", "ME281", "ME283", "ME285", "ME287",
        ("ME303", "MATSCI333"), ("ME305", "BIOE305"), "ME381", "ME485",
        "ME318", "ME324",
        "AA242A", "AA279A", "CS225A", ("CS327A", "ME323"), "ME331A", "ME331B", "ME334",
        "ME352B", "ME352C", "ME352D", "ME362A", "ME370A", "ME371", "ME457",
        "ME351A", "ME351B", "ME354", "ME355", "ME361", "ME451A", "ME451B", "ME451C", "ME461",
        "MATSCI251", "ME203", "ME204", "ME206B", "ME217", "ME219", "ME225", "ME227",
        "ME321", "MATSCI351", "ME325",
        ("ME258", "MATSCI358"), "ME329", "ME340",
        "ME210", "ME220",
        "ENGR240", "ENGR241",
        "ME362B", "ME363", "ME364", "ME366", "ME372", "ME374", "ME463",
        ("CS229", "STATS229"), "CS326", ("ME320", "CS223A"), ("ME326", "CS339R"),
        "ME335A", "ME338", "ME339", "ME346A", "ME346B",
        "AA222", "CS131", "CS230", "CS234", "CS236", "EE364A", "EE364B",
        "ME233", "ME343", "ME470",
    ])
    breadth = RequirementRule(
        label="Breadth in Mechanical Engineering (at least 2 of the union of breadth areas)",
        kind="at_least_n", n=2, courses=breadth_pool, source=SRC,
    )

    # -- Laboratory Requirement: enumerated list is intact in the raw text
    # (unlike the "course"-placeholder-riddled elective/petition prose later
    # on that we deliberately don't try to hand-encode -- those reference
    # specific approved courses that never rendered as text, just the
    # literal word "course").
    lab = RequirementRule(
        label="Laboratory Requirement (at least 1 course with a lab component)",
        kind="at_least_n", n=1,
        courses=[
            "ME203", "ME210", "ME218A", "ME218B", "ME218C", "ME220", "ME287", "ME327",
            "ME310B", "ME310C", "ME318", "ME324", "ME326", "ME354", "ME367",
            ("ME274B", "AA274B", "CS237B", "EE260B"),
        ], source=SRC,
    )

    return [math_fundamentals, depth, breadth, lab]


# ---------------------------------------------------------------------------
# MGTSC-MS -- Management Science and Engineering (MS)
# ---------------------------------------------------------------------------
# Bug confirmed against samples/MGTSC-MS.txt: within the "Data, AI, and
# Decisions" depth area's course list, line 121 has a bare "OR" between
# "MS&E232 - Introduction to Game Theory" and "MS&E232H - Introduction to
# Game Theory (Accelerated)" (an equivalence pair, same course two ways) --
# NOT at a list/section boundary like ME-MS's bug, but mid-list. This is
# exactly the shape the diagnosis describes as causing bulletin_parser.py to
# truncate/mislabel the rest of the list: everything from line 123
# (MS&E233) through line 156 (MS&E463) -- 33 real course-list entries --
# silently vanished from that area's parsed course pool.
#
# HOWEVER: fixing that OR-grouping bug does not make this program
# schedulable. Every single course MGTSC-MS requires, in both its Breadth
# and Depth sections, is MS&E-prefixed -- and the MS&E (Management Science
# and Engineering) subject is entirely absent from
# data/courses_snapshot.json (confirmed: 'MS&E' / 'MSE' not in
# d["departments"], and grepping course titles for "Introduction to
# Optimization" only turns up ENGR62, a cross-listing stub whose own title
# says "(MS&E 111, MS&E 211)" -- the MS&E-side course records themselves
# were never fetched). No CourseOverride can fix this: CourseOverride only
# corrects a field on an *existing* Course record (see models.py), and
# there's nothing to correct here -- around 100+ MS&E course records need to
# be added to the snapshot in the first place, which is a fetch/scrape gap,
# not a one-line data correction.
#
# Verdict: NEEDS_COURSE_OVERRIDE doesn't quite fit (it's not a single
# override, it's a missing department), but it's the closest of the four
# buckets -- the rule tree below is correct and ready to use the moment the
# MS&E course catalog is actually present in the snapshot.
#
# UPDATE: the missing department has since been fetched directly (explorecourses'
# real department code is "MS&E" -- confirmed via CourseConnection.get_schools()
# -- 160 courses added to data/courses_snapshot.json). Re-tested this exact,
# unmodified tree against the updated snapshot: build_and_solve() -> OPTIMAL.
# Verdict updated from this batch's original report: **FIXED**, now included
# in MANUAL_FIXES below.
def mgtsc_ms_manual_rules() -> list[RequirementRule]:
    SRC = "MGTSC-MS program page (hand-fixed mid-list OR truncation)"

    breadth = RequirementRule(
        label="MS&E Breadth (one course from each of 4 areas)", kind="all",
        sub_rules=[
            RequirementRule(
                label="Optimization and Algorithms", kind="at_least_n", n=1,
                courses=["MS&E211", "MS&E211DS", "MS&E211X", "MS&E212", "MS&E215", "MS&E236",
                         "MS&E311", "MS&E312", "MS&E315", "MS&E318", "MS&E319", "MS&E331"],
                source=SRC,
            ),
            RequirementRule(
                label="Data Science, Probability, and Stochastics", kind="at_least_n", n=1,
                courses=["MS&E220", "MS&E221", "MS&E223", "MS&E226", "MS&E228",
                         "MS&E321", "MS&E322", "MS&E323", "MS&E324"],
                source=SRC,
            ),
            RequirementRule(
                label="Strategy, Entrepreneurship, and Organizations", kind="at_least_n", n=1,
                courses=["MS&E270", "MS&E272", "MS&E280", "MS&E284"], source=SRC,
            ),
            RequirementRule(
                label="Modeling, Decision, and Management Science Applications", kind="at_least_n", n=1,
                courses=["MS&E230", "MS&E232", "MS&E232H", "MS&E235A", "MS&E235B", "MS&E239",
                         "MS&E241", "MS&E242", "MS&E243", "MS&E245A", "MS&E245B", "MS&E246",
                         "MS&E247", "MS&E250A", "MS&E250B", "MS&E252", "MS&E254", "MS&E260",
                         "MS&E262", "MS&E292"],
                source=SRC,
            ),
        ], source=SRC,
    )

    data_ai_decisions = RequirementRule(
        label="Depth: Data, AI, and Decisions", kind="at_least_n", n=3,
        courses=[
            "MS&E206", "MS&E211", "MS&E211DS", "MS&E211X", "MS&E212", "MS&E215", "MS&E218",
            "MS&E220", "MS&E221", "MS&E223", "MS&E226", "MS&E228", "MS&E229", "MS&E230",
            ("MS&E232", "MS&E232H"), "MS&E233", "MS&E235A", "MS&E235B", "MS&E236", "MS&E239",
            "MS&E250A", "MS&E250B", "MS&E252", "MS&E260", "MS&E262", "MS&E263",
            "MS&E311", "MS&E312", "MS&E315", "MS&E318", "MS&E319", "MS&E321", "MS&E322",
            "MS&E323", "MS&E324", "MS&E325", "MS&E326", "MS&E328", "MS&E330", "MS&E331",
            "MS&E333", "MS&E334", "MS&E335", "MS&E336", "MS&E337", "MS&E338", "MS&E339",
            "MS&E355", "MS&E463",
        ], source=SRC,
    )
    financial_analytics = RequirementRule(
        label="Depth: Financial Analytics", kind="at_least_n", n=3,
        courses=["MS&E242", "MS&E244", "MS&E245A", "MS&E245B", "MS&E246", "MS&E247",
                 "MS&E249", "MS&E342", "MS&E346", "MS&E347", "MS&E348", "MS&E349"],
        source=SRC,
    )
    tech_eng_mgmt = RequirementRule(
        label="Depth: Technology and Engineering Management", kind="at_least_n", n=3,
        courses=["ENGR245", "ENGR248", "MS&E240", "MS&E265", "MS&E270", "MS&E271",
                 "MS&E272", "MS&E273", "MS&E275", "MS&E276", "MS&E277B", "MS&E278",
                 "MS&E280", "MS&E284", "MS&E288", "MS&E289", "MS&E489"],
        source=SRC,
    )
    tech_policy = RequirementRule(
        label="Depth: Technology Policy", kind="at_least_n", n=3,
        courses=["MS&E241", "MS&E243", "MS&E254", "MS&E256", "MS&E292", "MS&E295",
                 "MS&E296", "MS&E297", "MS&E299", "MS&E394"],
        source=SRC,
    )
    depth = RequirementRule(
        label="MS&E Depth (any 1 of 4 areas, or self-designed)", kind="at_least_n", n=1,
        sub_rules=[data_ai_decisions, financial_analytics, tech_eng_mgmt, tech_policy],
        source=SRC,
    )

    return [breadth, depth]


# ---------------------------------------------------------------------------
# STATS-MS -- Statistics (MS)
# ---------------------------------------------------------------------------
# Bug confirmed against samples/STATS-MS.txt: the "Breadth Requirement"
# section (line 98 on) lists 11 elective category sub-headers (Biological
# Sciences, CME, Computer Science, Economic Analysis & Policy/GSB/MS&E,
# Operations Management, Mathematics, Statistics, Electrical Engineering,
# Civil & Environmental Engineering, Political Science and Public Policy,
# Psychology & Social Science) whose *only* stated connector is prose --
# "Minimum three courses (3 units each)" (line 103) -- with no literal
# ALL/ANY token anywhere above the sub-headers the way e.g. MGTSC-MS's
# Breadth has an explicit "Fulfill ALL of the following requirements:".
# Without that token, bulletin_parser.py has nothing to key off of and
# defaults every category to mandatory -- which would require completing
# essentially the entire elective catalog across a dozen departments
# simultaneously, obviously not what "minimum three courses" means.
# Re-encoded as a single flattened "at least 3 of the union" pool (matching
# the literal prose), skipping the MS&E-only "Operations Management"
# category and its MS&E items inside "Economic Analysis..." (MS&E is absent
# from the snapshot entirely -- see MGTSC-MS above) since there are already
# a dozen other real, schedulable categories to pick from.
#
# A second, separate issue in the same program: the Depth section's course
# list literally says "Untitled Course Set" (line 95) -- the live bulletin
# page embeds this as a dynamic widget that the raw-text scrape never
# captured any course codes for, unlike a normal enumerated list. Genuinely
# open-ended per this file's convention (see models.make_placeholder_course
# docstring) -- represented as a single repeatable placeholder standing in
# for "any STATS graduate course numbered >=200", taken 5 times (the
# solver's existing "solo at_least_n repeat" mechanism, see solver.py).
def stats_ms_manual_rules() -> list[RequirementRule]:
    SRC = "STATS-MS program page (hand-fixed breadth top-level connector + depth placeholder)"

    probability_core = RequirementRule(
        label="Probability core requirement", kind="all",
        courses=[("STATS118", "STATS218", "STATS310A")], source=SRC,
    )
    stochastic_core = RequirementRule(
        label="Stochastic Processes core requirement", kind="all",
        courses=[("STATS217", "STATS219", "STATS310A")], source=SRC,
    )
    theoretical_stats_core = RequirementRule(
        label="Theoretical Statistics core requirement", kind="all",
        courses=[("STATS200", "STATS270", "STATS300A")], source=SRC,
    )
    applied_stats = RequirementRule(
        label="Applied Statistics requirement", kind="all",
        courses=[("STATS203", "STATS305A")], source=SRC,
    )
    linear_algebra_core = RequirementRule(
        label="Linear Algebra core requirement", kind="all",
        courses=[("MATH104", "MATH113", "CME302", "CME364A")], source=SRC,
    )
    programming_req = RequirementRule(
        label="Programming requirement", kind="all",
        courses=[("CS106B", "CS107", "CME108")], source=SRC,
    )

    depth_placeholder = make_placeholder_course(
        "PLACEHOLDER:STATS-MS-DEPTH-STATS200PLUS",
        "any Statistics-department graduate course numbered 200 or above "
        "(not a workshop/seminar/practicum/independent-research course, and "
        "not STATS 200Q) -- bulletin page lists this as an unenumerated "
        "'Untitled Course Set' widget with no course codes in the raw text",
    )
    depth = RequirementRule(
        label="Depth Requirement (5 STATS courses at/above the 200 level)",
        kind="at_least_n", n=5,
        courses=[depth_placeholder.key], source=SRC,
    )

    breadth_pool = [
        # Biological Sciences
        "BIO244", "BIO283", "BMDS214", "BMDS241", "BMDS291A", "BMDS280A",
        "BMDS291B", "BMDS291C", "BMDS280B", "BMDS280C", "GENE211",
        # Computational and Mathematical Engineering (CME)
        "CME200", "CME302", "CME305", "CME306", "CME307", "CME309", "CME364A", "CME364B",
        # Computer Science
        "CS140", "CS142", "CS143", "CS144", "CS145", "CS147", "CS149", "CS154", "CS155",
        "CS157", "CS161", "CS182", "CS205L", "CS221", "CS224C", "CS224G", "CS224N", "CS224R",
        "CS224S", "CS224U", "CS224W", "CS228", "CS229", "CS246", "CS261", "CS273C", "CS288",
        "CS329X", "CS328",
        # Economic Analysis & Policy, GSB (non-MS&E items only -- MS&E absent from snapshot)
        "ECON202N", "ECON210", "ECON271", "ECON272", "ECON273", "ECON274",
        "FINANCE620", "FINANCE622", "MGTECON634",
        # Mathematics
        "MATH158", "MATH171", "MATH205A", "MATH220A", "MATH236",
        # Electrical Engineering
        "EE261", "EE263", "EE269", "EE278", "EE363", "EE364A", "EE364B", "EE377", "EE378A", "EE379A",
        # Civil & Environmental Engineering
        "CEE203", "CEE241", "CEE241A", "CEE244", "CEE289",
        # Political Science and Public Policy (flattened into the elective pool like
        # every other category -- the bulletin literally says "Complete ALL" for this
        # one category alone, which would be an odd outlier vs. every sibling category
        # saying "ANY"; treated as individually-optional electives consistent with the
        # rest of Breadth rather than as a mandatory 7-course sub-sequence)
        "POLISCI450A", "POLISCI450B", "POLISCI450C", "POLISCI450D", "POLISCI451",
        "POLISCI455", "PUBLPOL50",
        # Psychology & Social Science
        "PSYCH253", "SOC383", "PSYCH299",
    ]
    breadth = RequirementRule(
        label="Breadth Requirement (at least 3 courses across elective categories)",
        kind="at_least_n", n=3, courses=breadth_pool, source=SRC,
    )

    # Note: the "Data Science Subplan" sections (a separate, much more
    # ML/stats-heavy course of study under the same bulletin page) are
    # deliberately excluded here, same as how requirements.py's
    # rules_from_parsed_program() defaults a BS with named subplans/tracks
    # to just the base curriculum unless a student opts in -- this program
    # sheet's own section title is literally "Data Science Subplan".
    return [probability_core, stochastic_core, theoretical_stats_core, applied_stats,
            linear_algebra_core, programming_req, depth, breadth]


def stats_ms_placeholder_courses() -> dict[str, str]:
    """Placeholder key -> description for stats_ms_manual_rules(); merge
    into the course dict via course_data.add_placeholder_courses() before
    solving, same contract as build_program_from_sample()'s second return
    value."""
    ph = make_placeholder_course(
        "PLACEHOLDER:STATS-MS-DEPTH-STATS200PLUS",
        "any Statistics-department graduate course numbered 200 or above",
    )
    return {ph.key: ph.description}


# ---------------------------------------------------------------------------
# URBST-BA -- Urban Studies (BA)
# ---------------------------------------------------------------------------
# Bug confirmed against samples/URBST-BA.txt: "Depth in Discipline" (line
# 142) reads "Students must complete at least 20 units in one of the
# following pathways:" followed by six named pathways (five real ones plus
# "Self-Designed"), each written up in its own "Fulfill ALL of the following
# requirements:" block further down the page. There is no literal OR/AND
# token anywhere joining the six pathway blocks to each other -- the only
# signal that they're mutually-alternative is the natural-language "one of
# the following pathways" sentence. bulletin_parser.py has nothing to
# pattern-match there, so it fell through to treating every pathway (all 5
# real ones, each needing ~20 units of its own required+elective courses) as
# independently mandatory -- roughly 100 units of major requirements instead
# of 20, guaranteed INFEASIBLE. Re-encoded as at_least_n(n=1) over 6
# sub-rules, one per pathway.
def urbst_ba_manual_rules() -> list[RequirementRule]:
    SRC = "URBST-BA program page (hand-fixed 'one of the following pathways' choice)"

    core = RequirementRule(
        label="Urban Studies Core", kind="all",
        courses=["URBANST110", "URBANST112", "URBANST113", "URBANST114", "ECON1"],
        source=SRC,
    )

    skills = RequirementRule(
        label="Skills Courses", kind="all",
        sub_rules=[
            RequirementRule(label="Required Skills Courses", kind="all",
                             courses=["EARTHSYS144", "SOC180A"], source=SRC),
            RequirementRule(
                label="Additional Skills Course", kind="at_least_n", n=1,
                courses=[
                    "ANTHRO91", "ANTHRO98C", "ARCHLGY125", "CEE31Q", "CEE118X", "CEE118Y",
                    "CEE118Z", "CEE130", "CEE133F", "CEE154", "DATASCI112", "DATASCI154",
                    "EARTHSYS142", "ECON102A", "HUMBIO82A", "HUMBIO82B", "MED147",
                    "PEDS202C", "POLISCI150A", "POLISCI150B", "POLISCI150C", "SOC180B",
                    "STATS60", "URBANST123A", "URBANST123B", "URBANST145",
                ], source=SRC,
            ),
        ], source=SRC,
    )

    community_engaged_learning = RequirementRule(
        label="Community Engaged Learning", kind="at_least_n", n=1,
        courses=[
            "OSPPARIS17", "SINY56", "SINY101", "URBANST125", "URBANST141",
            "URBANST155", "URBANST164", "URBANST165", "URBANST201A",
        ], source=SRC,
    )

    global_urban = RequirementRule(
        label="Pathway: Global Urban Culture and History", kind="all",
        sub_rules=[
            RequirementRule(label="Required Course", kind="all",
                             courses=[("DLCL100", "COLLEGE118")], source=SRC),
            RequirementRule(
                label="Elective Courses", kind="at_least_n", n=4,
                courses=[
                    "AMSTUD145", "AMSTUD154D", "ANTHRO39", "ANTHRO167", "ARTHIST142",
                    "ARTHIST143A", "ARTHIST250", "ARTSTUDI11Q", "CEE32G", "CEE33B",
                    "CEE32R", "CEE41Q", "CLASSICS83", "CLASSICS84", "CLASSICS156",
                    "CSRE270", "EARTHSYS112", "ENGLISH83N", "HISTORY91B", "HISTORY106A",
                    "HISTORY106B", "HISTORY110C", "HISTORY150C", "HISTORY232B", "JAPAN125",
                    "ME120", "MUSIC11Q", "OSPBER30", "OSPBER60", "OSPCPTWN79",
                    "OSPFLOR115Y", "OSPHONGK30", "OSPHONGK39", "OSPHONGK89", "OSPISTAN88",
                    "OSPKYOCT199", "OSPMADRD8A", "OSPMADRD60", "OSPPARIS94", "OSPPARIS99P",
                    "OSPSANTG29", "OSPSANTG71", "POLISCI110C", "SINY114", "SLAVIC36N",
                    "URBANST27Q", "URBANST83N", "URBANST100UR", "URBANST114", "URBANST119",
                    "URBANST140F", "URBANST141", "URBANST147", "URBANST150", "URBANST169",
                    "URBANST169B",
                ], source=SRC,
            ),
        ], source=SRC,
    )

    race_ethnicity = RequirementRule(
        label="Pathway: Race, Ethnicity, and Urban Life", kind="all",
        sub_rules=[
            RequirementRule(label="Required Course", kind="all", courses=["CSRE100"], source=SRC),
            RequirementRule(
                label="Elective Courses", kind="at_least_n", n=4,
                courses=[
                    "AFRICAAM80Q", "AFRICAAM105", "ANTHRO31Q", "ANTHRO111C", "ANTHRO167",
                    "ARTHIST291", "ASNAMST100", "ASNAMST110", "ASNAMST261", "CHILATST100",
                    "CHILATST173", "CHILATST177A", "CHILATST180E", "CSRE30Q", "CSRE147A",
                    "EDUC103B", "EDUC392", "HISTORY200C", "HISTORY255D", "HISTORY271B",
                    "HISTORY274C", "OSPHONGK89", "OSPPARIS94", "PEDS150", "POLISCI121L",
                    "POLISCI141A", "PWR194AJB", "SINY15", "SINY63", "SOC3", "SOC179A",
                    "SOC189", "URBANST125", "URBANST139", "URBANST140F", "URBANST141",
                    "URBANST156A", "URBANST164", "URBANST169", "URBANST169B",
                ], source=SRC,
            ),
        ], source=SRC,
    )

    urban_education = RequirementRule(
        label="Pathway: Urban Education", kind="all",
        sub_rules=[
            RequirementRule(label="Required Course", kind="all", courses=["URBANST115"], source=SRC),
            RequirementRule(
                label="Elective Courses", kind="at_least_n", n=4,
                courses=[
                    "AFRICAAM106", "EDUC101", "EDUC103A", "EDUC103B", "EDUC107",
                    "EDUC271", "EDUC131", "EDUC201", "EDUC202", "EDUC204", "EDUC220C",
                    "EDUC277", "EDUC283", ("HUMBIO142", "PSYCH60"), "PWR194AJB",
                ], source=SRC,
            ),
        ], source=SRC,
    )

    urban_society = RequirementRule(
        label="Pathway: Urban Society & Social Change", kind="all",
        sub_rules=[
            RequirementRule(label="Required Course", kind="all", courses=["URBANST156A"], source=SRC),
            RequirementRule(
                label="Elective Courses", kind="at_least_n", n=4,
                courses=[
                    "AMSTUD154D", "CEE32A", "CEE32B", "CEE141A", "CEE177L", "CEE141B",
                    "CEE246", "CSRE12", "CSRE107", "CSRE100", "EARTHSYS181", "ECON150",
                    "ECON155", "EDUC107", "ENVRES223", "HISTORY106A", "HISTORY106B",
                    "INTNLREL142", "LAW2505", ("MS&E180", "SOC160"), "OSPCPTWN79",
                    "OSPSANTG29", "OSPSANTG71", "PEDS150", "POLISCI121L", "POLISCI236",
                    "PUBLPOL135", "SINY101", "SOC3", "SOC14N", "SOC118", "SOC145",
                    "SOC146", "SOC160", "SOC179A", "URBANST33", "URBANST103C",
                    "URBANST108H", "URBANST115", "URBANST123A", "URBANST125",
                    "URBANST126A", "URBANST127A", "URBANST127B", "URBANST127C",
                    "URBANST131", "URBANST132", "URBANST133", "URBANST141",
                    "URBANST148", "URBANST155", "URBANST163", "URBANST164",
                    "URBANST168", "URBANST169", "URBANST169B", "URBANST171",
                    "URBANST172A", "URBANST173", "URBANST183", "URBANST190A",
                ], source=SRC,
            ),
        ], source=SRC,
    )

    urban_sustainability = RequirementRule(
        label="Pathway: Urban Sustainability", kind="all",
        sub_rules=[
            RequirementRule(label="Required Course", kind="all", courses=["EARTHSYS112"], source=SRC),
            RequirementRule(
                label="Elective: Environmental Sustainability", kind="at_least_n", n=1,
                courses=[
                    "BIO81", "CEE41Q", "CEE64", "CEE100", "CEE107A", "CEE172", "CEE173",
                    "CEE176A", "CEE178", "CEE243", "CEE265F", "CEE308", "EARTHSYS10",
                    "EARTHSYS101", "EARTHSYS104", "ECON17N", "ECON155", "ENGR90",
                    "OSPSANTG29",
                ], source=SRC,
            ),
            RequirementRule(
                label="Elective: Social Sustainability", kind="at_least_n", n=1,
                courses=[
                    "AMSTUD154D", "ANTHRO167", "ENVRES221", "ENVRES223", "HISTORY200B",
                    "PEDS150", "PUBLPOL153", "SINY122", "SOC3", "URBANST103C",
                    "URBANST108H", "URBANST125", "URBANST155", "URBANST155A",
                    "URBANST156A", "URBANST163", "URBANST165", "URBANST169",
                ], source=SRC,
            ),
            RequirementRule(
                label="Elective: Project-Based Courses", kind="at_least_n", n=1,
                courses=[
                    "CEE124", "URBANST125", "URBANST164", "URBANST165", "URBANST171",
                    "URBANST172A", "URBANST181", "URBANST183",
                ], source=SRC,
            ),
        ], source=SRC,
    )

    self_designed_placeholder = make_placeholder_course(
        "PLACEHOLDER:URBST-BA-SELF-DESIGNED-PATHWAY",
        "20 units in a self-designed area of urban study, approved by the "
        "Director of Urban Studies and an Academic-Council-member advisor "
        "(bulletin gives no enumerated course list for this option)",
    )
    self_designed = RequirementRule(
        label="Pathway: Self-Designed", kind="all",
        courses=[self_designed_placeholder.key], source=SRC,
    )

    depth = RequirementRule(
        label="Urban Studies Pathways (at least 1 of 6)", kind="at_least_n", n=1,
        sub_rules=[global_urban, race_ethnicity, urban_education, urban_society,
                   urban_sustainability, self_designed],
        source=SRC,
    )

    wim_capstone = RequirementRule(
        label="Writing in the Major / Capstone (Junior + Senior Seminar)", kind="all",
        courses=["URBANST202A", "URBANST203"], source=SRC,
    )

    return [core, skills, community_engaged_learning, depth, wim_capstone]


def urbst_ba_placeholder_courses() -> dict[str, str]:
    ph = make_placeholder_course(
        "PLACEHOLDER:URBST-BA-SELF-DESIGNED-PATHWAY",
        "20 units in a self-designed area of urban study, approved by the "
        "Director of Urban Studies and an Academic-Council-member advisor",
    )
    return {ph.key: ph.description}


# ---------------------------------------------------------------------------
# Registry -- only codes with a build_and_solve()-verified working fix.
# MGTSC-MS was initially excluded (see mgtsc_ms_manual_rules()'s docstring)
# because the entire MS&E subject was absent from data/courses_snapshot.json
# -- since fixed by fetching that department directly, so it's included here.
#
# STATS-MS and URBST-BA are wrapped in a lambda here rather than registered
# directly: each uses a PLACEHOLDER course but (unlike the ee_bs/intlr_ba/
# envse_bs functions in manual_fix_batchB.py) returns a plain rules list,
# with the placeholder dict only available from a SEPARATE getter function
# (stats_ms_placeholder_courses / urbst_ba_placeholder_courses). Left as two
# separate functions each rather than retrofitted into one tuple-returning
# function (to avoid touching the already-verified rule-building code) --
# the lambda just bundles them into the (rules, placeholders) tuple contract
# every other MANUAL_FIXES entry and build_program_from_sample() use. Without
# this, the placeholder key would never be merged into the course dict, and
# the solver would report it (and, transitively, the whole model) INFEASIBLE
# instead of OPTIMAL -- confirmed this was happening before this fix.
# ---------------------------------------------------------------------------
MANUAL_FIXES = {
    "ME-MS": me_ms_manual_rules,
    "STATS-MS": lambda: (stats_ms_manual_rules(), stats_ms_placeholder_courses()),
    "URBST-BA": lambda: (urbst_ba_manual_rules(), urbst_ba_placeholder_courses()),
    "MGTSC-MS": mgtsc_ms_manual_rules,
}
