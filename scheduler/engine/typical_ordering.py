"""
"Typical ordering" detection: course sequences whose titles are numbered
with Roman numerals (e.g. "Inorganic Chemistry I" / "Inorganic Chemistry
II") get a SOFT nudge from the solver to schedule the lower numeral before
the higher one -- never a hard prerequisite. Prompted directly by user
feedback on a CHEM-BS validation run: CHEM153 ("Inorganic Chemistry II")
doesn't actually require CHEM151 ("Inorganic Chemistry I") as a bulletin
prerequisite, but taking II before I "wouldn't make much sense" -- exactly
the gap between "enforced prerequisite" (engine/course_data.py's
is_high_confidence_prerequisite, extracted from real "Prerequisite:" text)
and "commonsense sequencing" that this module fills.

Deliberately keyed on the Roman numeral in the TITLE, never the course
NUMBER -- confirmed directly by the same user feedback: MATH51/52/53 are
numbered sequentially but 52 and 53 can be taken in either order with no
consequence, so "lower course number first" would be a wrong general rule.
A title-based "I before II before III" reading has no such
counterexample in this catalog (a sequence's Roman-numeral suffix is
specifically how Stanford titles denote "part N of this sequence").
"""
from __future__ import annotations

import re
from collections import defaultdict

from .models import Course

# Only I through X -- real course sequences at Stanford never go past III
# or IV in practice; capping here avoids false positives on unrelated
# titles that happen to end in a capital letter run (there are none in the
# roman-numeral alphabet past X worth worrying about for course titles).
_ROMAN_VALUES = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}
_TITLE_ROMAN_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)\s+(?P<numeral>I|II|III|IV|V|VI|VII|VIII|IX|X)$"
)

# Hand-curated pairs for real sequences the title-based Roman-numeral
# detector above structurally can't see, because the first course's actual
# bulletin title doesn't follow the "Base Name" + " " + numeral pattern at
# all. Real case that prompted this: CHEM171's actual title is "Linking the
# Molecular to the Macroscopic: Foundations of Physical Chemistry" -- no
# Roman numeral, no shared base string with CHEM173 ("Physical Chemistry
# II") -- even though CHEM171 is the conventional first course of the same
# physical-chemistry sequence that CHEM173 -> CHEM175 ("Physical Chemistry
# III") already auto-pairs correctly. Same "manual override layered on top
# of an automatic heuristic" precedent as engine/manual_fix_batch*.py.
#
# Kept as a SOFT nudge, same as every other pair here -- deliberately not a
# hard prerequisite, since CHEM175's actual bulletin prerequisite is "either
# CHEM 173 or CHEM 171" (an OR, already hard-enforced by
# course_data.high_confidence_prerequisite_groups()/solver.py): a student
# can genuinely enter this sequence at CHEM173 directly and skip CHEM171
# entirely. This pair only says "if you take CHEM171 at all, it should come
# before CHEM173" -- it doesn't require taking CHEM171.
_MANUAL_EXTRA_PAIRS: tuple[tuple[str, str], ...] = (
    ("CHEM171", "CHEM173"),
)


def _split_title(title: str) -> tuple[str, int] | None:
    """('Inorganic Chemistry', 2) for 'Inorganic Chemistry II', else None."""
    m = _TITLE_ROMAN_SUFFIX_RE.match(title.strip())
    if not m:
        return None
    return m.group("base").strip().casefold(), _ROMAN_VALUES[m.group("numeral")]


def find_typical_order_pairs(courses: dict[str, Course]) -> list[tuple[str, str]]:
    """(earlier_key, later_key) pairs for consecutive Roman-numeral steps
    within the same subject and same title base -- e.g. CHEM151 ("Inorganic
    Chemistry I") -> CHEM153 ("Inorganic Chemistry II"). Only consecutive
    numeral steps are paired (I->II, II->III, ...), not every combination,
    since that's what "prioritize ordering them" means for a sequence --
    no transitive I->III pair is added when II exists between them (the
    I->II and II->III pairs already imply it via the solver's soft nudges,
    without double-penalizing the same underlying gap)."""
    groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for key, course in courses.items():
        split = _split_title(course.title)
        if split is None:
            continue
        base, numeral = split
        groups[(course.subject, base)].append((numeral, key))

    pairs: list[tuple[str, str]] = []
    for (_subject, _base), members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda pair: pair[0])
        # De-duplicate identical numerals (shouldn't happen in real data,
        # but don't pair a numeral with itself if it somehow does) and
        # only pair genuinely consecutive numeral VALUES -- if "I" and
        # "III" exist but "II" doesn't, still pair them (nothing else
        # would), but never re-pair "I"->"II" twice.
        seen_numerals = []
        for numeral, key in members:
            if seen_numerals and seen_numerals[-1][0] == numeral:
                continue
            seen_numerals.append((numeral, key))
        for (_, early_key), (_, late_key) in zip(seen_numerals, seen_numerals[1:]):
            pairs.append((early_key, late_key))

    # Layer the hand-curated pairs on top -- only if both courses actually
    # exist in this run's snapshot, and only if the automatic detector
    # above didn't already find the exact same pair (it won't, by
    # construction, since these exist specifically for titles that don't
    # match the Roman-numeral pattern -- but de-duplicating is cheap and
    # future-proofs against a snapshot title change making one coincide).
    existing = set(pairs)
    for early_key, late_key in _MANUAL_EXTRA_PAIRS:
        if early_key in courses and late_key in courses and (early_key, late_key) not in existing:
            pairs.append((early_key, late_key))
            existing.add((early_key, late_key))
    return pairs
