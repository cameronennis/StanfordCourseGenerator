"""
Deterministic, no-browser fetcher for bulletin.stanford.edu program pages.

Why this exists: the original scale-up approach (see bulletin_parser.py's
docstring) used Claude in Chrome to render each page, because the FIRST
naive attempt (plain HTTP fetch) appeared to return an unhydrated shell.
That turned out to be almost right but not quite: bulletin.stanford.edu is
actually server-rendered (Nuxt SSR) -- a plain `requests.get()` DOES return
the full page structure (every section, header, wrapper, AND/OR connector,
note). The only thing genuinely missing from the raw HTML is course
labels: a course reference renders server-side as a bare internal numeric
ID (e.g. "2085321"), and only gets swapped for "CS103 - Mathematical
Foundations of Computing" by client-side JS after hydration -- which is
exactly the "raw internal IDs" failure mode bulletin_parser.py's docstring
already warned about.

So the fix isn't rendering in a browser at all -- it's resolving those
numeric IDs ourselves. Each `https://bulletin.stanford.edu/courses/{id}`
page is ALSO plain-HTTP-fetchable and reliably exposes the code in
`<h1 class="heading-1 mb-2">` and the title in the following
`<h2 class="heading-5 mb-2">`. Courses repeat heavily across programs
(MATH19 alone appears in nearly every engineering major), so resolved IDs
are cached to disk (COURSE_ID_CACHE_PATH) and reused across the whole run
and future runs -- most of the ~355 program pages should need only a
handful of NEW id lookups by the time a large batch is underway.

Usage:
    python -m engine.fetch_bulletin CS-BS EE-BS ...
    python -m engine.fetch_bulletin --all-in-scope   # everything in
        data/bulletin_program_codes.json's in_scope_218 not already in samples/
"""
from __future__ import annotations

import html as html_entities
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
SAMPLES_DIR = ROOT / "samples"
COURSE_ID_CACHE_PATH = ROOT / "data" / "course_id_cache.json"
CODES_PATH = ROOT / "data" / "bulletin_program_codes.json"

PROGRAM_URL = "https://bulletin.stanford.edu/programs/{code}"
COURSE_URL = "https://bulletin.stanford.edu/courses/{id}"

_BARE_ID_RE = re.compile(r"^\d{5,8}$")
_H1_RE = re.compile(r'<h1[^>]*class="heading-1 mb-2"[^>]*>([^<]+)</h1>', re.IGNORECASE)
_H2_RE = re.compile(r'<h2[^>]*class="heading-5 mb-2"[^>]*>([^<]*)</h2>', re.IGNORECASE)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StanfordScheduleGenerator/1.0)"}

REQUEST_DELAY_SECONDS = 1.5  # deliberately slow -- see module docstring on the WAF incident

_WAF_MARKERS = ("Human Verification", "gokuProps", "awsWafCookieDomainList")


class WafBlockedError(Exception):
    """Raised the moment a response looks like an AWS WAF challenge page
    rather than real content -- distinct from an ordinary 404/timeout on
    purpose. This must STOP the batch, not get logged as one more
    per-item warning: a WAF block affects every subsequent request until
    it clears, so treating it as "this one course/program failed" would
    silently misreport a systemic block as ~200 unrelated-looking data
    problems. See the module docstring for how this was first hit (a
    16-way-concurrent burst) and why the fix is pacing, not retrying
    harder. This code never attempts to solve or work around the
    challenge itself."""


def _looks_like_waf_challenge(text: str) -> bool:
    return any(marker in text for marker in _WAF_MARKERS)


def load_course_id_cache() -> dict[str, str]:
    if COURSE_ID_CACHE_PATH.exists():
        return json.loads(COURSE_ID_CACHE_PATH.read_text())
    return {}


def save_course_id_cache(cache: dict[str, str]) -> None:
    COURSE_ID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COURSE_ID_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


UNRESOLVED_MARKER = "__UNRESOLVED__"  # cached value for a course id confirmed (via HTTP 404) to have no page --
# distinct from a transient failure (timeout, connection error, weird-but-maybe-real markup), which is
# deliberately left OUT of the cache so it gets retried on the next run instead of assumed dead forever.


def _fetch_course_label(course_id: str) -> tuple[str, str | None, str | None, bool]:
    """Fetch one course page and pull just the code/title out of it. Each
    course page is a full ~1-2MB Nuxt SSR page, but the <h1>/<h2> we need
    land in the first few KB, so stream the response and stop reading as
    soon as we've seen enough -- no need to pull the whole multi-megabyte
    page over the wire just for two tags.
    Returns (course_id, label_or_None, error_or_None, permanent_failure).
    `permanent_failure` is True only for a clean HTTP 404 -- some program
    pages reference old/renumbered/cross-listed course ids that genuinely
    no longer have a page, and without caching THAT fact too, every future
    run would re-fetch the exact same dead id and get the exact same 404,
    forever (this is what stalled real progress on ARCHA-MIN: most of its
    "unresolved" ids were dead ones, re-attempted every single run).
    Raises WafBlockedError (does NOT return it as an ordinary error) if
    the response looks like a bot-detection challenge rather than a real
    course page -- see WafBlockedError's docstring for why that
    distinction matters."""
    try:
        with requests.get(COURSE_URL.format(id=course_id), headers=HEADERS, timeout=15, stream=True) as resp:
            chunks = []
            total = 0
            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True):
                chunks.append(chunk)
                total += len(chunk)
                joined = "".join(chunks)
                if _looks_like_waf_challenge(joined):
                    raise WafBlockedError(f"course id {course_id!r}: got a WAF challenge page (HTTP {resp.status_code})")
                if _H1_RE.search(joined) and _H2_RE.search(joined):
                    break
                if total > 200_000:  # safety valve -- give up early rather than pulling the whole page
                    break
            html = "".join(chunks)
            status = resp.status_code
    except WafBlockedError:
        raise
    except requests.RequestException as e:
        return course_id, None, f"fetch failed ({e})", False
    m1, m2 = _H1_RE.search(html), _H2_RE.search(html)
    if not m1 or not m2:
        return course_id, None, f"couldn't find code/title markup on its page (HTTP {status})", status == 404
    # The <h1>/<h2> markup carries raw HTML entities (e.g. "MS&amp;E252",
    # "Master&#39;s Paper") -- unescape before caching, otherwise every
    # course whose title contains "&", an apostrophe, or a quote comes out
    # of the cache still HTML-encoded. That corruption is more than
    # cosmetic: bulletin_parser.py's COURSE_LINE_RE requires an uppercase
    # subject code immediately before the first digit, and "MS&amp;E252"
    # fails that match (the lowercase "amp;" isn't a valid subject
    # character) -- so an affected course silently ends whatever course
    # list it's sitting in, dropping every real option after it. Confirmed
    # happening for real in CS-BS's own "General Electives"/"Track
    # Electives" lists, where "MS&amp;E252"/"MS&amp;E355" (and CS-BS's
    # General Electives' own "MS&amp;E244"/"MS&amp;E265"/"MS&amp;E296")
    # cut the list short right where they sat.
    code = html_entities.unescape(m1.group(1).strip())
    title = html_entities.unescape(m2.group(1).strip())
    return course_id, (f"{code} - {title}" if title else code), None, False


def resolve_course_ids(
    course_ids: set[str], cache: dict[str, str], warnings: list[str], deadline: float | None = None
) -> bool:
    """Resolve every not-yet-cached id in `course_ids`, ONE AT A TIME, with
    a fixed delay between requests.

    This used to run 16-way concurrent -- which tripped
    bulletin.stanford.edu's AWS WAF bot-protection within a few hundred
    requests (it started serving a "Human Verification" challenge page
    instead of course pages). That's a hard stop: this codebase will not
    attempt to solve or bypass a CAPTCHA/WAF challenge, no matter how the
    request volume is justified. The fix is to not trigger it in the first
    place -- strictly sequential, paced requests, indistinguishable from a
    slow, well-behaved crawler. It's slower wall-clock time, but that costs
    nothing in tokens -- but IS bounded by how long a single sandboxed
    shell call is allowed to run (see main()'s docstring), which turned
    out to be well under a minute of *network-bound* work at a time in
    practice. So this saves `cache` to disk after every single successful
    resolution (not just once per program) and, if `deadline` (an absolute
    time.time() value) is given and gets hit mid-list, stops cleanly and
    returns False -- meaning the caller should NOT proceed to treat this
    program as fully resolved (some ids may still be bare/unresolved this
    run; a later run picks them up from the cache automatically). Returns
    True if every id in `course_ids` got resolved (or permanently failed
    with a warning) before the deadline.

    Never raises for an ordinary per-item failure (warns and leaves that
    one id unresolved instead -- see fetch_and_save_program's docstring on
    why a silent guess would be worse), but DOES propagate WafBlockedError
    so the caller stops the whole batch rather than grinding through the
    rest looking blocked."""
    todo = [c for c in course_ids if c not in cache]
    for cid in todo:
        if deadline is not None and time.time() > deadline:
            return False
        course_id, label, error, permanent = _fetch_course_label(cid)
        if label is not None:
            cache[course_id] = label
            save_course_id_cache(cache)
        else:
            warnings.append(f"course id {course_id!r}: {error}")
            if permanent:
                # Confirmed dead (HTTP 404) -- cache that fact so every
                # future run doesn't re-fetch and re-fail on the exact
                # same id forever (see _fetch_course_label's docstring).
                cache[course_id] = UNRESOLVED_MARKER
                save_course_id_cache(cache)
        time.sleep(REQUEST_DELAY_SECONDS)
    return True


def fetch_program_html(code: str) -> str:
    url = PROGRAM_URL.format(code=urllib.parse.quote(code))
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if _looks_like_waf_challenge(resp.text[:5000]):
        raise WafBlockedError(f"program {code!r}: got a WAF challenge page (HTTP {resp.status_code})")
    resp.raise_for_status()
    return resp.text


def html_to_lines(html: str) -> list[str]:
    """Strip a fetched program page down to the same plain-text line
    sequence bulletin_parser.py expects (matching what Claude in Chrome's
    get_page_text used to produce from the rendered DOM)."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("article") or soup.find("main") or soup.body
    text = main.get_text("\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # The DOM sometimes splits a header phrase across two text nodes, e.g.
    # "Complete at least 2 of the following" / "Courses:" landing on
    # consecutive lines purely due to a <br> or wrapping <span> boundary
    # -- rejoin so the parser's header regexes (which match a single full
    # line) still see one line.
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 1 < len(lines)
            and lines[i + 1] in ("Courses:", "Course Sets:")
            and line.endswith("following")
        ):
            merged.append(f"{line} {lines[i + 1]}")
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def resolve_course_ids_in_lines(
    lines: list[str], cache: dict[str, str], warnings: list[str], deadline: float | None = None
) -> tuple[list[str], bool]:
    ids_in_lines = {l for l in lines if _BARE_ID_RE.match(l)}
    complete = resolve_course_ids(ids_in_lines, cache, warnings, deadline=deadline)

    def substitute(line: str) -> str:
        if not _BARE_ID_RE.match(line):
            return line
        cached = cache.get(line)
        if cached is None or cached == UNRESOLVED_MARKER:
            # Leave the bare id in place rather than writing the sentinel
            # into the sample -- bulletin_parser.py's COURSE_LINE_RE won't
            # match a bare number, so this shows up as one missing course
            # in that slot (already-flagged via a warning here), not a
            # corrupted/misattributed course list.
            return line
        return cached

    return [substitute(l) for l in lines], complete


def fetch_and_save_program(code: str, cache: dict[str, str], deadline: float | None = None) -> dict:
    """Fetch one program page, resolve its course ids, write
    samples/{code}.txt -- but ONLY if every course id in it resolved
    before `deadline`. A program hitting the deadline partway through
    comes back with ok=False, incomplete=True and nothing written to
    samples/ -- the next run refetches its program page (cheap, one
    request) and, thanks to the cache, only needs to resolve whatever
    ids didn't finish last time. Writing a .txt with bare unresolved
    numeric ids sitting in it would look like a silent data-loss bug
    identical to the ones already fixed in bulletin_parser.py, so this
    never does that.

    Returns a small summary dict for ordinary HTTP/parse problems (so a
    batch run over ~200 codes doesn't die on one flaky page) -- but
    WafBlockedError is deliberately NOT caught here; it propagates up to
    main(), which stops the whole batch. See WafBlockedError's docstring."""
    warnings: list[str] = []
    try:
        html = fetch_program_html(code)
    except requests.RequestException as e:
        return {"code": code, "ok": False, "incomplete": False, "error": f"program page fetch failed: {e}"}

    lines = html_to_lines(html)
    if not lines or not any(l == "Completion requirement" for l in lines):
        return {
            "code": code, "ok": False, "incomplete": False,
            "error": "no 'Completion requirement' markers found -- page shape unexpected",
        }

    resolved, complete = resolve_course_ids_in_lines(lines, cache, warnings, deadline=deadline)
    if not complete:
        return {
            "code": code, "ok": False, "incomplete": True,
            "error": "ran out of time budget mid-resolution -- will resume next run",
        }
    out_path = SAMPLES_DIR / f"{code}.txt"
    out_path.write_text("\n".join(resolved) + "\n")
    return {"code": code, "ok": True, "incomplete": False, "path": str(out_path), "lines": len(resolved), "resolve_warnings": warnings}


def main(argv: list[str]) -> None:
    """Each run of this script has to fit inside ONE sandboxed shell call
    (max ~10 minutes wall clock in this environment -- background/nohup'd
    processes don't survive past the call that started them here, so
    there's no way to "kick this off and check back later" within a
    single process). Course-id resolution is the slow part (network
    latency dominates, not the deliberate pacing), and a program with many
    not-yet-cached courses can eat minutes on its own -- so this self-
    limits to a time budget and stops CLEANLY (not via WafBlockedError)
    once it's used up, saving everything done so far. Since already-
    fetched samples/*.txt and cached course ids are always skipped on the
    next invocation, re-running the exact same `--all-in-scope` command
    repeatedly just keeps making forward progress until nothing's left --
    that's the intended usage, across as many calls as it takes.

    Empirically the host caps a single call well under the 10-minute
    figure the shell tool advertises (observed ~180s) -- so the default
    budget here is deliberately conservative, leaving margin for the
    program-page fetch itself plus whatever's in flight when the deadline
    is checked (mid-course-lookup progress isn't lost either way, since
    resolve_course_ids saves the cache after every single resolution --
    but leaving margin means the LAST resolved course before a stop is
    still followed by a clean, logged stop rather than a hard kill)."""
    time_budget_seconds = 150.0
    for arg in argv:
        if arg.startswith("--time-budget="):
            time_budget_seconds = float(arg.split("=", 1)[1])
    argv = [a for a in argv if not a.startswith("--time-budget=")]
    start = time.time()
    deadline = start + time_budget_seconds

    cache = load_course_id_cache()
    if argv and argv[0] == "--all-in-scope":
        codes_data = json.loads(CODES_PATH.read_text())
        already_done = {p.stem for p in SAMPLES_DIR.glob("*.txt")}
        codes = [c for c in codes_data["in_scope_218"] if c not in already_done]
    else:
        codes = argv

    print(f"{len(codes)} codes queued this run (time budget {time_budget_seconds:.0f}s)", file=sys.stderr)
    results = []
    for i, code in enumerate(codes, 1):
        if time.time() > deadline:
            print(
                f"\nTime budget used up after {i - 1}/{len(codes)} codes this run -- "
                "stopping cleanly. Re-run the same command to continue; already-fetched "
                "programs and cached course ids are skipped automatically.",
                file=sys.stderr,
            )
            break
        print(f"[{i}/{len(codes)}] fetching {code} ...", file=sys.stderr)
        try:
            result = fetch_and_save_program(code, cache, deadline=deadline)
        except WafBlockedError as e:
            save_course_id_cache(cache)
            print(f"\nSTOPPING: {e}", file=sys.stderr)
            print(
                f"Hit a WAF challenge page after {i - 1} programs this run "
                f"({len(results)} completed, {len(cache)} course ids cached so far). "
                "Not retrying or working around it -- wait a few minutes for it to "
                "clear, then re-run with --all-in-scope; already-fetched programs "
                "and cached course ids are skipped automatically.",
                file=sys.stderr,
            )
            sys.exit(2)
        results.append(result)
        if result["incomplete"]:
            print(f"  ran out of time mid-{code} -- stopping (resumes next run)", file=sys.stderr)
            break
        if not result["ok"]:
            print(f"  FAILED: {result['error']}", file=sys.stderr)
        elif result["resolve_warnings"]:
            for w in result["resolve_warnings"]:
                print(f"  [warn] {w}", file=sys.stderr)
        time.sleep(REQUEST_DELAY_SECONDS)

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"] and not r["incomplete"]]
    print(f"\n{len(ok)} succeeded, {len(failed)} failed this run", file=sys.stderr)
    for r in failed:
        print(f"  FAILED {r['code']}: {r['error']}", file=sys.stderr)
    codes_data = json.loads(CODES_PATH.read_text())
    remaining = len(codes_data["in_scope_218"]) - len({p.stem for p in SAMPLES_DIR.glob("*.txt")} & set(codes_data["in_scope_218"]))
    print(f"course id cache: {len(cache)} entries; {remaining} in-scope programs still remaining", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
