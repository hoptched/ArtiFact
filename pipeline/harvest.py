"""D1: harvest the AIC corpus into data/raw/artworks.jsonl.

Three API facts shape this module.

**A search query returns at most its first 1,000 results.** Asking for a
deeper offset returns HTTP 403 "You have requested too many results", so a
filtered query cannot be used to walk a corpus of tens of thousands. The
plain /artworks listing endpoint has no such cap — it pages to the end of
all ~133k works — but it cannot filter. So we walk the listing and apply
the corpus predicate ourselves, one page at a time.

**Pages hold at most 100 records**, making a full walk ~1,330 requests.

**Anonymous callers get 60 requests/minute.** Every request goes through
`_get`, which paces itself and retries on 429 and 5xx. A full harvest is
therefore a ~25 minute unattended run.

We deliberately do *not* filter by artwork type here. Type is the one
corpus choice still open (prints are AIC's largest public-domain category
but sit far from WikiArt's painting domain), and keeping every type in the
raw file means changing that decision is a D2 re-run, not another 25
minutes of API calls.

Resumable by design: the last completed page lives in a state file, so a
re-run after a crash resumes from there and a re-run after success is a
no-op.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pipeline.config import Config

LIST_URL = "https://api.artic.edu/api/v1/artworks"
PAGE_SIZE = 100          # the endpoint's ceiling; 200 returns 403
MIN_REQUEST_INTERVAL = 1.1   # 60/min anonymous, with headroom
MAX_RETRIES = 5

FIELDS = [
    "id", "title", "artist_title", "artist_display",
    "date_start", "date_end", "date_display",
    "place_of_origin", "style_title", "style_titles",
    "classification_title", "classification_titles",
    "artwork_type_title", "medium_display", "dimensions",
    "department_title", "credit_line", "image_id", "is_public_domain",
]


class RateLimiter:
    """Spaces requests so the anonymous 60/min ceiling is never hit."""

    def __init__(self, interval: float = MIN_REQUEST_INTERVAL) -> None:
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.monotonic()


@dataclass
class HarvestState:
    """How far through the listing we got."""

    path: Path
    last_page: int = 0
    total_pages: int = 0

    @classmethod
    def load(cls, path: Path) -> HarvestState:
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text())
        return cls(path=path, last_page=raw.get("last_page", 0),
                   total_pages=raw.get("total_pages", 0))

    def save(self) -> None:
        self.path.write_text(json.dumps(
            {"last_page": self.last_page, "total_pages": self.total_pages},
            indent=2))


def _get(client: httpx.Client, limiter: RateLimiter, page: int) -> dict[str, Any]:
    """One paced, retried page fetch. Raises if the API keeps refusing."""
    params = {"page": page, "limit": PAGE_SIZE, "fields": ",".join(FIELDS)}
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            response = client.get(LIST_URL, params=params, timeout=30.0)
        except httpx.TransportError as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    network error ({exc}); retrying")
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            wait = 2 ** attempt
            print(f"    HTTP {response.status_code} on page {page}; "
                  f"backing off {wait}s")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"gave up on page {page} after {MAX_RETRIES} attempts")


def keep(record: dict[str, Any]) -> bool:
    """The corpus predicate.

    Public domain and an image are hard requirements: without both, a work
    cannot legally or practically appear on the site. A parseable date is
    required because period is one of the three browsing axes. Artwork type
    is *not* filtered here — see the module docstring.
    """
    return bool(record.get("is_public_domain")
                and record.get("image_id")
                and record.get("date_start") is not None)


def main() -> None:
    config = Config.load()
    config.paths.ensure()

    out_path = config.paths.raw / "artworks.jsonl"
    state = HarvestState.load(config.paths.interim / "harvest_state.json")

    seen: set[int] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                if line.strip():
                    seen.add(json.loads(line)["id"])
        print(f"{len(seen):,} works already harvested, resuming after page "
              f"{state.last_page}\n")

    headers = {"AIC-User-Agent": "Minerva personal project"}
    written = 0
    kinds: Counter[str] = Counter()

    with httpx.Client(headers=headers) as client:
        limiter = RateLimiter()
        page = state.last_page + 1

        with out_path.open("a") as fh:
            while True:
                payload = _get(client, limiter, page)
                rows = payload.get("data", [])
                pagination = payload["pagination"]

                if not state.total_pages:
                    state.total_pages = pagination["total_pages"]
                    minutes = state.total_pages * MIN_REQUEST_INTERVAL / 60
                    print(f"{pagination['total']:,} works across "
                          f"{state.total_pages:,} pages "
                          f"(~{minutes:.0f} min at 60 req/min)\n")

                if not rows:
                    break

                for record in rows:
                    if not keep(record) or record["id"] in seen:
                        continue
                    seen.add(record["id"])
                    kinds[record.get("artwork_type_title") or "Unknown"] += 1
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

                fh.flush()
                state.last_page = page
                state.save()

                if page % 25 == 0 or page == state.total_pages:
                    pct = 100 * page / state.total_pages
                    print(f"  page {page:,}/{state.total_pages:,} "
                          f"({pct:4.1f}%)  kept {len(seen):,}")

                if page >= state.total_pages:
                    break
                page += 1

    print(f"\nD1: wrote {written:,} new works, {len(seen):,} total")
    if written == 0 and seen:
        print("Re-run was a no-op, as intended.")
    elif kinds:
        print("\nTop artwork types kept:")
        for kind, count in kinds.most_common(10):
            print(f"  {count:>7,}  {kind}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
