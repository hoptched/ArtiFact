"""D2: filter the raw harvest to the painting-like corpus and normalize
the three browsing axes.

Reads data/raw/artworks.jsonl (all ~59k public-domain works, every type)
and writes data/processed/corpus.jsonl plus a taxonomy sidecar. The type
filter lives here rather than in D1 so the corpus decision stays cheap to
revisit: change painting_like_types in config.yaml and re-run this, no API
walk required.

Two of the three axes come straight from museum metadata and are settled
here. The third, style, cannot be: AIC's own style_title is centuries and
cultures at 19% coverage, so this stage only fixes the *label set* the D4
head will predict into. Every work leaves here with style unset.

Run:  .venv/bin/python -m pipeline.normalize [--sample N]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import Any

from pipeline.config import Config
from pipeline.taxonomy import (
    PLACE_TO_COUNTRY, STYLE_LABELS, UNKNOWN_PLACES, VAGUE_PLACES,
    country_for, date_precision, period_bin, period_bins, period_label,
)


def unmapped_places(raw_places: Counter[str]) -> list[str]:
    """Place values the lookup does not cover. Should always be empty:
    every value in the corpus was mapped by hand, so a non-empty result
    means the corpus changed and the table needs revisiting."""
    known = set(PLACE_TO_COUNTRY) | set(VAGUE_PLACES) | UNKNOWN_PLACES
    return sorted(p for p in raw_places if p not in known)


def normalize(record: dict[str, Any], config: Config) -> dict[str, Any] | None:
    """One raw AIC record into one corpus row, or None if it drops out."""
    bins = period_bins(record.get("date_start"), record.get("date_end"),
                       config.corpus.max_span_years)
    if not bins:
        return None
    bin_start = period_bin(record["date_start"], record["date_end"],
                           config.corpus.max_span_years)

    place = record.get("place_of_origin")
    unknown = config.taxonomy.unknown_label
    return {
        "id": record["id"],
        "title": record.get("title"),
        "artist": record.get("artist_title") or unknown,
        # artist_title flattens "After Raphael" to "Raphael"; the display
        # string is the only place the attribution survives intact.
        "artist_display": record.get("artist_display"),
        "image_id": record["image_id"],
        # Period: observed
        "date_start": record["date_start"],
        "date_end": record["date_end"],
        "date_display": record.get("date_display"),
        # The midpoint bin is the one label to show; period_bins is what a
        # filter matches on, so a work the museum dated only to a century
        # turns up under both halves instead of inflating the first.
        "period_bin": bin_start,
        "period": period_label(bin_start),
        "period_bins": bins,
        "date_precision": date_precision(record["date_start"],
                                         record["date_end"]),
        # Place: observed, normalized
        "place_raw": place,
        "country": country_for(place, unknown),
        # Style: inferred later by D4/D5. Present so the schema is stable
        # and the site can tell observed from predicted from missing.
        "style": None,
        "style_confidence": None,
        "style_source": "pending",
        # Kept for D3 filtering and the detail page
        "artwork_type": record.get("artwork_type_title"),
        "medium": record.get("medium_display"),
        "aic_style_title": record.get("style_title"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0,
                        help="print N random normalized rows for eyeballing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Config.load()
    config.paths.ensure()
    keep_types = set(config.corpus.painting_like_types)

    raw_path = config.paths.raw / "artworks.jsonl"
    out_path = config.paths.processed / "corpus.jsonl"

    seen = kept = 0
    dropped_type = dropped_date = 0
    raw_places: Counter[str] = Counter()
    countries: Counter[str] = Counter()
    periods: Counter[int] = Counter()
    precisions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    with raw_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            record = json.loads(line)

            if config.corpus.painting_like_only and \
                    record.get("artwork_type_title") not in keep_types:
                dropped_type += 1
                continue

            raw_places[record.get("place_of_origin")] += 1
            row = normalize(record, config)
            if row is None:
                dropped_date += 1
                continue

            rows.append(row)
            countries[row["country"]] += 1
            for b in row["period_bins"]:
                periods[b] += 1
            precisions[row["date_precision"]] += 1
            kept += 1
            if kept >= config.corpus.max_works:
                break

    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    taxonomy_path = config.paths.processed / "taxonomy.json"
    taxonomy_path.write_text(json.dumps({
        "period_scheme": config.taxonomy.period_scheme,
        "periods": [{"bin": b, "label": period_label(b), "count": periods[b]}
                    for b in sorted(periods)],
        "precision_levels": ["exact", "loose", "vague"],
        "countries": [{"name": c, "count": n}
                      for c, n in countries.most_common()],
        "styles": STYLE_LABELS,
        "unknown_label": config.taxonomy.unknown_label,
    }, indent=2, ensure_ascii=False))

    # --- coverage report, the D2 acceptance check ------------------------
    print(f"read {seen:,} raw works")
    print(f"  dropped, not painting-like : {dropped_type:,}")
    print(f"  dropped, date span > {config.corpus.max_span_years}y   : {dropped_date:,}")
    print(f"  kept                       : {kept:,}\n")

    missing = unmapped_places(raw_places)
    print("=== axis coverage ===")
    labelled = kept - countries[config.taxonomy.unknown_label]
    print(f"  period   {kept:>6,}  100.0%   ({len(periods)} bins)")
    print(f"  country  {labelled:>6,}  {100*labelled/kept:5.1f}%   "
          f"({len(countries)} values, "
          f"{countries[config.taxonomy.unknown_label]} unknown)")
    print(f"  style         0    0.0%   (D4 fills this; "
          f"{len(STYLE_LABELS)} labels defined)")
    print(f"\n  unmapped place values: {len(missing)}"
          + (f" -> {missing}" if missing else " (lookup is complete)"))

    print("\n=== date precision ===")
    for level in ("exact", "loose", "vague"):
        n = precisions[level]
        note = {"exact": "<=10y", "loose": "11-50y", "vague": "51-100y"}[level]
        print(f"  {level:<7} {note:<8} {n:>5,}  {100*n/kept:5.1f}%")

    print("\n=== periods (a work counts in every bin it overlaps) ===")
    for b in sorted(periods):
        bar = "#" * max(1, round(40 * periods[b] / max(periods.values())))
        print(f"  {period_label(b):<12} {periods[b]:>5,}  {bar}")

    print("\n=== top countries ===")
    for name, n in countries.most_common(12):
        print(f"  {n:>5,}  {name}")

    if args.sample:
        random.seed(args.seed)
        print(f"\n=== {args.sample} random rows ===")
        for row in random.sample(rows, min(args.sample, len(rows))):
            print(f"  {row['period']:<11} {row['country']:<18} "
                  f"{(row['artist'] or '?')[:24]:<24} "
                  f"{(row['title'] or '')[:40]}")
            print(f"              raw place: {row['place_raw']!r}  "
                  f"dates: {row['date_display']!r}")

    print(f"\nD2: {kept:,} works -> {out_path}")
    print(f"    taxonomy -> {taxonomy_path}")


if __name__ == "__main__":
    main()
