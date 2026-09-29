"""D5: predict style over the AIC corpus and build the site's data bundle.

    .venv/bin/python -m pipeline.build_site_data

Everything the website needs is computed here and frozen to disk. There is
no model and no backend at request time; the site reads JSON and nothing
else.

Ships precomputed neighbours rather than raw embeddings: 9,101 x 512
floats is ~19 MB in the browser, while 9,101 rows of twenty ids is a small
fraction of that, and the browser never needs the vectors for anything
else.

Outputs to web/public/data/:
    works.json      one row per work: metadata, predicted style, 2D point
    neighbors.json  id -> the twenty most visually similar ids
    facets.json     the filter vocabularies and their counts
"""

from __future__ import annotations

import gzip
import json
from collections import Counter

import numpy as np

from pipeline.config import Config
from pipeline import layout
from pipeline.spatial_order import hilbert_order, locality
from pipeline.taxonomy import OUTSIDE_TAXONOMY, period_label, style_applies

KNN_CHUNK = 512          # rows of the similarity matrix held at once


def load(config: Config):
    import joblib
    P = config.paths.processed
    manifest = json.loads((P / "manifest.json").read_text())
    if manifest["fingerprint"] != config.model.fingerprint:
        raise SystemExit(f"AIC vectors are {manifest['fingerprint']}, "
                         f"config.yaml is {config.model.fingerprint}")
    head = joblib.load(P / "style_head.joblib")
    if head["fingerprint"] != config.model.fingerprint:
        raise SystemExit("the style head was trained in a different vector "
                         "space than these embeddings")
    vecs = np.load(P / "embeddings.npy")
    ids = json.loads((P / "ids.json").read_text())
    # One representative colour per work, so the zoomed-out map can render
    # 9,101 tiles before a single image has loaded. Optional: the bundle
    # still builds without it, the map is just grey until thumbnails land.
    colors = np.load(P / "colors.npy") if (P / "colors.npy").exists() else None
    if colors is None:
        print("  no colors.npy — re-run D3 to get the mosaic colours\n")
    # Aspect ratio, so a tile is the right shape before any of its pixels
    # have arrived. Without it every work is drawn square, which for a
    # collection this full of hanging scrolls is simply wrong.
    dims = np.load(P / "dims.npy") if (P / "dims.npy").exists() else None
    if dims is None:
        print("  no dims.npy — re-run D3; tiles will be square until then\n")
    corpus = {r["id"]: r for r in
              (json.loads(l) for l in (P / "corpus.jsonl").open())}
    return vecs, ids, corpus, head, colors, dims


def top_k_neighbors(vecs: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k most similar rows for every row.

    Vectors are unit length from D3, so the dot product is cosine. Chunked
    because the full 9,101^2 similarity matrix is 331 MB and there is no
    reason to hold it all at once.
    """
    n = len(vecs)
    out = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, KNN_CHUNK):
        stop = min(start + KNN_CHUNK, n)
        sims = vecs[start:stop] @ vecs.T
        for r in range(stop - start):
            sims[r, start + r] = -np.inf          # never a neighbour of itself
        out[start:stop] = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(stop - start):             # argpartition is unordered
            row = out[start + r]
            out[start + r] = row[np.argsort(-sims[r, row])]
    return out


def shown_date(row: dict) -> str:
    """A year, or a span of years, and nothing else.

    The museum writes dates as prose, and the prose takes every shape it
    can: "1904", "c. 1797", "1770-1810", "1540/50", "n.d." for one work
    in six. Read down a column of those, nothing lines up and the same
    fact appears in four costumes.

    So the year range behind the wording is used instead, which every
    work has: a single year when the two ends agree, and start-end when
    they differ. What is lost is the museum's hedging, the "c." and the
    slash, which the range already carries by being a range.
    """
    start, end = row.get("date_start"), row.get("date_end")
    if start is None and end is None:
        return (row.get("date_display") or "").strip() or "date unknown"
    if start is None or end is None:
        return str(start if start is not None else end)
    if start == end:
        return str(start)
    return f"{start}-{end}"


def main() -> None:
    config = Config.load()
    config.paths.ensure()
    vecs, ids, corpus, head, colors, dims = load(config)
    model, labels = head["model"], head["labels"]
    print(f"{len(ids):,} vectors, {len(corpus):,} corpus rows, "
          f"{len(labels)} style labels\n")

    # --- style prediction ------------------------------------------------
    proba = model.predict_proba(vecs)

    # A work cannot belong to a movement that had not started yet. Softmax
    # has to answer with something, and on a corpus that is mostly pre-1900
    # under a taxonomy that reaches into the 1960s, what it answered with
    # was often impossible: before this, every work called Surrealism,
    # Social Realism, Regionalism or Photorealism predated the movement.
    #
    # Compared against the latest year the work could have been made, so a
    # label is ruled out only when even that is too early. A work with no
    # date is never gated, and if a gate somehow left nothing, the row is
    # restored rather than guessed at.
    from pipeline.taxonomy_v2 import STYLE_EARLIEST
    gate = np.array([STYLE_EARLIEST.get(l, -10_000) for l in labels])
    # corpus is keyed by artwork id; ids is in vector order.
    ends = np.array([
        (corpus.get(i, {}).get("date_end") if
         corpus.get(i, {}).get("date_end") is not None else 10_000)
        for i in ids
    ], dtype=float)
    impossible = ends[:, None] < gate[None, :]
    emptied = impossible.all(axis=1)
    impossible[emptied] = False
    moved = int((impossible[np.arange(len(proba)), proba.argmax(axis=1)]).sum())
    proba = np.where(impossible, 0.0, proba)
    total = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, total, out=proba, where=total > 0)
    print(f"  era gate: {moved:,} works were given a movement that had not "
          f"started when they were made")

    best = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    runner_up = np.partition(proba, -2, axis=1)[:, -2]
    print("=== predicted style confidence ===")
    for lo in (0.2, 0.3, 0.4, 0.5, 0.7):
        n = int((confidence >= lo).sum())
        print(f"  >= {lo:.1f}   {n:>6,}  {100*n/len(ids):5.1f}%")
    print(f"  median {np.median(confidence):.3f}, "
          f"mean margin over runner-up {np.mean(confidence - runner_up):.3f}")

    print("\n=== predicted style distribution ===")
    counts = Counter(labels[i] for i in best)
    widest = max(counts.values())
    for label, n in counts.most_common():
        print(f"  {label:<20} {n:>5,}  {'#' * round(36 * n / widest)}")
    unused = [l for l in labels if l not in counts]
    if unused:
        print(f"  never predicted: {', '.join(unused)}")

    # --- neighbours ------------------------------------------------------
    k = config.site.neighbors_per_work
    print(f"\ncomputing top-{k} neighbours over {len(vecs):,} vectors...")
    nn = top_k_neighbors(vecs, k)

    # --- 2D map ----------------------------------------------------------
    print("fitting UMAP (this is the slow part)...")
    import umap
    xy = umap.UMAP(n_components=2, metric="cosine", random_state=0,
                   n_neighbors=25, min_dist=0.12).fit_transform(vecs)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))   # 0..1

    # --- assemble --------------------------------------------------------
    out = config.paths.site_data
    works, neighbors = [], {}
    periods, countries, styles = Counter(), Counter(), Counter()

    suppressed = 0
    for i, artwork_id in enumerate(ids):
        r = corpus[artwork_id]
        # The taxonomy is 16 European movements plus Ukiyo-e. Outside that
        # domain the head is confidently wrong rather than uncertain, so
        # the label is withheld instead of shown with a caveat.
        in_domain = style_applies(r["country"], config.taxonomy.unknown_label)
        style = labels[best[i]] if in_domain else None
        if not in_domain:
            suppressed += 1
        works.append({
            "id": artwork_id,
            "t": r["title"],
            # artist_display keeps "After Raphael"; artist_title flattens it
            "a": r.get("artist_display") or r.get("artist"),
            "img": r["image_id"],
            "d": shown_date(r),
            "pb": r["period_bins"],
            # Midpoint year: the period arc uses it to order works inside a
            # bin, so the seam between two bins is where their dates meet.
            "my": (r["date_start"] + r["date_end"]) // 2,
            "p": r["period"],
            # The bin's numeric start, so nothing downstream has to parse
            # a human label back into a year.
            "pbin": r["period_bin"],
            "prec": r["date_precision"],
            "c": r["country"],
            "s": style,
            "sc": round(float(confidence[i]), 3) if in_domain else None,
            "xy": [round(float(xy[i][0]), 4), round(float(xy[i][1]), 4)],
            "type": r.get("artwork_type"),
            **({"k": "#%02x%02x%02x" % tuple(colors[i])} if colors is not None else {}),
            **({"ar": round(float(dims[i][0]) / max(float(dims[i][1]), 1), 3)}
               if dims is not None else {}),
        })
        neighbors[str(artwork_id)] = [int(ids[j]) for j in nn[i]]
        for b in r["period_bins"]:
            periods[b] += 1
        countries[r["country"]] += 1
        styles[style if in_domain else OUTSIDE_TAXONOMY] += 1

    (out / "works.json").write_text(json.dumps(works, ensure_ascii=False,
                                               separators=(",", ":")))
    (out / "neighbors.json").write_text(json.dumps(neighbors,
                                                   separators=(",", ":")))
    (out / "facets.json").write_text(json.dumps({
        "periods": [{"bin": b, "label": period_label(b), "count": periods[b]}
                    for b in sorted(periods)],
        "countries": [{"name": c, "count": n} for c, n in countries.most_common()],
        "styles": [{"name": s, "count": n} for s, n in styles.most_common()],
        "outside_taxonomy_label": OUTSIDE_TAXONOMY,
        "style_domain_note": (
            "Style is predicted only for works from traditions the 17-label "
            "taxonomy covers: European painting and Japanese ukiyo-e. "
            "Elsewhere it is withheld, because the model is confidently "
            "wrong rather than uncertain."),
        "style_labels": labels,
        "precision_levels": ["exact", "loose", "vague"],
        # Confined sizing: AIC 403s any request that would upscale, so a
        # fixed width breaks for every work narrower than it.
        "iiif": "https://www.artic.edu/iiif/2/{image_id}/full/!{width},{width}/0/default.jpg",
        "counts": {"works": len(works), "neighbors_per_work": k},
    }, indent=2, ensure_ascii=False))

    print(f"\n  style withheld for {suppressed:,} works "
          f"({100*suppressed/len(works):.1f}%) outside the taxonomy's domain")

    # --- map layouts, one per facet --------------------------------------
    layouts = {}
    for facet in ("similarity", "period", "country", "style"):
        pos, centres, radii, labels, tile = layout.build(vecs, works, facet)
        sizes: dict[str, int] = {}
        for l in labels:
            sizes[l] = sizes.get(l, 0) + 1
        hues = layout.region_hues(centres)
        layouts[facet] = {
            "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in pos],
            # The similarity layout has no regions at all — position is the
            # embedding and nothing else — so it ships an empty list and
            # the client draws no labels.
            "regions": [
                {"name": name,
                 "c": [round(float(centres[name][0]), 4),
                       round(float(centres[name][1]), 4)],
                 "r": round(float(radii[name]), 4),
                 "n": sizes[name],
                 "h": hues.get(name, 0.0),
                 "o": layout.region_outline(
                     pos[[i for i, l in enumerate(labels) if l == name]],
                     centres[name], float(radii[name]))}
                for name in sorted(sizes, key=lambda k: -sizes[k])
                if name in centres],
            "region_of": labels,
            # Per facet: a ribbon carries a different tile size than a disc.
            "work_radius": round(float(tile), 6),
            # Century boundaries, for the timeline to rule itself against.
            "grid": (layout.timeline_grid(
                        pos, np.array([w["my"] for w in works]))
                     if facet in layout.ARC_FACETS else []),
        }
        drawn = sum(1 for n in sizes if n in centres)
        print(f"  layout {facet:<10} {drawn:>3} regions"
              + ("  (position is the embedding alone)" if not drawn else ""))
    # --- pack order for the lazily-loaded high-resolution atlas ----------
    # Hilbert order through the UMAP plane, so a screenful of the map lives
    # in a few sheets rather than all of them: measured 3.5 sheets against
    # 40 for ids order at 100 tiles a sheet.
    umap_xy = np.array([w["xy"] for w in works])
    order = hilbert_order(umap_xy)
    slot_of = np.empty(len(order), dtype=np.int64)
    slot_of[order] = np.arange(len(order))
    (out / "hires_slots.json").write_text(json.dumps(
        [int(v) for v in slot_of], separators=(",", ":")))
    (config.paths.processed / "hires_order.json").write_text(json.dumps(
        [int(works[i]["id"]) for i in order]))
    print(f"  hires pack order: a map neighbourhood spans "
          f"{locality(order, umap_xy, 100):.1f} sheets "
          f"(vs {locality(np.arange(len(works)), umap_xy, 100):.1f} unordered)")

    # The 64px sheets were packed along this same curve, and the tiles in
    # them cannot be checked from here. If this run produces a different
    # order, every tile on the map is the wrong picture and nothing else
    # would say so, so say it here.
    packed = out / "atlas_slots.json"
    if packed.exists():
        was = json.loads(packed.read_text())
        if was != [int(v) for v in slot_of]:
            print("\n  !! the 64px atlas was packed in a different order "
                  "than this run computed.")
            print("     Re-run scripts/d5c_repack.py, or every tile on the "
                  "map will be the wrong work.\n")
        else:
            print("  64px atlas: packed in this order, tiles agree")

    (out / "layouts.json").write_text(json.dumps({
        "work_radius": round(layout.work_radius(len(works)), 5),
        "facets": layouts,
    }, separators=(",", ":"), ensure_ascii=False))

    # Measured gzipped, because that is what a visitor downloads: every
    # static host compresses JSON, and these files compress about 3.7x.
    # Capping raw bytes was capping a number nobody pays.
    def wire_size(path: Path) -> int:
        return len(gzip.compress(path.read_bytes(), 6))

    total = sum(wire_size(out / f)
                for f in ("works.json", "neighbors.json", "facets.json",
                          "layouts.json", "hires_slots.json"))
    print(f"\n=== bundle ===")
    for f in ("works.json", "neighbors.json", "facets.json", "layouts.json",
              "hires_slots.json"):
        raw = (out / f).stat().st_size
        print(f"  {f:<16} {wire_size(out/f)/1e6:6.2f} MB gzipped "
              f"({raw/1e6:5.2f} raw)")
    print(f"  {'total':<16} {total/1e6:6.2f} MB gzipped   "
          f"(cap {config.site.max_bundle_mb} MB)")
    if total > config.site.max_bundle_mb * 1e6:
        raise SystemExit("bundle is over the cap set in config.yaml")
    print(f"\nD5: {len(works):,} works -> {out}")


if __name__ == "__main__":
    main()
