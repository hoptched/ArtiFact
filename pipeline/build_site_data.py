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

import json
from collections import Counter

import numpy as np

from pipeline.config import Config
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
    corpus = {r["id"]: r for r in
              (json.loads(l) for l in (P / "corpus.jsonl").open())}
    return vecs, ids, corpus, head


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


def main() -> None:
    config = Config.load()
    config.paths.ensure()
    vecs, ids, corpus, head = load(config)
    model, labels = head["model"], head["labels"]
    print(f"{len(ids):,} vectors, {len(corpus):,} corpus rows, "
          f"{len(labels)} style labels\n")

    # --- style prediction ------------------------------------------------
    proba = model.predict_proba(vecs)
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
        in_domain = style_applies(r["country"])
        style = labels[best[i]] if in_domain else None
        if not in_domain:
            suppressed += 1
        works.append({
            "id": artwork_id,
            "t": r["title"],
            # artist_display keeps "After Raphael"; artist_title flattens it
            "a": r.get("artist_display") or r.get("artist"),
            "img": r["image_id"],
            "d": r.get("date_display"),
            "pb": r["period_bins"],
            "p": r["period"],
            "prec": r["date_precision"],
            "c": r["country"],
            "s": style,
            "sc": round(float(confidence[i]), 3) if in_domain else None,
            "xy": [round(float(xy[i][0]), 4), round(float(xy[i][1]), 4)],
            "type": r.get("artwork_type"),
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
        "iiif": "https://www.artic.edu/iiif/2/{image_id}/full/{width},/0/default.jpg",
        "counts": {"works": len(works), "neighbors_per_work": k},
    }, indent=2, ensure_ascii=False))

    print(f"\n  style withheld for {suppressed:,} works "
          f"({100*suppressed/len(works):.1f}%) outside the taxonomy's domain")

    total = sum((out / f).stat().st_size
                for f in ("works.json", "neighbors.json", "facets.json"))
    print(f"\n=== bundle ===")
    for f in ("works.json", "neighbors.json", "facets.json"):
        print(f"  {f:<16} {(out/f).stat().st_size/1e6:6.2f} MB")
    print(f"  {'total':<16} {total/1e6:6.2f} MB   "
          f"(cap {config.site.max_bundle_mb} MB)")
    if total > config.site.max_bundle_mb * 1e6:
        raise SystemExit("bundle is over the cap set in config.yaml")
    print(f"\nD5: {len(works):,} works -> {out}")


if __name__ == "__main__":
    main()
