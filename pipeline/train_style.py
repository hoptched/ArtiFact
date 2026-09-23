"""D4b: train the style head on cached WikiArt vectors.

    .venv/bin/python -m pipeline.train_style

Runs locally in seconds. The expensive work was D3/D4a; a linear head over
frozen CLIP vectors is a convex problem on 73k rows of 512 floats.

Two evaluations, because they answer different questions:

  random  — shuffle all works, split 80/20. Asks "can it label another
            Monet as Impressionism?" Optimistic: the same painter appears
            on both sides, and CLIP recognises painters readily (D3 found
            18.1% of nearest neighbours share an artist).

  grouped — hold out whole artists. Asks "can it label a painter it has
            never seen?" This is the number that transfers to AIC, whose
            works are largely by painters absent from WikiArt's 126.

The gap between them is itself a result: it measures how much of the score
is style recognition and how much is painter recognition.

Outputs to data/processed/:
    style_head.joblib     trained on everything, for D5 inference
    style_eval.json       both splits, per-class F1, confusion matrix
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.config import Config
from pipeline.taxonomy import STYLE_LABELS

SEED = 0
TEST_FRACTION = 0.2

# WikiArt's artist 0 is "Unknown Artist" and covers 34,444 works, 47% of
# the corpus. It is not a painter, it is the absence of attribution, so it
# cannot be held out as a group: doing so put half the corpus in one test
# fold. Those works still train the head, but the grouped evaluation is
# measured only on attributed works, where "an artist the model has never
# seen" is a claim that can actually be checked.
UNKNOWN_ARTIST = 0


def load(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    d = config.paths.interim / "wikiart"
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["fingerprint"] != config.model.fingerprint:
        raise SystemExit(
            f"fingerprint mismatch: WikiArt vectors are {manifest['fingerprint']}, "
            f"config.yaml is {config.model.fingerprint}. These do not share a "
            f"vector space; re-encode one of them.")
    X = np.load(d / "wikiart_embeddings.npy")
    y = np.load(d / "wikiart_labels.npy")
    g = np.load(d / "wikiart_artists.npy")
    return X, y, g, manifest


def evaluate(X, y, g, train_idx, test_idx, name: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score

    # Balanced weights: the corpus runs 14:1 from Impressionism to Fauvism,
    # and an unweighted fit buys accuracy by ignoring the thin classes.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    true = y[test_idx]

    present = sorted(set(true.tolist()))
    per_class = f1_score(true, pred, labels=present, average=None, zero_division=0)
    result = {
        "split": name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": float((pred == true).mean()),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "per_class_f1": {STYLE_LABELS[c]: float(s)
                         for c, s in zip(present, per_class)},
        # Support matters as much as the score: a class with one test
        # example produces an F1 of 0.0 or 1.0 and means neither.
        "per_class_support": {STYLE_LABELS[c]: int((true == c).sum())
                              for c in present},
        "classes_absent_from_test": [STYLE_LABELS[i]
                                     for i in range(len(STYLE_LABELS))
                                     if i not in present],
        "confusion": confusion_matrix(
            true, pred, labels=list(range(len(STYLE_LABELS)))).tolist(),
    }
    if name == "grouped":
        result["train_artists"] = int(len(set(g[train_idx].tolist())))
        result["test_artists"] = int(len(set(g[test_idx].tolist())))
    return result


def pool_folds(folds: list[dict]) -> dict:
    """Combine per-fold results into one. Confusions add up, since each
    work is a test work exactly once across the folds; F1 is recomputed
    from that pooled matrix rather than averaged, so thin classes are not
    dominated by whichever fold happened to hold two of them."""
    cm = np.sum([np.array(f["confusion"]) for f in folds], axis=0)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros(len(tp)), where=denom > 0)
    present = [i for i in range(len(STYLE_LABELS)) if support[i] > 0]
    return {
        "split": "grouped",
        "n_folds": len(folds),
        "n_train": int(np.mean([f["n_train"] for f in folds])),
        "n_test": int(cm.sum()),
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(np.mean([f1[i] for i in present])),
        "per_class_f1": {STYLE_LABELS[i]: float(f1[i]) for i in present},
        "per_class_support": {STYLE_LABELS[i]: int(support[i]) for i in present},
        "classes_absent_from_test": [STYLE_LABELS[i]
                                     for i in range(len(STYLE_LABELS))
                                     if support[i] == 0],
        "confusion": cm.tolist(),
    }


def report(res: dict) -> None:
    print(f"\n=== {res['split']} split ===")
    if res.get("n_folds"):
        print(f"  {res['n_folds']} folds pooled; every work tested once")
    print(f"  train {res['n_train']:,}   test {res['n_test']:,}")
    if "test_artists" in res:
        print(f"  artists: {res['train_artists']} train / "
              f"{res['test_artists']} test, disjoint")
    print(f"  accuracy  {res['accuracy']:.3f}")
    print(f"  macro F1  {res['macro_f1']:.3f}")
    if res["classes_absent_from_test"]:
        print(f"  NOT IN TEST SET: {', '.join(res['classes_absent_from_test'])}")
    print("\n  per-class F1 (n = test examples):")
    for label, score in sorted(res["per_class_f1"].items(), key=lambda kv: -kv[1]):
        bar = "#" * round(30 * score)
        n = res["per_class_support"][label]
        flag = "  <- too few to trust" if n < 30 else ""
        print(f"    {label:<20} {score:.3f}  n={n:<6,} {bar}{flag}")


def worst_confusions(res: dict, k: int = 6) -> list[tuple[str, str, int]]:
    """The k largest off-diagonal cells — the mistakes worth explaining."""
    cm = np.array(res["confusion"])
    np.fill_diagonal(cm, 0)
    flat = [(STYLE_LABELS[i], STYLE_LABELS[j], int(cm[i, j]))
            for i in range(len(STYLE_LABELS)) for j in range(len(STYLE_LABELS))
            if cm[i, j] > 0]
    return sorted(flat, key=lambda t: -t[2])[:k]


def main() -> None:
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    config = Config.load()
    X, y, g, manifest = load(config)
    print(f"{len(y):,} vectors, dim {X.shape[1]}, "
          f"{len(set(g.tolist()))} artists, {len(STYLE_LABELS)} classes")
    print(f"fingerprint {manifest['fingerprint']} matches config.yaml")

    idx = np.arange(len(y))

    # --- random split: the optimistic number ----------------------------
    tr, te = train_test_split(idx, test_size=TEST_FRACTION,
                              stratify=y, random_state=SEED)
    random_res = evaluate(X, y, g, tr, te, "random")
    report(random_res)

    # --- grouped split: the number that transfers -----------------------
    # Every fold, not one. Several classes come from only 5-8 artists, so a
    # single fold can leave a class with one test example, whose F1 is 0.0
    # or 1.0 and means neither. Averaging over all folds makes every work
    # a test work exactly once.
    sgkf = StratifiedGroupKFold(n_splits=int(1 / TEST_FRACTION),
                                shuffle=True, random_state=SEED)
    attributed = np.where(g != UNKNOWN_ARTIST)[0]
    print(f"\n  grouping over {len(attributed):,} attributed works "
          f"({len(y) - len(attributed):,} unattributed still train the head)")
    folds = []
    all_idx = np.arange(len(y))
    for n, (_, te_a) in enumerate(
            sgkf.split(X[attributed], y[attributed], groups=g[attributed]), 1):
        te = attributed[te_a]
        # Train on everything not under test, unattributed works included.
        tr = np.setdiff1d(all_idx, te, assume_unique=False)
        held = set(g[te].tolist())
        assert not (held & set(g[tr].tolist()) - {UNKNOWN_ARTIST}), "artist leaked"
        print(f"  fold {n}: {len(held)} held-out artists, {len(te):,} works")
        folds.append(evaluate(X, y, g, tr, te, "grouped"))
    grouped_res = pool_folds(folds)
    report(grouped_res)

    # --- the comparison that matters ------------------------------------
    print("\n=== what the gap means ===")
    d_acc = grouped_res["accuracy"] - random_res["accuracy"]
    d_f1 = grouped_res["macro_f1"] - random_res["macro_f1"]
    print(f"  accuracy {random_res['accuracy']:.3f} -> "
          f"{grouped_res['accuracy']:.3f}  ({d_acc:+.3f})")
    print(f"  macro F1 {random_res['macro_f1']:.3f} -> "
          f"{grouped_res['macro_f1']:.3f}  ({d_f1:+.3f})")
    print("  that drop is painter recognition, not style recognition;")
    print("  the grouped number is what D5 will actually deliver on AIC.")
    print("  (grouped is measured on attributed works only; the 34,444")
    print("   unattributed ones train the head but cannot be held out.)")

    print("\n=== worst confusions (grouped) ===")
    for a, b, n in worst_confusions(grouped_res):
        print(f"  {n:>4}  {a}  ->  {b}")

    # --- final head, fitted on everything, for D5 -----------------------
    from sklearn.linear_model import LogisticRegression
    import joblib
    final = LogisticRegression(max_iter=2000, class_weight="balanced")
    final.fit(X, y)
    out = config.paths.processed
    joblib.dump({"model": final, "labels": STYLE_LABELS,
                 "fingerprint": manifest["fingerprint"]},
                out / "style_head.joblib")
    (out / "style_eval.json").write_text(json.dumps({
        "fingerprint": manifest["fingerprint"],
        "labels": STYLE_LABELS,
        "n_total": int(len(y)),
        "n_artists": int(len(set(g.tolist()))),
        "random": random_res,
        "grouped": grouped_res,
    }, indent=2))
    print(f"\nD4b: head -> {out/'style_head.joblib'}")
    print(f"     eval -> {out/'style_eval.json'}")


if __name__ == "__main__":
    main()
