"""D8b: train the 46-class style head on Artificio/WikiArt vectors.

    .venv/bin/python -m pipeline.train_style_v2

Same discipline as D4, which is the point: artists held out of the test
fold, folds pooled so every work is tested exactly once, balanced class
weights, and per-class F1 reported beside its support so a score computed
from nine examples cannot pass for a result.

What changed is the problem, not the method. 46 classes against 17, on
101,917 works against 73,334, from a backbone with 768 dimensions against
512. Macro F1 will read lower than D4's 0.516 and that is not a
regression: D4 scored well partly because its taxonomy had merged the
hard distinctions away. Judge this by whether Precisionism, Surrealism
and Neoclassicism become sayable at all.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from pipeline.config import Config
from pipeline.taxonomy_v2 import STYLE_LABELS, STYLE_MAP

SEED = 0
FOLDS = 5
# WikiArt's artist field is free text. Works whose artist is unknown cannot
# be held out as a group — "unknown" is not a painter — so they train the
# head but are never part of a test fold. Same rule as D4.
UNKNOWN_ARTISTS = {"", "unknown artist", "unknown", "anonymous", "none"}


def load(config: Config):
    d = config.paths.interim / "wikiart2"
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["fingerprint"] != config.model.fingerprint:
        raise SystemExit(
            f"vectors are {manifest['fingerprint']}, config.yaml is "
            f"{config.model.fingerprint} — these do not share a space")
    X = np.load(d / "vectors.npy")
    meta = [json.loads(l) for l in (d / "meta.jsonl").open()]
    if len(meta) != len(X):
        raise SystemExit(f"{len(X)} vectors but {len(meta)} rows of metadata")

    keep = [i for i, m in enumerate(meta) if m.get("style") in STYLE_MAP]
    y = np.array([STYLE_LABELS.index(STYLE_MAP[meta[i]["style"]]) for i in keep])
    artist = [(meta[i].get("artist") or "").strip().lower() for i in keep]
    ids = {a: n for n, a in enumerate(sorted(set(artist)), start=1)}
    g = np.array([0 if a in UNKNOWN_ARTISTS else ids[a] for a in artist])
    return X[keep], y, g, manifest, [meta[i] for i in keep]


def evaluate(X, y, train_idx, test_idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    clf = LogisticRegression(max_iter=1200, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    return confusion_matrix(y[test_idx], pred,
                            labels=list(range(len(STYLE_LABELS))))


def scores(cm: np.ndarray):
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp), float),
                          where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp), float),
                       where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros(len(tp), float), where=denom > 0)
    return f1, support, tp.sum() / max(cm.sum(), 1)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    import joblib

    config = Config.load()
    X, y, g, manifest, _ = load(config)
    named = np.where(g != 0)[0]
    print(f"{len(y):,} works, {len(STYLE_LABELS)} classes, dim {X.shape[1]}")
    print(f"backbone {manifest['backbone']}  fingerprint "
          f"{manifest['fingerprint']}")
    print(f"{len(set(g[named].tolist())):,} named artists; "
          f"{len(y) - len(named):,} works unattributed\n")

    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    all_idx = np.arange(len(y))
    total = np.zeros((len(STYLE_LABELS), len(STYLE_LABELS)), dtype=np.int64)
    for n, (_, te_a) in enumerate(
            sgkf.split(X[named], y[named], groups=g[named]), 1):
        te = named[te_a]
        tr = np.setdiff1d(all_idx, te)
        assert not (set(g[te].tolist()) & set(g[tr].tolist()) - {0})
        print(f"  fold {n}: {len(set(g[te].tolist())):>4} held-out artists, "
              f"{len(te):>6,} works")
        total += evaluate(X, y, tr, te)

    f1, support, accuracy = scores(total)
    present = [i for i in range(len(STYLE_LABELS)) if support[i] > 0]
    macro = float(np.mean([f1[i] for i in present]))
    print(f"\n=== {FOLDS} folds pooled, every work tested once ===")
    print(f"  accuracy  {accuracy:.3f}")
    print(f"  macro F1  {macro:.3f}   over {len(present)} classes")
    print(f"  D4 was 0.539 / 0.516 over 17 classes on a smaller corpus\n")

    print("  per-class F1 (n = test examples):")
    for i in sorted(present, key=lambda i: -f1[i]):
        flag = "  <- too few to trust" if support[i] < 200 else ""
        bar = "#" * round(26 * f1[i])
        print(f"    {STYLE_LABELS[i]:<24}{f1[i]:.3f}  n={support[i]:<6,}"
              f"{bar}{flag}")

    cm = total.copy()
    np.fill_diagonal(cm, 0)
    print("\n  worst confusions:")
    flat = sorted(((cm[i, j], i, j) for i in present for j in present
                   if cm[i, j] > 0), reverse=True)[:8]
    for n, i, j in flat:
        print(f"    {n:>5}  {STYLE_LABELS[i]} -> {STYLE_LABELS[j]}")

    final = LogisticRegression(max_iter=1200, class_weight="balanced")
    final.fit(X, y)
    out = config.paths.processed
    joblib.dump({"model": final, "labels": STYLE_LABELS,
                 "fingerprint": manifest["fingerprint"]},
                out / "style_head.joblib")
    (out / "style_eval.json").write_text(json.dumps({
        "fingerprint": manifest["fingerprint"],
        "backbone": manifest["backbone"],
        "labels": STYLE_LABELS,
        "n_total": int(len(y)),
        "accuracy": accuracy,
        "macro_f1": macro,
        "per_class_f1": {STYLE_LABELS[i]: float(f1[i]) for i in present},
        "per_class_support": {STYLE_LABELS[i]: int(support[i]) for i in present},
        "confusion": total.tolist(),
    }, indent=2))
    print(f"\nD8b: head -> {out / 'style_head.joblib'}")


if __name__ == "__main__":
    main()
