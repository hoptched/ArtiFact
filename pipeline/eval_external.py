"""D8c: test the style head against a label it has never seen.

    .venv/bin/python -m pipeline.eval_external

The 0.486 from D8b is an in-distribution number: trained and tested on one
WikiArt scrape, one set of contributors, one set of photographic
conventions. It says the head learned that dataset. It does not say the
head learned style.

The Art Institute publishes its own `style_title` on about 1,800 works.
Most of it is centuries and cultures and unusable — "18th Century",
"Chinese (culture or style)" — which is why D2 refused to treat it as a
taxonomy. But a few hundred entries do name a movement, assigned by AIC's
curators from AIC's photography, with no connection to WikiArt at all.
That is a genuine held-out test, and it is the only one this project has.

Only unambiguous names are mapped. "Renaissance" is skipped because it
could be Early, High or Northern; "Modernism" and "20th Century" name
nothing the head predicts; "Hudson River School" sits between Realism and
Romanticism and a coin flip would be dressed up as a measurement.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from pipeline.config import Config

# AIC's wording -> the class this head predicts.
AIC_TO_CLASS: dict[str, str] = {
    "Impressionism": "Impressionism",
    "Post-Impressionism": "Post-Impressionism",
    "Realism": "Realism",
    "Barbizon School": "Realism",
    "medieval": "Medieval",
    "Folk Art": "Naive art",
    "Neoclassicism": "Neoclassicism",
    "Baroque": "Baroque",
    "Mannerism": "Mannerism",
    "romantic": "Romanticism",
    "Pointillism": "Post-Impressionism",
    "synthetist": "Post-Impressionism",
    "Rococo": "Rococo",
    "safavid": "Persian & Ottoman",
    "isfahan": "Persian & Ottoman",
    "saffarid": "Persian & Ottoman",
}


def main() -> None:
    import joblib

    config = Config.load()
    P = config.paths.processed
    head = joblib.load(P / "style_head.joblib")
    if head["fingerprint"] != config.model.fingerprint:
        raise SystemExit("head and config disagree on the vector space")
    labels = head["labels"]

    vecs = np.load(P / "embeddings.npy")
    ids = json.loads((P / "ids.json").read_text())
    row = {i: n for n, i in enumerate(ids)}
    corpus = {r["id"]: r for r in
              (json.loads(l) for l in (P / "corpus.jsonl").open())}

    idx, truth = [], []
    for artwork_id, r in corpus.items():
        aic = r.get("aic_style_title")
        if aic in AIC_TO_CLASS and artwork_id in row:
            cls = AIC_TO_CLASS[aic]
            if cls in labels:
                idx.append(row[artwork_id])
                truth.append(labels.index(cls))
    idx, truth = np.array(idx), np.array(truth)
    print(f"  {len(idx)} works AIC labels with a movement this head predicts")
    print(f"  across {len(set(truth.tolist()))} classes\n")

    proba = head["model"].predict_proba(vecs[idx])
    order = np.argsort(-proba, axis=1)
    top1 = order[:, 0] == truth
    top3 = np.array([truth[i] in order[i, :3] for i in range(len(truth))])

    print(f"  top-1  {100*top1.mean():.1f}%")
    print(f"  top-3  {100*top3.mean():.1f}%")
    # A head that always answered with the commonest class in this set.
    biggest = Counter(truth.tolist()).most_common(1)[0][1]
    print(f"  always guessing the commonest class here: "
          f"{100*biggest/len(truth):.1f}%")
    print(f"  guessing uniformly among {len(labels)} classes: "
          f"{100/len(labels):.1f}%\n")

    print(f"  {'AIC says':<22}{'n':>5}{'top-1':>8}{'top-3':>8}   "
          f"commonest prediction")
    for cls in sorted(set(truth.tolist()), key=lambda c: -(truth == c).sum()):
        m = truth == cls
        guesses = Counter(labels[i] for i in order[m, 0]).most_common(1)[0]
        print(f"  {labels[cls]:<22}{int(m.sum()):>5}"
              f"{100*top1[m].mean():>7.0f}%{100*top3[m].mean():>7.0f}%   "
              f"{guesses[0]} ({guesses[1]})")

    conf = Counter()
    for i in range(len(truth)):
        if not top1[i]:
            conf[(labels[truth[i]], labels[order[i, 0]])] += 1
    print("\n  where it goes wrong:")
    for (t, g), n in conf.most_common(6):
        print(f"    {n:>3}  {t} -> {g}")

    (P / "eval_external.json").write_text(json.dumps({
        "source": "Art Institute of Chicago style_title",
        "n": int(len(idx)),
        "top1": float(top1.mean()),
        "top3": float(top3.mean()),
        "classes": sorted({labels[c] for c in truth.tolist()}),
    }, indent=2))


if __name__ == "__main__":
    main()
