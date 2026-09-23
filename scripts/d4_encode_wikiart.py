#!/usr/bin/env python3
"""D4 (first half): encode WikiArt with the same frozen CLIP tower as D3.

    python3 d4_encode_wikiart.py --out out_wikiart

Needs `taxonomy.py` beside it — copy pipeline/taxonomy.py across. That file
imports nothing, and sharing it rather than duplicating the 27->17 style
map is the point: the labels the head trains on must be the same object
the site later displays.

WikiArt is 81,444 images in 72 parquet shards, 33.7 GB. Shards are pulled
one at a time, encoded, and deleted, so peak disk stays near 1 GB instead
of 34 and a killed run resumes at shard granularity.

**WikiArt images are training data only and must never be redistributed.**
Nothing here writes an image out; only vectors and labels leave.

Outputs to --out:
    wikiart_embeddings.npy  float32 [N, D], L2-normalized
    wikiart_labels.npy      int16 [N], index into STYLE_LABELS
    manifest.json           backbone, fingerprint, per-class counts

The fingerprint and the preprocessor must match the D3 run exactly, or the
head trains in one vector space and predicts in another.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

REPO = "huggan/wikiart"
N_SHARDS = 72
INFOS = "https://huggingface.co/datasets/huggan/wikiart/resolve/main/dataset_infos.json"


def fingerprint(backbone: str, image_size: int) -> str:
    return hashlib.sha256(f"{backbone}@{image_size}".encode()).hexdigest()[:12]


def wikiart_style_names() -> list[str]:
    """The 27 class names in index order, read from the dataset itself.

    Fetched rather than hardcoded: the parquet stores style as an integer,
    and if the class order ever changed underneath a hardcoded list every
    label would silently shift by one.
    """
    import urllib.request
    with urllib.request.urlopen(INFOS, timeout=60) as fh:
        infos = json.load(fh)
    info = next(iter(infos.values()))
    names = info["features"]["style"]["names"]
    if len(names) != 27:
        sys.exit(f"expected 27 style classes, dataset reports {len(names)}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--shards", default=f"0-{N_SHARDS-1}",
                    help="inclusive range, e.g. 0-9 for a trial run")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="do not delete each shard after encoding it")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    try:
        from taxonomy import (DROPPED_WIKIART_STYLES, STYLE_LABELS,
                              WIKIART_STYLE_MAP)
    except ImportError:
        sys.exit("taxonomy.py must sit beside this script "
                 "(copy it from pipeline/taxonomy.py)")

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    lo, hi = (int(x) for x in args.shards.split("-"))
    fp = fingerprint(args.backbone, args.image_size)
    args.out.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out / "shards"
    shard_dir.mkdir(exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    names = wikiart_style_names()
    # index into WikiArt's 27 -> index into our 17, or -1 for the postwar
    # classes we drop rather than merge.
    label_of = np.full(len(names), -1, dtype=np.int16)
    for i, name in enumerate(names):
        if name in DROPPED_WIKIART_STYLES:
            continue
        mapped = WIKIART_STYLE_MAP.get(name)
        if mapped is None:
            sys.exit(f"style {name!r} is neither mapped nor dropped in taxonomy.py")
        label_of[i] = STYLE_LABELS.index(mapped)

    print(f"device      {device}")
    if device == "cuda":
        print(f"gpu         {torch.cuda.get_device_name(0)}")
    print(f"backbone    {args.backbone}")
    print(f"fingerprint {fp}   (must match the D3 run)")
    print(f"labels      {len(STYLE_LABELS)} kept, "
          f"{len(DROPPED_WIKIART_STYLES)} WikiArt classes dropped")
    print(f"shards      {lo}-{hi} of 0-{N_SHARDS-1}\n")

    processor = CLIPImageProcessor.from_pretrained(args.backbone)
    print(f"preprocess  {type(processor).__name__}   (must match the D3 run)\n")
    model = CLIPVisionModelWithProjection.from_pretrained(args.backbone)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    started = time.time()
    total_seen = total_kept = 0

    for shard in range(lo, hi + 1):
        target = shard_dir / f"shard_{shard:05d}.npz"
        if target.exists():
            print(f"  shard {shard:>2}: done already, skipping")
            continue

        name = f"data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        path = Path(hf_hub_download(REPO, name, repo_type="dataset"))

        vecs_acc, labs_acc = [], []
        images, labels = [], []
        seen = kept = 0

        def flush_batch() -> None:
            nonlocal images, labels
            if not images:
                return
            inputs = processor(images=images, return_tensors="pt")
            with torch.no_grad():
                out = model(pixel_values=inputs["pixel_values"].to(device))
                v = out.image_embeds
                v = v / v.norm(dim=-1, keepdim=True)
            vecs_acc.append(v.cpu().numpy().astype(np.float32))
            labs_acc.append(np.array(labels, dtype=np.int16))
            images, labels = [], []

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["image", "style"]):
            imgs = batch.column("image").to_pylist()
            styles = batch.column("style").to_pylist()
            for blob, style in zip(imgs, styles):
                seen += 1
                lab = label_of[style]
                if lab < 0:                      # a dropped postwar class
                    continue
                try:
                    img = Image.open(io.BytesIO(blob["bytes"])).convert("RGB")
                except Exception:
                    continue
                images.append(img)
                labels.append(int(lab))
                kept += 1
                if len(images) >= args.batch_size:
                    flush_batch()
        flush_batch()

        if vecs_acc:
            np.savez(target, vecs=np.vstack(vecs_acc),
                     labels=np.concatenate(labs_acc))
        if not args.keep_parquet:
            # hf_hub_download returns a symlink under snapshots/ pointing at
            # the real file in blobs/. Deleting only the link frees nothing,
            # so resolve it first and remove both.
            blob = path.resolve()
            path.unlink(missing_ok=True)
            if blob != path:
                blob.unlink(missing_ok=True)

        total_seen += seen
        total_kept += kept
        rate = total_seen / max(time.time() - started, 1e-6)
        left = (hi - shard) * (total_seen / max(shard - lo + 1, 1))
        print(f"  shard {shard:>2}: {seen:>5,} rows, {kept:>5,} kept  "
              f"| running {total_kept:>6,}/{total_seen:>6,}  "
              f"{rate:5.1f} img/s  eta {left/max(rate,1e-6)/60:5.1f} min")

    # --- merge -----------------------------------------------------------
    vecs_all, labs_all = [], []
    for f in sorted(shard_dir.glob("shard_*.npz")):
        z = np.load(f)
        vecs_all.append(z["vecs"])
        labs_all.append(z["labels"])
    if not vecs_all:
        sys.exit("no shards encoded")
    vecs = np.vstack(vecs_all)
    labs = np.concatenate(labs_all)

    np.save(args.out / "wikiart_embeddings.npy", vecs)
    np.save(args.out / "wikiart_labels.npy", labs)

    counts = {STYLE_LABELS[i]: int((labs == i).sum())
              for i in range(len(STYLE_LABELS))}
    (args.out / "manifest.json").write_text(json.dumps({
        "source": REPO,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "fingerprint": fp,
        "preprocessor": type(processor).__name__,
        "count": int(len(labs)),
        "dim": int(vecs.shape[1]),
        "rows_seen": total_seen,
        "normalized": True,
        "style_labels": STYLE_LABELS,
        "class_counts": counts,
        "dropped_wikiart_styles": sorted(DROPPED_WIKIART_STYLES),
    }, indent=2))

    print(f"\n=== class balance ({len(labs):,} training images) ===")
    widest = max(counts.values()) or 1
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        bar = "#" * max(0, round(40 * n / widest))
        print(f"  {label:<20} {n:>6,}  {bar}")
    thin = [l for l, n in counts.items() if n < 500]
    if thin:
        print(f"\n  thin classes (<500): {', '.join(thin)}")
        print("  expect weak per-class F1 there; consider class weighting in D4b")

    print(f"\nD4a: {len(labs):,} vectors, dim {vecs.shape[1]}")
    print(f"     copy back wikiart_embeddings.npy, wikiart_labels.npy, manifest.json")


if __name__ == "__main__":
    main()
