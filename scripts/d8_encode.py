#!/usr/bin/env python3
"""D8: encode Artificio/WikiArt with a chosen CLIP backbone.

    python3 d8_encode.py --out out_wa2 --backbone openai/clip-vit-base-patch32

Standalone, like the other remote scripts. Replaces huggan/wikiart for
training: 103,250 works against 81,444, and 137 style labels against 27,
in 1.7 GB against 33.7 because the images ship pre-resized to 256px.

Cross-source compatibility was checked before committing to it: the head
trained on huggan's full-resolution images scores 51% top-1 and 88% top-3
on these 256px ones, against 54% on its own held-out test. The smaller
images are not a meaningful domain shift.

Also takes --backbone, because which CLIP to use is the open question the
encode exists to answer. Run it two or three times and compare the heads
on identical splits; the vectors carry the backbone in their manifest so
two runs can never be mixed by accident.

Outputs to --out:
    vectors.npy    float32 [N, D], L2-normalized
    meta.jsonl     style, artist, date, genre per row, aligned
    manifest.json  backbone, fingerprint, dim, counts
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

REPO = "Artificio/WikiArt"
SHARDS = [
    "data/train-00000-of-00004-3c65976b59bc0ab4.parquet",
    "data/train-00001-of-00004-441bd829579dead0.parquet",
    "data/train-00002-of-00004-7b0bbb36fb350222.parquet",
    "data/train-00003-of-00004-971fec8ddd44fece.parquet",
]
COLUMNS = ["image", "style", "artist", "date", "genre"]


def fingerprint(backbone: str, image_size: int) -> str:
    return hashlib.sha256(f"{backbone}@{image_size}".encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N works, for a backbone bake-off")
    ap.add_argument("--keep-parquet", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fp = fingerprint(args.backbone, args.image_size)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"device      {device}")
    if device == "cuda":
        print(f"gpu         {torch.cuda.get_device_name(0)}")
    print(f"backbone    {args.backbone}")
    print(f"fingerprint {fp}")

    processor = CLIPImageProcessor.from_pretrained(args.backbone)
    print(f"preprocess  {type(processor).__name__}")
    model = CLIPVisionModelWithProjection.from_pretrained(args.backbone)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    dim = int(model.config.projection_dim)
    print(f"dim         {dim}\n")

    vecs: list[np.ndarray] = []
    flushes = 0
    meta_path = args.out / "meta.jsonl"
    meta = meta_path.open("w")
    seen = kept = 0
    started = time.time()

    def flush(images, rows):
        nonlocal kept, flushes
        if not images:
            return
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            out = model(pixel_values=inputs["pixel_values"].to(device))
            v = out.image_embeds
            v = v / v.norm(dim=-1, keepdim=True)
        vecs.append(v.cpu().numpy().astype(np.float32))
        for r in rows:
            meta.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta.flush()
        kept += len(rows)
        flushes += 1
        # Count batches, not works. Keying the report off `kept % N == 0`
        # meant that once a shard ended on a partial batch, `kept` stopped
        # landing on the multiple and the run went silent for the rest of
        # its life — indistinguishable from a hang.
        if flushes % 8 == 0:
            rate = kept / max(time.time() - started, 1e-6)
            print(f"  {kept:>7,} encoded  {rate:6.1f} img/s", flush=True)

    for name in SHARDS:
        if args.limit and kept >= args.limit:
            break
        # Downloading a 426 MB shard prints nothing on its own, so say so.
        print(f"  fetching {name.split('/')[-1]} ...", flush=True)
        path = Path(hf_hub_download(REPO, name, repo_type="dataset"))
        print(f"  encoding {name.split('/')[-1]}", flush=True)
        images, rows = [], []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256,
                                                       columns=COLUMNS):
            cols = {c: batch.column(c).to_pylist() for c in COLUMNS}
            for i in range(len(cols["image"])):
                seen += 1
                style = cols["style"][i]
                if not style or style == "None":
                    continue
                try:
                    img = Image.open(io.BytesIO(cols["image"][i]["bytes"])).convert("RGB")
                except Exception:
                    continue
                images.append(img)
                rows.append({"style": style,
                             "artist": cols["artist"][i],
                             "date": cols["date"][i],
                             "genre": cols["genre"][i]})
                if len(images) >= args.batch_size:
                    flush(images, rows)
                    images, rows = [], []
                    if args.limit and kept >= args.limit:
                        break
            if args.limit and kept >= args.limit:
                break
        flush(images, rows)
        if not args.keep_parquet:
            blob = path.resolve()
            path.unlink(missing_ok=True)
            if blob != path:
                blob.unlink(missing_ok=True)

    meta.close()
    if not vecs:
        sys.exit("nothing encoded")
    V = np.vstack(vecs)
    np.save(args.out / "vectors.npy", V)
    (args.out / "manifest.json").write_text(json.dumps({
        "source": REPO,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "fingerprint": fp,
        "preprocessor": type(processor).__name__,
        "count": int(len(V)),
        "dim": int(V.shape[1]),
        "rows_seen": seen,
        "normalized": True,
    }, indent=2))
    print(f"\nD8: {len(V):,} vectors, dim {V.shape[1]}, from {seen:,} rows")
    print(f"    copy back vectors.npy, meta.jsonl, manifest.json")


if __name__ == "__main__":
    main()
