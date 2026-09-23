#!/usr/bin/env python3
"""D3: encode the AIC corpus with a frozen CLIP image tower.

Standalone by design. This file imports nothing from the ArtiFact package
and reads no config.yaml, so it can be copied to a machine that has never
seen the repo. Its only inputs are corpus.jsonl and the CLI flags below.

    python3 d3_embed.py --corpus corpus.jsonl --out out/

Outputs, all written to --out:
    embeddings.npy   float32 [N, D], L2-normalized, row i belongs to ids[i]
    ids.json         the AIC artwork ids, aligned to the rows
    manifest.json    backbone, image size, fingerprint, counts

The fingerprint is sha256("<backbone>@<image_size>")[:12] and must match
the one pipeline/config.py derives from config.yaml. Vectors built under a
different fingerprint are not comparable and must not be mixed: the head
would train in one space and predict in another, which shows up as
mediocre accuracy rather than as an error.

Resumable. Work is written as shards under --out/shards/ as it completes;
re-running skips every id already in a shard, so a killed run costs only
the batch in flight. Shards merge into embeddings.npy at the end.

Images are fetched from AIC's IIIF server into a scratch cache and are
never part of the output. Delete the cache when the run is done.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/{width},/0/default.jpg"


def fingerprint(backbone: str, image_size: int) -> str:
    return hashlib.sha256(f"{backbone}@{image_size}".encode()).hexdigest()[:12]


def load_corpus(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("image_id"):
                    rows.append({"id": r["id"], "image_id": r["image_id"]})
    return rows


def done_ids(shard_dir: Path) -> set[int]:
    import numpy as np
    seen: set[int] = set()
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        try:
            seen.update(int(i) for i in np.load(shard)["ids"])
        except Exception as exc:              # a shard killed mid-write
            print(f"  discarding unreadable shard {shard.name} ({exc})")
            shard.unlink()
    return seen


def fetch(session, image_id: str, width: int, cache: Path, retries: int = 3):
    """Bytes for one image, from the cache if present. None if it fails."""
    cached = cache / f"{image_id}.jpg"
    if cached.exists():
        return cached.read_bytes()
    url = IIIF.format(image_id=image_id, width=width)
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60.0)
        except Exception:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt + 1)
            continue
        if resp.status_code != 200:
            return None
        cached.write_bytes(resp.content)
        return resp.content
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--iiif-width", type=int, default=400,
                    help="pixels requested from AIC; downscaled to --image-size")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent image downloads; keep modest, it is a museum CDN")
    ap.add_argument("--shard-every", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="stop after N (smoke test)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="image scratch dir (default: <out>/imgcache)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import httpx
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    fp = fingerprint(args.backbone, args.image_size)
    args.out.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out / "shards"
    shard_dir.mkdir(exist_ok=True)
    cache = args.cache or (args.out / "imgcache")
    cache.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device      {device}")
    if device == "cuda":
        print(f"gpu         {torch.cuda.get_device_name(0)} "
              f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
    print(f"backbone    {args.backbone}")
    print(f"fingerprint {fp}")

    corpus = load_corpus(args.corpus)
    already = done_ids(shard_dir)
    todo = [r for r in corpus if r["id"] not in already]
    if args.limit:
        todo = todo[:args.limit]
    print(f"corpus {len(corpus):,} | done {len(already):,} | to encode {len(todo):,}")
    if not todo:
        print("nothing to do; merging what exists")

    processor = CLIPImageProcessor.from_pretrained(args.backbone)
    # Not covered by the fingerprint, but it must match across the AIC and
    # WikiArt runs: PIL and torchvision resize differently, and vectors
    # preprocessed two ways do not share a space. Recorded so a mismatch is
    # visible instead of silent.
    backend = type(processor).__name__
    print(f"preprocess  {backend}\n")
    model = CLIPVisionModelWithProjection.from_pretrained(args.backbone)
    model.eval().to(device)
    # Frozen: no gradients anywhere, ever. The heads are trained in D4 on
    # top of these vectors, the backbone itself never moves.
    for p in model.parameters():
        p.requires_grad_(False)

    pending_ids: list[int] = []
    pending_vecs: list["np.ndarray"] = []
    shard_n = len(list(shard_dir.glob("shard_*.npz")))
    failed = 0
    started = time.time()

    def flush() -> None:
        nonlocal shard_n, pending_ids, pending_vecs
        if not pending_ids:
            return
        np.savez(shard_dir / f"shard_{shard_n:05d}.npz",
                 ids=np.array(pending_ids, dtype=np.int64),
                 vecs=np.vstack(pending_vecs).astype(np.float32))
        shard_n += 1
        pending_ids, pending_vecs = [], []

    headers = {"User-Agent": "ArtiFact/0.1 (personal project)"}
    with httpx.Client(headers=headers, follow_redirects=True) as session, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            blobs = list(pool.map(
                lambda r: fetch(session, r["image_id"], args.iiif_width, cache),
                batch))

            images, ids = [], []
            for row, blob in zip(batch, blobs):
                if blob is None:
                    failed += 1
                    continue
                try:
                    images.append(Image.open(io.BytesIO(blob)).convert("RGB"))
                    ids.append(row["id"])
                except Exception:
                    failed += 1
            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt")
            with torch.no_grad():
                out = model(pixel_values=inputs["pixel_values"].to(device))
                vecs = out.image_embeds
                # Normalize here so D5's cosine KNN is a plain dot product
                # and nothing downstream has to remember to do it.
                vecs = vecs / vecs.norm(dim=-1, keepdim=True)
            pending_ids.extend(ids)
            pending_vecs.append(vecs.cpu().numpy())

            if len(pending_ids) >= args.shard_every:
                flush()

            done = start + len(batch)
            rate = done / max(time.time() - started, 1e-6)
            eta = (len(todo) - done) / max(rate, 1e-6)
            print(f"  {done:>6,}/{len(todo):,}  {rate:5.1f} img/s  "
                  f"eta {eta/60:5.1f} min  failed {failed}")
        flush()

    # --- merge shards into the aligned pair D4 and D5 consume -------------
    ids_all, vecs_all = [], []
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        z = np.load(shard)
        ids_all.append(z["ids"])
        vecs_all.append(z["vecs"])
    if not ids_all:
        sys.exit("no shards produced; nothing to merge")

    ids = np.concatenate(ids_all)
    vecs = np.vstack(vecs_all)
    # Shards can overlap if a run was killed between write and exit.
    _, keep = np.unique(ids, return_index=True)
    ids, vecs = ids[keep], vecs[keep]

    np.save(args.out / "embeddings.npy", vecs)
    (args.out / "ids.json").write_text(json.dumps([int(i) for i in ids]))
    (args.out / "manifest.json").write_text(json.dumps({
        "backbone": args.backbone,
        "image_size": args.image_size,
        "iiif_width": args.iiif_width,
        "fingerprint": fp,
        "count": int(len(ids)),
        "dim": int(vecs.shape[1]),
        "corpus_rows": len(corpus),
        "failed": failed,
        "normalized": True,
        "preprocessor": backend,
    }, indent=2))

    print(f"\nD3: {len(ids):,} vectors, dim {vecs.shape[1]}, "
          f"{failed} images unfetchable")
    print(f"    {args.out/'embeddings.npy'}  "
          f"({vecs.nbytes/1e6:.1f} MB)")
    print(f"    copy back embeddings.npy, ids.json, manifest.json")
    print(f"    the image cache at {cache} is scratch, delete it")


if __name__ == "__main__":
    main()
