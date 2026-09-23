#!/usr/bin/env python3
"""D5b: pack 32px thumbnails into sprite atlases for the map's mid zoom.

    python3 d5b_atlas.py --ids out/ids.json --out out --cache out/imgcache

Standalone, like d3_embed.py. Run it on whatever machine has the image
cache from D3 still warm and it never touches the network; point it at a
cold directory and it fetches from IIIF itself.

Why an atlas rather than IIIF per tile: zoom-out to zoom-in is the map's
core interaction and has to feel instant. A few hundred IIIF round-trips
per pan would lag exactly where the site should feel good, and would
hammer a museum CDN on every gesture. Three sheets fetched once and cached
forever cost about 2 MB.

Tiles are packed in the order ids.json gives, so a work's index is its
position in that file and the client needs no lookup table: sheet =
index // (grid*grid), then row and column within it. At 9,101 works that
is three sheets, which is few enough that sharding by map region would
save nothing.

Outputs to --out:
    atlas_000.jpg ...   2048x2048 sheets, 64x64 tiles of 32px each
    atlas.json          tile size, grid, sheet count, id order
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

# Fast, CDN-cached form first; the confined form only for works too
# narrow for it, which AIC answers with a 403. See d3_embed.py.
IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/{width},/0/default.jpg"
IIIF_CONFINED = ("https://www.artic.edu/iiif/2/{image_id}"
                 "/full/!{width},{width}/0/default.jpg")

SHEET_PX = 2048
FETCH_WIDTH = 200          # plenty for a 32px tile, cheap if we must fetch


def square(img, size: int, background=(13, 13, 15)):
    """Fit the whole work inside a square cell, centred, without cropping.

    Contain rather than cover. A cover-crop keeps the mosaic perfectly
    regular but cuts the edges off every non-square work, and this corpus
    is full of hanging scrolls at 1:3 — drawing one as a square is a lie
    about the object. The client knows each work's aspect ratio and draws
    only the occupied part of the cell, so the padding never shows.
    """
    w, h = img.size
    fit = size / max(w, h)
    iw, ih = max(1, round(w * fit)), max(1, round(h * fit))
    cell = Image.new("RGB", (size, size), background)
    cell.paste(img.resize((iw, ih), 1), ((size - iw) // 2, (size - ih) // 2))
    return cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=Path, required=True,
                    help="ids.json from D3; its order defines tile indices")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="corpus.jsonl; maps a work id to its IIIF image id")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, help="D3's image cache, if still warm")
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ids = json.loads(args.ids.read_text())
    grid = SHEET_PX // args.tile
    per_sheet = grid * grid
    sheets = (len(ids) + per_sheet - 1) // per_sheet
    args.out.mkdir(parents=True, exist_ok=True)

    image_of: dict[int, str] = {}
    with args.corpus.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("image_id"):
                    image_of[r["id"]] = r["image_id"]

    warm = args.cache is not None and args.cache.is_dir()
    print(f"{len(ids):,} works -> {sheets} sheet(s) of {grid}x{grid} "
          f"at {args.tile}px")
    print(f"images from {'the D3 cache' if warm else 'IIIF'}\n")

    session = None
    if not warm:
        import httpx
        session = httpx.Client(headers={"User-Agent": "ArtiFact/0.1"},
                               follow_redirects=True)

    def load(artwork_id: int):
        """Tile bytes for one work: cache first, IIIF only if it must."""
        image_id = image_of.get(artwork_id)
        if not image_id:
            return None
        if warm:
            path = args.cache / f"{image_id}.jpg"
            if path.exists():
                return path.read_bytes()
        if session is None:
            return None
        for template in (IIIF, IIIF_CONFINED):
            try:
                r = session.get(
                    template.format(image_id=image_id, width=FETCH_WIDTH),
                    timeout=60.0)
            except Exception:
                return None
            if r.status_code == 200:
                return r.content
            if r.status_code != 403:
                return None
        return None

    missing = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sheet in range(sheets):
            chunk = ids[sheet * per_sheet:(sheet + 1) * per_sheet]
            canvas = Image.new("RGB", (SHEET_PX, SHEET_PX), (17, 17, 17))
            for n, (artwork_id, blob) in enumerate(zip(chunk,
                                                       pool.map(load, chunk))):
                if blob is None:
                    missing += 1
                    continue
                try:
                    img = Image.open(io.BytesIO(blob)).convert("RGB")
                except Exception:
                    missing += 1
                    continue
                canvas.paste(square(img, args.tile),
                             ((n % grid) * args.tile, (n // grid) * args.tile))
            path = args.out / f"atlas_{sheet:03d}.jpg"
            canvas.save(path, "JPEG", quality=args.quality, optimize=True)
            print(f"  {path.name}  {len(chunk):>5,} tiles  "
                  f"{path.stat().st_size/1e6:5.2f} MB")

    (args.out / "atlas.json").write_text(json.dumps({
        "tile": args.tile,
        "sheet_px": SHEET_PX,
        "grid": grid,
        "per_sheet": per_sheet,
        "sheets": sheets,
        "count": len(ids),
        "missing": missing,
        "note": "tile index == position in ids.json; "
                "sheet = i // per_sheet, col = i % grid, row = (i % per_sheet) // grid",
    }, indent=2))

    total = sum((args.out / f"atlas_{s:03d}.jpg").stat().st_size
                for s in range(sheets))
    print(f"\nD5b: {sheets} sheets, {total/1e6:.2f} MB total, "
          f"{missing} tiles missing")
    print(f"     copy back atlas_*.jpg and atlas.json")


if __name__ == "__main__":
    main()
