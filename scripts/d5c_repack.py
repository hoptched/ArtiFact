#!/usr/bin/env python3
"""D5c: re-pack the 64px atlas into map order.

    python3 scripts/d5c_repack.py

The sheets were packed in artwork-id order, which has nothing to do with
where a work sits on the map, so 400 neighbouring works spanned 23.6 of
the 25 sheets. Loading a screenful meant loading the whole atlas, which
is why the client loads all of it up front and every visitor pays 24 MB
before seeing anything.

Packed along the same Hilbert curve the high-resolution atlas uses, the
same 400 works span 2.8 sheets under Similarity, 10.2 under Place and
9.6 under Style. Period stays poor at 18.4, because its ribbon reorders
the corpus completely; one curve can only follow one arrangement.

No images are fetched. Every tile is already in the existing sheets, so
this only moves them, and both atlases end up sharing one slot table.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pipeline.config import Config


def main() -> None:
    cfg = Config.load()
    web = Path("web/public/atlas")
    meta = json.loads((web / "atlas.json").read_text())
    tile, grid, per = meta["tile"], meta["grid"], meta["per_sheet"]

    ids = json.loads((cfg.paths.processed / "ids.json").read_text())
    order = json.loads((cfg.paths.processed / "hires_order.json").read_text())
    slot_of_id = {aid: n for n, aid in enumerate(order)}
    # Work index -> the slot it should occupy.
    slots = [slot_of_id[aid] for aid in ids]
    assert sorted(slots) == list(range(len(ids))), "slots are not a permutation"

    old = [Image.open(web / f"atlas_{s:03d}.jpg").convert("RGB")
           for s in range(meta["sheets"])]
    # Slot -> the work index that belongs in it, so each new sheet can be
    # filled in one pass over its own thousand positions.
    work_at = [0] * len(slots)
    for i, s in enumerate(slots):
        work_at[s] = i

    print(f"re-packing {len(ids):,} tiles across {meta['sheets']} sheets")
    for s in range(meta["sheets"]):
        canvas = Image.new("RGB", (meta["sheet_px"], meta["sheet_px"]), (13, 13, 15))
        for n in range(per):
            slot = s * per + n
            if slot >= len(work_at):
                break
            i = work_at[slot]
            src = old[i // per]
            sx, sy = (i % per) % grid * tile, (i % per) // grid * tile
            cell = src.crop((sx, sy, sx + tile, sy + tile))
            canvas.paste(cell, ((n % grid) * tile, (n // grid) * tile))
        canvas.save(web / f"atlas_{s:03d}.jpg", quality=88, optimize=True)
        print(f"  sheet {s + 1}/{meta['sheets']}", end="\r")

    meta["ordered"] = True
    meta["note"] = ("tile index is the work's slot in data/atlas_slots.json, "
                    "the same order the high-resolution atlas uses")
    (web / "atlas.json").write_text(json.dumps(meta, indent=2))
    out = Path("web/public/data/atlas_slots.json")
    out.write_text(json.dumps(slots, separators=(",", ":")))
    print(f"\nwrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
