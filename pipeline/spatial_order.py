"""Pack order for the high-resolution atlas.

A lazily-loaded atlas only helps if the tiles visible at one moment live
in a handful of sheets. The 64px atlas is packed in ids.json order, which
bears no relation to where a work sits on the map, so any screenful is
scattered across every sheet and lazy loading would fetch all of them.

Packing along a Hilbert curve through the UMAP plane fixes that for every
facet at once. Each layout places a work at its region's centre plus its
own UMAP offset, so two works close in UMAP are close on screen whenever
they share a region — which is the case that matters, since that is what
you are looking at when zoomed in far enough to need this atlas.
"""

from __future__ import annotations

import numpy as np

BITS = 10          # 1024x1024 grid over the unit square


def _hilbert_d(x: int, y: int, bits: int = BITS) -> int:
    """Distance along a Hilbert curve for one grid cell."""
    d = 0
    s = 1 << (bits - 1)
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:                      # rotate the quadrant
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def hilbert_order(xy: np.ndarray, bits: int = BITS) -> np.ndarray:
    """Indices of `xy` (in 0..1) sorted along a Hilbert curve."""
    grid = np.clip((xy * ((1 << bits) - 1)).astype(np.int64), 0, (1 << bits) - 1)
    keys = np.array([_hilbert_d(int(x), int(y), bits) for x, y in grid])
    return np.argsort(keys, kind="stable")


def locality(order: np.ndarray, xy: np.ndarray, per_sheet: int,
             samples: int = 400, radius: float = 0.03,
             seed: int = 0) -> float:
    """Mean number of sheets a small neighbourhood of the map spans.

    The number this ordering exists to reduce: 1.0 would mean every
    neighbourhood sits in one sheet, and the sheet count would mean the
    ordering achieved nothing.
    """
    slot = np.empty(len(order), dtype=np.int64)
    slot[order] = np.arange(len(order))
    rng = np.random.default_rng(seed)
    spans = []
    for i in rng.choice(len(xy), samples, replace=False):
        near = np.where(np.linalg.norm(xy - xy[i], axis=1) < radius)[0]
        if len(near) < 4:
            continue
        spans.append(len(set((slot[near] // per_sheet).tolist())))
    return float(np.mean(spans)) if spans else float("nan")
