"""Map layouts for D6: one 2D position per work, per facet.

Precomputed rather than done in the browser. Three facets x 9,101 works is
small on the wire, identical for every visitor, and keeps a facet switch
free of a few hundred milliseconds of layout jank.

The facet and the embedding work at different scales and the map uses
both. The facet decides which region a work belongs to; its embedding
decides where inside that region it sits. Region centres are themselves
placed by similarity, so neighbouring regions are genuinely related and
the blend zone between two of them is real ambiguity rather than
decoration.

Placing period regions by similarity rather than by date was checked, not
assumed: across the 13 period bins with 20+ works, the principal axis of
the similarity layout correlates with chronology at Spearman 0.995, and
13 of 13 bins are most similar to an adjacent bin against a chance rate
near 2/13. The map recovers chronology from pixels alone.
"""

from __future__ import annotations

import numpy as np

from pipeline.taxonomy import OUTSIDE_TAXONOMY, period_label

# A region holding fewer than this is a speck on the map; 23 of the 35
# countries are below it, together holding 292 works.
MIN_REGION = 50

# Everything below works in a 0..1 square holding a disc of radius 0.5.
DISC_R = 0.5
# Fraction of the disc the works may occupy. Trades density against
# overlap and against how faithfully region order survives: 0.20 gives a
# denser map but chronology at 0.84, 0.08 gives 0.92 and a sparse one.
PACKING = 0.30
RELAX_ITERS = 400
RELAX_STEP = 0.5


def work_radius(n: int) -> float:
    """Per-work radius that lets n works fill PACKING of the disc."""
    return float(np.sqrt(PACKING * DISC_R ** 2 / max(n, 1)))


def fitted_radius(pos: np.ndarray, nominal: float) -> float:
    """The tile radius this layout can actually carry.

    work_radius() assumes the works fill a disc, which they do when the
    map is round. A left-to-right timeline is a ribbon ten times wider
    than it is tall — an eighth of that area — so the same tiles crammed
    into it overlapped at 0.82 of their own width. Reading the spacing off
    the finished layout instead means the tile size follows whatever shape
    the facet turned out to be.
    """
    from scipy.spatial import cKDTree

    if len(pos) < 2:
        return nominal
    gaps = cKDTree(pos).query(pos, k=2)[0][:, 1]
    # Half the typical gap, so neighbours touch rather than overlap.
    return float(min(nominal, np.median(gaps) / 2))


def footprint_scale(works: list[dict]) -> np.ndarray:
    """Each work's collision radius relative to the nominal one.

    A tile is drawn at its true aspect inside a square box, so a 1:3 print
    covers a third of that box. Reserving the whole square for it spaced
    the map out by the part of the box the picture never reaches. Scaling
    by the square root of the fraction actually covered lets narrow works
    sit closer, which is most of this corpus — prints are 64% of it.
    """
    ar = np.array([w.get("ar") or 1.0 for w in works], dtype=np.float64)
    fill = np.minimum(ar, 1.0 / np.maximum(ar, 1e-6))
    return np.sqrt(np.clip(fill, 0.25, 1.0))


FALLBACK_REGIONS = {"Other Asia", "Other Europe", "Africa",
                    "Latin America", "Oceania"}


def group_country(country: str, small: set[str]) -> str:
    """Countries too thin to be their own region fall back to a bucket."""
    if country not in small:
        return country
    if country in {"Japan", "China", "Korea", "Mongolia", "Tibet", "Nepal",
                   "India", "Pakistan", "Afghanistan", "Iran", "Uzbekistan",
                   "Turkey"}:
        return "Other Asia"
    if country in {"Egypt", "Ethiopia"}:
        return "Africa"
    if country in {"Mexico", "Peru"}:
        return "Latin America"
    if country in {"Australia"}:
        return "Oceania"
    return "Other Europe"


SIMILARITY = "similarity"


def facet_values(works: list[dict], facet: str) -> list[str]:
    """One region label per work, aligned to `works`."""
    if facet == SIMILARITY:
        return ["All works"] * len(works)
    if facet == "period":
        # The midpoint bin, not the overlap set: a region layout needs one
        # home per work. period_bins still drives filtering.
        #
        # Bins below MIN_REGION are merged into the ends, and a merged end
        # only earns its own region if it is big enough to be one — a
        # labelled region for a single work is noise, not information.
        counts: dict[int, int] = {}
        for w in works:
            counts[w["pbin"]] = counts.get(w["pbin"], 0) + 1
        keep = {b for b, n in counts.items() if n >= MIN_REGION}
        if not keep:
            keep = {max(counts, key=lambda b: counts[b])}
        first, last = min(keep), max(keep)

        ends = {"before": 0, "after": 0}
        for w in works:
            if w["pbin"] in keep:
                continue
            ends["before" if w["pbin"] < first else "after"] += 1

        out = []
        for w in works:
            b = w["pbin"]
            if b in keep:
                out.append(period_label(b))
            elif b < first:
                out.append(f"Before {period_label(first)}"
                           if ends["before"] >= MIN_REGION else period_label(first))
            else:
                out.append(f"{period_label(last)} onward"
                           if ends["after"] >= MIN_REGION else period_label(last))
        return out
    if facet == "style":
        # The 43-class head predicts 21 styles fewer than 50 times each on
        # this corpus — 213 works between them, nearly all postwar labels
        # turning up once or twice on a collection that stops at 1900.
        # A region of one work is a speck with a label, so they share one.
        #
        # Only the region is grouped. Each work keeps its own predicted
        # style in the bundle, so the detail panel still says Precisionism
        # even though the map does not draw a Precisionism region.
        counts: dict[str, int] = {}
        for w in works:
            s = w["s"] or OUTSIDE_TAXONOMY
            counts[s] = counts.get(s, 0) + 1
        thin = {s for s, n in counts.items()
                if n < MIN_REGION and s != OUTSIDE_TAXONOMY}
        return [("Other styles" if (w["s"] or OUTSIDE_TAXONOMY) in thin
                 else (w["s"] or OUTSIDE_TAXONOMY)) for w in works]
    if facet == "country":
        counts: dict[str, int] = {}
        for w in works:
            counts[w["c"]] = counts.get(w["c"], 0) + 1
        small = {c for c, n in counts.items() if n < MIN_REGION}
        grouped = [group_country(w["c"], small) for w in works]

        # The continental fallbacks can themselves come out tiny — Oceania
        # and Latin America hold one work each — and a region of one is
        # not a region. Anything still short is swept into the largest
        # fallback rather than getting its own label.
        after: dict[str, int] = {}
        for g in grouped:
            after[g] = after.get(g, 0) + 1
        fallbacks = {g for g in after if g in FALLBACK_REGIONS}
        thin = {g for g in fallbacks if after[g] < MIN_REGION}
        if thin:
            survivors = fallbacks - thin
            target = (max(survivors, key=lambda g: after[g]) if survivors
                      else max(after, key=lambda g: after[g]))
            grouped = [target if g in thin else g for g in grouped]
        return grouped
    raise ValueError(f"unknown facet {facet!r}")


def region_centres(vecs: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """Place each region by how its works look, not by its name.

    Regions whose works resemble each other end up adjacent, which is what
    makes a boundary between two regions meaningful.
    """
    names = sorted(set(labels))
    arr = np.array(labels)
    means = np.vstack([vecs[arr == n].mean(axis=0) for n in names])
    means /= np.linalg.norm(means, axis=1, keepdims=True)

    if len(names) == 1:
        return {names[0]: np.zeros(2)}
    dissim = 1.0 - means @ means.T
    np.fill_diagonal(dissim, 0.0)

    # Classical MDS (principal coordinates): double-centre the squared
    # dissimilarities and take the top two eigenvectors. Deterministic and
    # closed-form, unlike sklearn's iterative SMACOF, which lands in a
    # different local minimum per seed — with it the map recovered
    # chronology at 0.78, with this at 0.99, on identical input.
    sq = dissim ** 2
    n = len(names)
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ sq @ centring
    vals, vecs2 = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1][:2]
    xy = vecs2[:, order] * np.sqrt(np.clip(vals[order], 0, None))

    xy -= xy.mean(axis=0)
    span = np.abs(xy).max()
    if span > 0:
        xy = xy / span * DISC_R
    return dict(zip(names, xy))


# Facets whose regions have a genuine one-dimensional order, laid out
# along an arc instead of freely in the plane.
ARC_FACETS = {"period"}
ARC_SWEEP = 1.5 * np.pi     # 270 degrees, so the two ends never meet
# Bins are drawn well inside one another, so a century boundary reads as a
# transition rather than a seam.
ARC_OVERLAP = 0.58
# How wide a bin is across the arc, as a fraction of its radius. Narrow,
# because the arc is the axis that carries meaning here and a wide band
# just thins the works out across space that says nothing.
ARC_BAND = 0.34
# How much of the along-arc position is the date itself rather than its
# rank. All date crowds the busy years; all rank spaces everything alike
# and flattens the century into a slab.
ARC_DATE_WEIGHT = 0.28
# Rounds of relax-then-put-back. More rounds hold the band more
# tightly; each one costs a fraction of the relaxation budget.
ARC_CONSTRAIN_ROUNDS = 8
# Fraction of a century's works forming the neighbourhood a work is
# ranked against. Small enough that the band's width tracks local
# density, large enough that the rank is not noise.
ARC_WINDOW = 0.015


def _period_sort_key(label: str) -> int:
    """Chronological order of a period label, for the merged end buckets
    and ordinary century labels alike. Used only to decide which way round
    to draw the arc, never which order the regions go in."""
    if label.startswith("Before"):
        return -9999
    if label.endswith("onward"):
        return 9999
    digits = "".join(c for c in label.split()[0] if c.isdigit())
    century = int(digits) if digits else 0
    return (century - 1) * 100 * (-1 if "BCE" in label else 1)


def arc_centres(mds: dict[str, np.ndarray], sizes: dict[str, int],
                earliest_first: list[str] | None = None,
                ) -> tuple[dict[str, np.ndarray], dict[str, float],
                           dict[str, np.ndarray]]:
    """Place regions along an arc, ordered by the first similarity axis.

    For period the similarity structure is 93% one-dimensional and that
    axis tracks chronology at 0.993, so a free 2D placement spends its
    second axis on 7% signal and then inflates it to 15% by shoving
    regions sideways to make room — displacing each one by about its own
    radius. The result invites the question "what is the y axis?" and has
    no good answer.

    The order here still comes from the pixels, not from the dates: the
    regions are sorted along MDS axis 1 exactly as before. Only the
    direction of travel consults the labels, so that time reads forwards
    rather than backwards, which changes nothing about what was found.

    An arc rather than a line because a straight chronological line needs
    a span of about 3, at which point normalising shrinks every region
    below the area its works need.
    """
    names = list(mds)
    M = np.vstack([mds[n] for n in names])
    centred = M - M.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along = centred @ vt[0]

    if earliest_first is not None:
        rank = {n: i for i, n in enumerate(earliest_first)}
        known = [i for i, n in enumerate(names) if n in rank]
        if len(known) > 2:
            a = np.array([rank[names[i]] for i in known], dtype=float)
            b = along[known]
            if np.corrcoef(a, b)[0, 1] < 0:
                along = -along

    order = np.argsort(along)
    total = sum(sizes.values())
    rad = {n: DISC_R * np.sqrt(sizes[n] / total) for n in names}

    # Walk left to right. A timeline read the way a timeline is read, and
    # it retires the overshoot problem outright: every century shares one
    # orientation now, so no band can leave the line its neighbour is on.
    # Bins are spaced at less than their own width, so they interpenetrate
    # rather than sitting as separate blocks.
    centres: dict[str, np.ndarray] = {}
    travelled = 0.0
    for i in order:
        name = names[i]
        travelled += rad[name] * ARC_OVERLAP
        centres[name] = np.array([travelled, 0.0])
        travelled += rad[name] * ARC_OVERLAP

    span = max(travelled, 1e-9)
    centres = {n: np.array([c[0] - span / 2, 0.0]) for n, c in centres.items()}
    return centres, rad, {"span": span}


def pack_regions(centres: dict[str, np.ndarray], sizes: dict[str, int],
                 iters: int = 300, anchor: float = 0.08
                 ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Give each region room proportional to how many works it holds.

    MDS says which regions belong near each other but knows nothing about
    how much space each needs, so a region of 2,000 lands on top of its
    neighbours. Two obvious repairs both fail:

    - Free repulsion resolves the overlap but scrambles the ordering. The
      chronology the period map recovers fell from 0.99 to 0.55, because
      large regions shoved small ones wherever there was room.
    - Scaling the whole arrangement up preserves ordering exactly, but one
      tight pair of large regions sets the scale for everything, and after
      normalising, each region is so small that relaxing the works inside
      it pushes them into their neighbours.

    So: repel, then pull each region back toward where similarity says it
    belongs. The anchor keeps the ordering while the repulsion keeps the
    spacing, and the scale is set by a high percentile rather than the
    single worst pair.
    """
    names = list(centres)
    total = sum(sizes.values())
    home = np.vstack([centres[n] for n in names])
    rad = np.array([DISC_R * np.sqrt(sizes[n] / total) for n in names])

    ratios = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            gap = float(np.linalg.norm(home[i] - home[j]))
            if gap > 1e-9:
                ratios.append((rad[i] + rad[j]) / gap)
    if ratios:
        home = home * max(1.0, float(np.quantile(ratios, 0.90)))

    # A region can only hold its works if it stays big enough relative to
    # them, and normalising to 0..1 divides every region by the layout's
    # span. Working that through, a region of n works needs
    #     DISC_R*sqrt(n/total)/span  >=  sqrt(n) * work_radius
    # which reduces to span <= 1/sqrt(PACKING), independent of n. Past that
    # point the works spill out of their region and the map turns to soup.
    # Regions that still overlap are fine: fluid boundaries are the design.
    limit = 0.85 / np.sqrt(PACKING)
    span = float(np.abs(home).max()) * 2
    if span > limit:
        home = home * (limit / span)

    pos = home.copy()
    for _ in range(iters):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                delta = pos[i] - pos[j]
                dist = float(np.linalg.norm(delta))
                want = rad[i] + rad[j]
                if 1e-9 < dist < want:
                    push = delta / dist * (want - dist) * 0.5
                    pos[i] += push
                    pos[j] -= push
        pos += (home - pos) * anchor
    return dict(zip(names, pos)), dict(zip(names, rad))


# How much of a region's own spread defines its edge. Normalising by the
# maximum lets a single atypical work set the scale for everyone: India
# has a couple of extreme outliers, and dividing by them crushed its other
# 96 works into 9% of the region's radius, where every other country sits
# at 54-82%. Relaxation then had to shove them apart, and the region came
# out looking scattered and nearly empty. A high percentile is robust to
# that; anything beyond it is pulled back to the rim.
EXTENT_PERCENTILE = 92


def _fit_unit(local: np.ndarray) -> np.ndarray:
    """Scale a region's offsets so most sit inside the unit circle, with
    outliers clamped to its edge rather than deciding its size."""
    radius = np.linalg.norm(local, axis=1)
    scale = float(np.percentile(radius, EXTENT_PERCENTILE))
    if scale <= 0:
        scale = float(radius.max()) or 1.0
    local = local / scale
    over = np.linalg.norm(local, axis=1)
    beyond = over > 1.0
    if beyond.any():
        local[beyond] /= over[beyond][:, None]
    return local


def place(xy_local: np.ndarray, labels: list[str],
          centres: dict[str, np.ndarray], radii: dict[str, float],
          arc: dict | None = None,
          along_key: np.ndarray | None = None,
          tile_radius: float = 0.002) -> np.ndarray:
    """Work positions: region centre plus the work's own offset within it.

    The offset is the work's UMAP position, so visually similar works sit
    together inside a region and the map has structure at both zoom levels.
    """
    arr = np.array(labels)
    out = np.zeros((len(labels), 2), dtype=np.float64)
    for name, centre in centres.items():
        mask = arr == name
        local = xy_local[mask] - xy_local[mask].mean(axis=0)
        local = _fit_unit(local)

        if arc is None or along_key is None:
            out[mask] = centre + local * radii[name] * 0.92
            continue

        # Along the arc, the date. Across it, the work's rank among its
        # own date-neighbours — so the two axes are not independent, and
        # the band can have a shape.
        #
        # This is the fourth attempt and the first that is not a tweak.
        # Raw embedding offsets gave wisps; a uniform rank gave a slab; a
        # tapered rank gave a slab with softer density. All three computed
        # `across` over the whole century at once, independent of `along`,
        # and two independent distributions draw a rectangle however their
        # densities are shaped. Measured on the 19th century: half-width
        # varied 1.14x end to end and |across| correlated with along at
        # -0.066, which is the arithmetic of a slab.
        #
        # Ranking within a sliding window of date-neighbours couples them.
        # The band's half-width at a date follows the square root of how
        # many works sit near that date, so the ribbon swells where the
        # collection is thick — an outline that reports something rather
        # than one that has been roughened to look organic.
        # Along the timeline is x and across it is y, the same for every
        # century — which is what keeps the bands flush with each other
        # instead of each overshooting the next.
        u = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        keys = along_key[mask].astype(np.float64)
        order = np.argsort(keys, kind="stable")
        n = len(order)

        # Position along: the date, smoothed by blending with rank, so a
        # year everything is dated to does not become a column.
        rank_pos = np.empty(n)
        rank_pos[order] = np.arange(n) / max(n - 1, 1)
        lo, hi = keys.min(), keys.max()
        by_date = (keys - lo) / max(hi - lo, 1e-9)
        along = ((ARC_DATE_WEIGHT * by_date
                  + (1 - ARC_DATE_WEIGHT) * rank_pos) - 0.5) * 2.0

        # Local density: works within a window of dates either side.
        window = max(int(n * ARC_WINDOW), 12)
        half = window // 2
        idx_in_order = np.empty(n, dtype=np.int64)
        idx_in_order[order] = np.arange(n)
        starts = np.clip(idx_in_order - half, 0, max(n - window, 0))
        span = np.maximum(keys[order][np.clip(starts + window - 1, 0, n - 1)]
                          - keys[order][starts], 1e-6)
        density = float(window) / span                 # works per year, locally
        # Linear in density, not its square root: the root damped the
        # signal so far that the band still read as a slab (1.85x end
        # to end against the 2x this needed to clear).
        width = density / np.median(density)
        # The upper bound barely matters now: capping how far a tied
        # date may spread is the binding constraint, and that cap is a
        # uniform-density rule, so it flattens the band on its own. Width
        # variation sits near 1.4x whether this is 2.0 or 4.0.
        width = np.clip(width, 0.30, 2.6)              # indexed in date order

        # Position across: rank within the window, centred, scaled by that
        # local width — so the edge follows the data instead of the frame.
        proj = local @ perp                     # how far off-spine each work is
        proj_sorted = proj[order]               # walked in date order
        across_sorted = np.empty(n)
        for k in range(n):
            a = min(max(k - half, 0), max(n - window, 0))
            b = min(a + window, n)
            neighbourhood = proj_sorted[a:b]
            across_sorted[k] = (np.count_nonzero(neighbourhood < proj_sorted[k])
                                / max(len(neighbourhood) - 1, 1) - 0.5) * 2.0
        across = np.empty(n)
        across[order] = across_sorted * width[order]

        # A tied date must be a blob, not a spoke. 437 works are dated
        # exactly 1849; they share an along-arc position, so spreading
        # them across the full local width drew a line 39 times longer
        # than it was wide, and with the density term widening the band
        # it shot outside the region entirely — 2,135 works ended up past
        # their own radius.
        #
        # A group of m tiles laid out squarely needs about sqrt(m)*2r on
        # each side, so that is all the across-extent a tied date gets.
        # Relaxation rounds the rest out.
        span_unit = max(radii[name] * ARC_BAND, 1e-9)
        uniq, inverse, counts = np.unique(keys, return_inverse=True,
                                          return_counts=True)
        for g in np.flatnonzero(counts > 1):
            members = np.flatnonzero(inverse == g)
            room = np.sqrt(len(members)) * 2.0 * tile_radius / span_unit
            extent = np.abs(across[members]).max()
            if extent > room / 2:
                across[members] *= (room / 2) / extent

        # And nothing leaves the band. A date with 437 works needs more
        # room than the band is thick, and given a free radial axis it
        # took it — the biggest tied dates reached 1.2 to 1.5 times the
        # thickness and their blobs read as bumps on the ribbon's edge.
        # Held to the band, the crowding resolves along the arc instead,
        # where there is somewhere to go.
        across = np.clip(across, -1.0, 1.0)

        out[mask] = (centre
                     + np.outer(along * radii[name], u)
                     + np.outer(across * radii[name] * ARC_BAND, perp))
    return out


def relax(pos: np.ndarray, radius: float, iters: int = RELAX_ITERS,
          step: float = RELAX_STEP, scale: np.ndarray | None = None) -> np.ndarray:
    """Push overlapping works apart so thumbnails stay readable.

    Region membership is already baked into the starting positions, so a
    purely local repulsion preserves it while removing collisions.
    """
    from scipy.spatial import cKDTree

    pos = pos.copy()
    # Per-work radii, not one radius for everything. Tiles are drawn at
    # their true aspect, so a 1:3 hanging scroll or print fills a third of
    # the square that a uniform circle reserves for it — and prints are
    # 64% of this corpus. Spacing them all as if they were square left the
    # map looking sparse when the works were in fact touching; the gap was
    # reserved space, not distance.
    rad = np.full(len(pos), radius) if scale is None else radius * scale
    largest = float(rad.max()) * 2

    for _ in range(iters):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(largest, output_type="ndarray")
        if len(pairs) == 0:
            break
        a, b = pairs[:, 0], pairs[:, 1]
        want = rad[a] + rad[b]
        delta = pos[a] - pos[b]
        dist = np.linalg.norm(delta, axis=1)
        dist[dist == 0] = 1e-9
        touching = dist < want
        if not touching.any():
            break
        a, b = a[touching], b[touching]
        delta, dist, want = delta[touching], dist[touching], want[touching]
        push = delta / dist[:, None] * (want - dist)[:, None] * step
        np.add.at(pos, a, push)
        np.subtract.at(pos, b, push)
    return pos


OUTLINE_BINS = 72
OUTLINE_PCT = 98          # reach far enough that the region owns its works
OUTLINE_SMOOTH = 6        # angular bins either side, so the edge is not jagged
OUTLINE_SLACK = 1.16      # and a margin beyond that, so tiles sit well inside


def region_hues(centres: dict[str, np.ndarray]) -> dict[str, float]:
    """One hue per region, spread evenly but in similarity order.

    Taking hue straight from the angle of a region's position bunched them:
    regions are not evenly distributed around the layout, so two thirds of
    the style map came out blue. Ranking by angle and then spreading the
    ranks over the full circle keeps neighbouring regions neighbouring in
    hue — which is the property worth having, since adjacency here means
    visual similarity — while using the whole wheel.
    """
    if not centres:
        return {}
    mean = np.mean(np.vstack(list(centres.values())), axis=0)
    order = sorted(centres, key=lambda n: float(
        np.arctan2(centres[n][1] - mean[1], centres[n][0] - mean[0])))
    return {name: round(i / len(order) * 360.0, 1)
            for i, name in enumerate(order)}


def region_outline(points: np.ndarray, centre: np.ndarray,
                   fallback: float) -> list[list[float]]:
    """A closed outline that follows the works inside a region.

    A circle is the wrong shape: these regions are UMAP offsets scaled into
    a disc, so they come out lobed and concave and a circle either clips
    them or floats far off their edge.

    They are, however, near enough star-shaped about their own centre —
    that is how place() builds them — so the boundary can be a radial
    profile: for each angular bin, how far out the works actually reach.
    A high percentile rather than the maximum, so one stray work does not
    pull a spike, and a circular smoothing pass so the edge reads as a
    shape rather than a sawtooth.
    """
    rel = points - centre
    radius = np.linalg.norm(rel, axis=1)
    angle = np.arctan2(rel[:, 1], rel[:, 0])
    bins = np.clip(((angle + np.pi) / (2 * np.pi) * OUTLINE_BINS).astype(int),
                   0, OUTLINE_BINS - 1)

    reach = np.full(OUTLINE_BINS, np.nan)
    for b in range(OUTLINE_BINS):
        here = radius[bins == b]
        if len(here):
            reach[b] = np.percentile(here, OUTLINE_PCT)

    if np.all(np.isnan(reach)):
        reach = np.full(OUTLINE_BINS, fallback)
    else:
        # Empty bins borrow from their neighbours rather than collapsing.
        idx = np.arange(OUTLINE_BINS)
        good = ~np.isnan(reach)
        reach = np.interp(idx, idx[good], reach[good], period=OUTLINE_BINS)

    k = OUTLINE_SMOOTH
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    smooth = np.convolve(np.r_[reach[-k:], reach, reach[:k]], kernel, "valid")
    # Air beyond the furthest works, so tiles sit well inside their own
    # region rather than straddling its edge.
    smooth = smooth * OUTLINE_SLACK + fallback * 0.06

    out = []
    for b in range(OUTLINE_BINS):
        a = (b + 0.5) / OUTLINE_BINS * 2 * np.pi - np.pi
        out.append([round(float(centre[0] + smooth[b] * np.cos(a)), 4),
                    round(float(centre[1] + smooth[b] * np.sin(a)), 4)])
    return out


def overlap_count(pos: np.ndarray, radius: float) -> int:
    from scipy.spatial import cKDTree
    return len(cKDTree(pos).query_pairs(radius * 2 * 0.95))


def build(vecs: np.ndarray, works: list[dict], facet: str) -> tuple:
    """(positions in 0..1, region centres in 0..1) for one facet."""
    xy_local = np.array([w["xy"] for w in works], dtype=np.float64)
    labels = facet_values(works, facet)

    if facet == SIMILARITY:
        # No regions at all: the embedding alone decides position. This is
        # the only layout where being next to something means the two works
        # look alike, and it keeps roughly three times as many true
        # neighbours adjacent as a grouped one — 1.8 of a work's top ten
        # against 0.6 — because nothing is pulled away to join a region.
        def unit(p):
            lo = p.min(axis=0)
            return (p - lo) / float((p.max(axis=0) - lo).max())

        pos = unit(relax(unit(xy_local), radius=work_radius(len(labels)),
                         scale=footprint_scale(works)))
        return pos, {}, {}, labels, fitted_radius(
            pos, work_radius(len(labels)))
    sizes: dict[str, int] = {}
    for l in labels:
        sizes[l] = sizes.get(l, 0) + 1

    mds = region_centres(vecs, labels)
    arc = None
    along_key = None
    if facet in ARC_FACETS:
        # No packing pass: the arc already spaces them, and packing was
        # what displaced regions and muddied the axis.
        centres, radii, arc = arc_centres(
            mds, sizes, earliest_first=sorted(sizes, key=_period_sort_key))
        along_key = np.array([w.get("my", 0) for w in works], dtype=np.float64)
    else:
        centres, radii = pack_regions(mds, sizes)

    def to_unit(p, c, r):
        lo = p.min(axis=0)
        span = float((p.max(axis=0) - lo).max())
        return ((p - lo) / span,
                {k: (v - lo) / span for k, v in c.items()},
                {k: v / span for k, v in r.items()})

    # Normalize BEFORE relaxing, so the work radius means the same thing
    # during relaxation as it does to the client. Normalizing afterwards
    # rescales every distance and silently undoes the spacing just applied.
    placed = place(xy_local, labels, centres, radii, arc, along_key,
                   work_radius(len(labels)))
    wr = work_radius(len(labels))
    fs = footprint_scale(works)

    if arc is not None:
        # Relax inside the band, not merely near it.
        #
        # Clamping the placement was not enough, because relaxation runs
        # afterwards and knows nothing about where the band is: it pushed
        # the biggest tied dates straight out, 437 works dated 1849
        # reaching 1.4 times the band's thickness and reading as a bump.
        # Stretching the radial axis to make radial moves expensive was
        # worse — separations won in the stretched space are divided away
        # coming back, so works ended up overlapping at 0.79 tile-widths.
        #
        # So: relax honestly in arc-length coordinates, then put anything
        # that left the band back, and repeat. Each round the crowding has
        # one less direction to escape in and resolves along the ribbon,
        # which is the only place with room.
        band = np.array([radii[l] * ARC_BAND for l in labels])
        for _ in range(ARC_CONSTRAIN_ROUNDS):
            placed = relax(placed, radius=wr, scale=fs,
                           iters=RELAX_ITERS // ARC_CONSTRAIN_ROUNDS)
            placed[:, 1] = np.clip(placed[:, 1], -band, band)
        pos, centres, radii = to_unit(placed, centres, radii)
    else:
        pos, centres, radii = to_unit(placed, centres, radii)
        pos = relax(pos, radius=wr, scale=fs)
    pos, centres, radii = to_unit(pos, centres, radii)
    # The tile size this facet can actually carry, read off the finished
    # layout rather than assumed from a disc.
    tile = fitted_radius(pos, work_radius(len(labels)))
    return pos, centres, radii, labels, tile
