import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildFields } from "./regionFields";
import type { Bundle } from "./useData";
import type { Facet } from "./types";
import { ImageCache, TIER2_MIN_PX } from "./imageCache";
import { HiResAtlas } from "./hiresAtlas";
import { placeAmong } from "./compare";
import texSimilarity from "./tex/similarity.png";
import texPeriod from "./tex/period.png";
import texCountry from "./tex/country.png";
import texStyle from "./tex/style.png";

/**
 * A ground for each grouping.
 *
 * The four came in at wildly different weights — one half solid black,
 * one pale beige line work — so they are greyed, which drops the colour,
 * and then each is scaled until all four carry the same mean ink. The
 * pattern changes with the grouping; how loudly it speaks does not.
 *
 * Fixed to the viewport rather than to the map. Tied to map space it
 * would slide under the works and grow with the zoom, which reads as
 * another layer of the data; held still it is what the map is drawn on.
 */
const GROUND: Record<Facet, string> = {
  similarity: texSimilarity,
  period: texPeriod,
  country: texCountry,
  style: texStyle,
};
const GROUND_ALPHA = 0.05;
/** How large the tile is drawn against how it was cut. Applied at the
 *  pattern rather than by resampling the files, so the tiles keep their
 *  full resolution and this stays one number to turn. */
const GROUND_SCALE = 0.5;

interface View { x: number; y: number; scale: number }

// Below this many pixels per tile there is no point paying for a texture
// lookup — a coloured square and a thumbnail are indistinguishable.
const THUMB_MIN_PX = 3.5;
/** Tile size in the reduced atlas: a quarter of the 64px cell. */
const MIP_TILE = 16;
/** How small the selected work is allowed to get, in CSS pixels. Zoomed
 *  out it is one speck among twenty thousand, and the thing you just
 *  chose should not be the hardest thing on the map to find. */
const SELECTED_MIN_PX = 26;
// Below this many CSS pixels a tile is too small to aim at, so a click
// inside a region means the region rather than whichever work happened to
// be nearest the cursor.
const WORK_CLICK_MIN_PX = 13;
/** How much of the viewport a region fills once you have flown to it. */
const FOCUS_FILL = 2.6;
/** Floor on a region's radius, so a tiny one does not fill the screen. */
const MIN_FOCUS_R = 0.02;
// Period is a timeline, and its regions already read as one continuous
// band running along the arc. Tinting each century made the sequence look
// like seven separate things. It stays clickable, just uncoloured.
const TINTED_FACETS = new Set<Facet>(["country", "style"]);
// Where a click on a region travels to it. Not the timeline: its bins are
// bands across the whole ribbon rather than places you aim at, and a
// century is a slice of a continuum, so flying to one answers a question
// nobody asked by clicking there. Similarity has no regions at all.
const REGION_CLICK_FACETS = new Set<Facet>(["country", "style"]);
/** Where a region name has faded out completely. */
const LABEL_MAX_SCALE = 4200;
/** Where it starts going. Zooming in is a move towards the works and
 *  away from the grouping, so the name should thin out as you go rather
 *  than switch off at a threshold you cannot see coming. */
const LABEL_FADE_FROM = 1600;
const ANIM_MS = 750;
/** How long the region fields take to appear once the works have settled. */
const FIELD_FADE_MS = 550;
const FLY_MS = 900;

/**
 * The halo that marks a work: the selection, and the uploaded picture.
 *
 * It has to be hollow, because the work sits inside it and a filled blur
 * would simply cover it. So the shape that casts the shadow is an
 * outline, drawn far off-canvas with the shadow offset back by the same
 * amount — only the light lands, never the shape. Three passes, because
 * one is too faint to find against the tiles, and the outline sits
 * slightly proud of the work so the light falls outside its edge.
 */
function glow(ctx: CanvasRenderingContext2D,
              x: number, y: number, w: number, h: number, dpr: number) {
  const OFF = 1e5;
  const gap = 2 * dpr;
  ctx.save();
  ctx.shadowColor = "rgba(245,196,81,0.9)";
  ctx.shadowBlur = 12 * dpr;
  ctx.shadowOffsetX = OFF;
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 3 * dpr;
  for (let pass = 0; pass < 3; pass++) {
    ctx.strokeRect(x - w / 2 - gap - OFF, y - h / 2 - gap,
                   w + gap * 2, h + gap * 2);
  }
  ctx.restore();
}

/** A uniform grid over the layout, so a frame can ask "what is in this
 *  rectangle" instead of walking all 25,515 works. Stored the way a
 *  sparse matrix is — a start offset per cell, and one flat array of work
 *  indices — because 25k small arrays would cost more to chase than the
 *  scan it replaces.
 *
 *  Cells are kept roughly square in map units rather than in grid units:
 *  the timeline is ten times wider than it is tall, and a grid that
 *  ignored that would put a whole column of the ribbon in one cell. */
export interface Grid {
  x0: number; y0: number; cw: number; ch: number;
  nx: number; ny: number;
  start: Int32Array; items: Int32Array;
}

function buildGrid(pts: Float32Array): Grid | null {
  const n = pts.length / 2;
  if (n === 0) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pts.length; i += 2) {
    if (pts[i] < x0) x0 = pts[i];
    if (pts[i] > x1) x1 = pts[i];
    if (pts[i + 1] < y0) y0 = pts[i + 1];
    if (pts[i + 1] > y1) y1 = pts[i + 1];
  }
  const spanX = Math.max(x1 - x0, 1e-6), spanY = Math.max(y1 - y0, 1e-6);
  // About two works per cell: enough that a query returns little waste,
  // not so many cells that walking empty ones dominates.
  const cells = Math.max(1, Math.round(n / 2));
  const nx = Math.max(1, Math.round(Math.sqrt(cells * spanX / spanY)));
  const ny = Math.max(1, Math.ceil(cells / nx));
  const cw = spanX / nx, ch = spanY / ny;

  const cellOf = (i: number) => {
    const gx = Math.min(nx - 1, Math.max(0, ((pts[i * 2] - x0) / cw) | 0));
    const gy = Math.min(ny - 1, Math.max(0, ((pts[i * 2 + 1] - y0) / ch) | 0));
    return gy * nx + gx;
  };

  const start = new Int32Array(nx * ny + 1);
  for (let i = 0; i < n; i++) start[cellOf(i) + 1]++;
  for (let c = 0; c < nx * ny; c++) start[c + 1] += start[c];
  const items = new Int32Array(n);
  const cursor = start.slice(0, nx * ny);
  for (let i = 0; i < n; i++) items[cursor[cellOf(i)]++] = i;
  return { x0, y0, cw, ch, nx, ny, start, items };
}

/** Ray casting, in the map's own 0..1 space so no per-frame screen copy
 *  of the outline is needed. */
function inside(poly: [number, number][], x: number, y: number) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y)
        && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

const easeInOut = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

export interface Focus { x: number; y: number; r: number; key: number }

export interface MapPin {
  img: HTMLImageElement;
  matches: { index: number; similarity: number }[];
  /** What the model made of the picture itself, so the map can put it
   *  where the panel says it belongs. */
  style: string | null;
}

export function MapCanvas({
  bundle, facet, selected, onSelect, focus, pins, activePin, onOpenPin,
  onRegion, rightInset, paused, onNeedAtlas,
}: {
  bundle: Bundle;
  facet: Facet;
  selected: number | null;
  onSelect: (index: number | null) => void;
  focus: Focus | null;
  /** Which uploaded picture is the thing currently selected, if any. */
  activePin: number | null;
  pins: MapPin[];
  onOpenPin: (which: number) => void;
  onRegion: (x: number, y: number, r: number) => void;
  /** How much of the canvas's right edge a panel is covering, in CSS
   *  pixels. The canvas runs the full width of the stage and the panel
   *  sits over it, so without this the map centres itself underneath
   *  the panel and half of what it framed cannot be seen. */
  rightInset: number;
  /** True while something else needs this thread. The map draws 25,515
   *  tiles a frame; giving that up is most of what makes the wait
   *  bearable, and a half-drawn map is worse than a still one. */
  paused: boolean;
  /** Called the first time a thumbnail is wanted. The sheets are not
   *  fetched before that, because the opening view draws flat colour. */
  onNeedAtlas: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<View>({ x: 0.5, y: 0.5, scale: 0 });
  const [ready, setReady] = useState(false);

  // Positions are animated between two precomputed layouts rather than
  // recomputed, so a facet switch costs an interpolation and nothing else.
  const posRef = useRef<Float32Array>(new Float32Array(0));
  const fromRef = useRef<Float32Array>(new Float32Array(0));
  const toRef = useRef<Float32Array>(new Float32Array(0));
  const animRef = useRef<{ start: number } | null>(null);
  // When the region fields may start appearing. Null while the works are
  // still travelling: a field is a claim about where a group of works is,
  // and drawing it over the empty ground they are heading towards makes
  // the claim before it is true. Zero at the start, so the first layout
  // is simply there.
  const fieldFadeRef = useRef<number | null>(0);
  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const hiWantedRef = useRef<Set<number>>(new Set());
  // The ground tiles, decoded once and kept. A pattern is rebuilt only
  // when the facet or the pixel ratio changes, not per frame.
  const groundRef = useRef<Partial<Record<Facet, HTMLImageElement>>>({});
  // The background painted out at viewport size, ground and all.
  const bakedRef = useRef<{ facet: Facet; w: number; h: number; dpr: number;
                            canvas: HTMLCanvasElement } | null>(null);
  // Which work wears the highlight. A ref rather than a dependency of the
  // render effect: selecting is the most common thing anyone does here,
  // and it changes one rectangle, not the loop that draws the map.
  const selectedRef = useRef<number | null>(selected);
  selectedRef.current = selected;
  const hoverRef = useRef<string | null>(null);
  // Screen rectangle of the uploaded picture, written each frame so a
  // click can be tested against it before the nearest-work search.
  // One per uploaded picture, written each frame, newest last — so a
  // click tested from the end finds whichever is drawn on top.
  const pinRectRef = useRef<
    { x: number; y: number; w: number; h: number; which: number }[]>([]);
  // Likewise the selected work, which is held at a legible size however
  // far out you are, so it stays as clickable as it is visible.
  const selRectRef = useRef<
    { x: number; y: number; w: number; h: number } | null>(null);
  const flyRef = useRef<
    { from: View; to: View; start: number; ms: number } | null>(null);

  const { works, layouts, atlasMeta, sheets, mips, slots, facets, hires }
    = bundle;
  // Per facet: a left-to-right timeline is a ribbon an eighth the area
  // of the round layouts, so it carries a smaller tile.
  const radius = layouts.facets[facet].work_radius ?? layouts.work_radius;
  const images = useMemo(() => new ImageCache(facets.iiif), [facets.iiif]);
  const hiAtlas = useMemo(
    () => (hires ? new HiResAtlas(import.meta.env.BASE_URL, hires.meta) : null),
    [hires],
  );

  const flatten = useCallback((f: Facet) => {
    const xy = layouts.facets[f].xy;
    const out = new Float32Array(xy.length * 2);
    for (let i = 0; i < xy.length; i++) {
      out[i * 2] = xy[i][0];
      out[i * 2 + 1] = xy[i][1];
    }
    return out;
  }, [layouts]);

  // Built for the facet's settled positions. While a facet change is
  // animating the works are somewhere between two layouts, so the frame
  // falls back to the full scan for those few hundred milliseconds
  // rather than querying an index that no longer describes them.
  // All four grounds up front. They are a few tens of kilobytes between
  // them and switching groupings should not wait on one.
  useEffect(() => {
    let live = true;
    for (const [name, src] of Object.entries(GROUND)) {
      const img = new Image();
      img.onload = () => {
        if (live) groundRef.current[name as Facet] = img;
      };
      img.src = src;
    }
    return () => { live = false; };
  }, []);

  const gridIndex = useMemo(
    () => buildGrid(flatten(facet)), [flatten, facet]);

  // Baked once per facet. Hovering only changes how heavily a field is
  // drawn, not the field, so it costs nothing here.
  const fields = useMemo(
    () => (TINTED_FACETS.has(facet)
      ? buildFields(layouts.facets[facet].regions) : []),
    [layouts, facet],
  );

  // Where the whole of *this* facet sits. Fitting to a fixed unit square
  // put the timeline near the top of the screen and small: it is a ribbon
  // about a tenth as tall as it is wide, so most of that square is empty.
  // Framing the works' own bounds centres whatever the facet actually
  // draws, and fills the viewport with it.
  // The shorter side of the visible strip, which is what a fitted view
  // has to fit inside.
  // Read through a ref, so these two keep the same identity when the
  // panel opens or closes. They are dependencies of the fit and of the
  // effect that reacts to a grouping change; if they changed with the
  // panel, that effect would fire on every open and close and fly the
  // map home. The value used is whatever the inset is at the moment
  // something actually asks to be framed.
  const insetRef = useRef(rightInset);
  insetRef.current = rightInset;
  // A ref, so pausing does not tear down and rebuild the render loop.
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const seen = useCallback(
    (rect: DOMRect) => Math.min(rect.width - insetRef.current, rect.height),
    []);

  // How far to aim left of a target so it lands in the middle of what the
  // panel leaves visible. Applied to the destination rather than to the
  // drawn centre: moving the centre re-centred the map the instant a
  // panel opened or closed, which is a jump nobody asked for. Applied
  // here it takes effect only when something is already travelling.
  const aimOff = useCallback(
    (scale: number) => (scale > 0 ? insetRef.current / (2 * scale) : 0),
    []);

  const homeView = useCallback((rect: DOMRect) => {
    const g = gridIndex;
    if (!g) {
      const flat = seen(rect) * 0.92;
      return { x: 0.5 + aimOff(flat), y: 0.5, scale: flat };
    }
    // Read off the index, which measured these bounds once when it was
    // built. This used to flatten the layout and scan all 25,515 works
    // on the spot — and one of its callers is the wheel handler, which
    // fires faster than frames do, so zooming rebuilt a 51,030-element
    // array and swept it several times per frame to arrive at a number
    // that cannot change until the facet does.
    const x0 = g.x0, x1 = g.x0 + g.cw * g.nx;
    const y0 = g.y0, y1 = g.y0 + g.ch * g.ny;
    // The bounds are tile centres, so leave a tile's width of air around
    // them or the outermost row is cut in half by the edge.
    const pad = (layouts.facets[facet].work_radius ?? layouts.work_radius) * 2;
    const spanX = x1 - x0 + pad * 2;
    const spanY = y1 - y0 + pad * 2;
    if (!(spanX > 0) || !(spanY > 0)) {
      const flat = seen(rect) * 0.92;
      return { x: 0.5 + aimOff(flat), y: 0.5, scale: flat };
    }
    const scale = Math.min((rect.width - insetRef.current) / spanX,
                           rect.height / spanY) * 0.92;
    return {
      x: (x0 + x1) / 2 + aimOff(scale),
      y: (y0 + y1) / 2,
      scale,
    };
  }, [layouts, facet, gridIndex, seen, aimOff]);

  useEffect(() => {
    const target = flatten(facet);
    if (posRef.current.length === 0) {
      posRef.current = target.slice();
      setReady(true);
      return;
    }
    fromRef.current = posRef.current.slice();
    toRef.current = target;
    animRef.current = { start: performance.now() };
    fieldFadeRef.current = null;

    // Pull back to the whole map as the regions re-form. Every facet
    // rearranges the entire corpus, so whatever you were looking at is
    // somewhere else afterwards — staying zoomed in just drops you in an
    // unfamiliar part of a map that has changed under you.
    const canvas = canvasRef.current;
    if (canvas) {
      flyRef.current = {
        from: { ...viewRef.current },
        to: homeView(canvas.getBoundingClientRect()),
        start: performance.now(),
        ms: ANIM_MS,
      };
    }
  }, [facet, flatten, homeView]);

  // Fit the map to the viewport once, and again whenever it resizes.
  const fit = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    Object.assign(viewRef.current, homeView(canvas.getBoundingClientRect()));
  }, [homeView]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let frame = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(rect.width * dpr);
      const h = Math.round(rect.height * dpr);
      // Only when it really changed. Assigning either of these wipes the
      // canvas even if the value is identical, and this runs on every
      // re-entry into the effect, so an unguarded write showed up as a
      // black frame on something as ordinary as a click.
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      if (viewRef.current.scale === 0) fit();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = () => {
      frame = requestAnimationFrame(draw);
      if (pausedRef.current) return;
      const ctx = canvas.getContext("2d");
      if (!ctx || posRef.current.length === 0) return;

      // Flying to a region: interpolate the scale in log space, so the
      // journey feels like a steady zoom rather than a slow crawl that
      // suddenly accelerates.
      const fly = flyRef.current;
      if (fly) {
        const p = Math.min(1, (performance.now() - fly.start) / fly.ms);
        const e = easeInOut(p);
        const v = viewRef.current;
        v.x = fly.from.x + (fly.to.x - fly.from.x) * e;
        v.y = fly.from.y + (fly.to.y - fly.from.y) * e;
        v.scale = Math.exp(
          Math.log(fly.from.scale)
          + (Math.log(fly.to.scale) - Math.log(fly.from.scale)) * e,
        );
        if (p >= 1) flyRef.current = null;
      }

      const anim = animRef.current;
      if (anim) {
        const t = Math.min(1, (performance.now() - anim.start) / ANIM_MS);
        const e = easeInOut(t);
        const from = fromRef.current, to = toRef.current, pos = posRef.current;
        for (let i = 0; i < pos.length; i++) {
          pos[i] = from[i] + (to[i] - from[i]) * e;
        }
        if (t >= 1) {
          animRef.current = null;
          fieldFadeRef.current = performance.now();
        }
      }

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.width, h = canvas.height;
      const { x: vx, y: vy, scale } = viewRef.current;
      const s = scale * dpr;
      const cx = w / 2, cy = h / 2;

      // The background, with the grouping's ground already in it.
      //
      // Painted once into an image the size of the viewport and blitted
      // after that. It was a full-screen pattern fill every frame, with
      // a transform on the pattern, which is the arrangement least
      // likely to stay on a fast path — and it cost that on every frame
      // to produce pixels that never change. It is repainted only when
      // the grouping, the viewport or the pixel ratio changes.
      const ground = groundRef.current[facet];
      const baked = bakedRef.current;
      if (ground && ground.complete && ground.naturalWidth
          && (!baked || baked.facet !== facet || baked.w !== w
              || baked.h !== h || baked.dpr !== dpr)) {
        const off = document.createElement("canvas");
        off.width = w;
        off.height = h;
        const oc = off.getContext("2d");
        if (oc) {
          oc.fillStyle = "#0d0d0f";
          oc.fillRect(0, 0, w, h);
          const pattern = oc.createPattern(ground, "repeat");
          if (pattern) {
            // Scaled by the pixel ratio, or the tile comes out half size
            // on a dense screen and twice as busy as it was drawn.
            pattern.setTransform(new DOMMatrix().scale(dpr * GROUND_SCALE));
            oc.globalAlpha = GROUND_ALPHA;
            oc.fillStyle = pattern;
            oc.fillRect(0, 0, w, h);
          }
          bakedRef.current = { facet, w, h, dpr, canvas: off };
        }
      }
      const ready = bakedRef.current;
      if (ready && ready.facet === facet && ready.w === w && ready.h === h) {
        ctx.drawImage(ready.canvas, 0, 0);
      } else {
        ctx.fillStyle = "#0d0d0f";
        ctx.fillRect(0, 0, w, h);
      }

      // Century rules, behind everything. Recessive: they are a scale to
      // read against, not a thing to look at.
      const rules = layouts.facets[facet].grid;
      if (rules && rules.length) {
        // Full height of the viewport, not of the map. The ribbon is a
        // tenth as tall as it is wide, so ruling map-space 0..1 put almost
        // all of every line below the works instead of through them.
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        for (const line of rules) {
          const gx = (line.x - vx) * s + cx;
          if (gx < -80 || gx > w + 80) continue;
          ctx.strokeStyle = "rgba(255,255,255,0.09)";
          ctx.lineWidth = 1 * dpr;
          ctx.beginPath();
          ctx.moveTo(gx, 0);
          ctx.lineTo(gx, h);
          ctx.stroke();

          const font = Math.min(20, Math.max(11, 12 * (s / 900))) * dpr;
          ctx.font = `500 ${font}px ui-sans-serif, system-ui, sans-serif`;
          ctx.fillStyle = "rgba(255,255,255,0.4)";
          ctx.fillText(line.label, gx, 10 * dpr);
        }
      }

      // Region fields, under the tiles so the art is never tinted.
      // One blit each, from a texture baked when the facet was chosen.
      // Filling and blurring these per frame cost in proportion to the
      // area they covered, which is why the tinted facets dragged and the
      // timeline did not.
      const fadeAt = fieldFadeRef.current;
      const fieldFade = fadeAt === null
        ? 0 : Math.min(1, (performance.now() - fadeAt) / FIELD_FADE_MS);
      if (fields.length && fieldFade > 0) {
        // Set explicitly: the tile pass below turns smoothing off once
        // tiles are large, and a field scaled up without it is a staircase.
        ctx.imageSmoothingEnabled = true;
        for (const f of fields) {
          const fx = (f.x0 - vx) * s + cx;
          const fy = (f.y0 - vy) * s + cy;
          const fw = f.w * s, fh = f.h * s;
          if (fx + fw < 0 || fy + fh < 0 || fx > w || fy > h) continue;
          ctx.globalAlpha =
            (hoverRef.current === f.name ? 0.42 : 0.24) * fieldFade;
          ctx.drawImage(f.tex, fx, fy, fw, fh);
        }
        ctx.globalAlpha = 1;
      }

      const pos = posRef.current;
      const size = Math.max(1, radius * 2 * s);
      if (size >= THUMB_MIN_PX) onNeedAtlas();
      const useThumbs = size >= THUMB_MIN_PX && sheets.length > 0;
      // Below this a tile does not use a 64px cell, it only pays to read
      // one. The reduction is exact at MIP_TILE and a clean downscale
      // under it; above it the full sheet is cheap anyway, because the
      // tiles are large enough that few are on screen.
      const useMip = useThumbs && size <= MIP_TILE && mips.length > 0;
      const srcTile = useMip ? MIP_TILE : atlasMeta.tile;
      const { tile, grid, per_sheet } = atlasMeta;
      // Cull generously: one tile of slop stops edges popping in.
      const pad = size + 4;
      // Images are fetched for a wider band than is drawn, so panning
      // moves into work that is already loaded instead of starting from
      // the sprite every time. Half a screen in each direction.
      const prefetch = Math.max(w, h) * 0.5;

      ctx.imageSmoothingEnabled = size < tile;

      // Past this size an atlas sprite is visibly blocky, so ask IIIF for
      // the real thing. Only visible works are ever requested.
      // Derive the switch point from the atlas actually loaded rather than
      // a constant, so re-running D5b at a different tile size needs no
      // client change.
      // Tier order by cost: the 64px atlas is already in memory, the
      // high-resolution atlas is a sheet fetch that serves ~100 works at
      // once, and IIIF is one request per work. Use the cheapest that is
      // sharp enough.
      const hiTile = hires?.meta.tile ?? 0;
      const wantHi = hires !== null && size >= tile * 0.9;
      const wantReal = size >= Math.max(TIER2_MIN_PX, hiTile * 0.9 || tile * 0.9);
      // Reused across frames rather than allocated each one.
      const hiWanted = hiWantedRef.current;
      hiWanted.clear();
      // Collected during the draw and requested after it, nearest to the
      // centre of the screen first, so what you are looking at arrives
      // before what is at the edge.
      const missing: { id: string; d2: number }[] = [];

      const visit = (i: number) => {
        const px = (pos[i * 2] - vx) * s + cx;
        const py = (pos[i * 2 + 1] - vy) * s + cy;
        const onScreen =
          px >= -pad && py >= -pad && px <= w + pad && py <= h + pad;
        if (!onScreen) {
          if (!wantReal) return;
          if (px < -prefetch || py < -prefetch ||
              px > w + prefetch || py > h + prefetch) return;
          if (images.wants(works[i].img, size)) {
            const ox = px - cx, oy = py - cy;
            // Ranked behind everything on screen, so visible tiles never
            // wait on a neighbour you have not reached yet.
            missing.push({ id: works[i].img, d2: ox * ox + oy * oy + 1e9 });
          }
          return;
        }

        // Every tier draws the work's true shape. `size` is the box it has
        // to fit inside, not the shape it takes: a 1:3 hanging scroll is a
        // sliver, a wide landscape is a band.
        const ar = works[i].ar ?? 1;
        const tw = ar >= 1 ? size : size * ar;
        const th = ar >= 1 ? size / ar : size;
        const hw = tw / 2, hh = th / 2;

        if (wantReal) {
          const real = images.peek(works[i].img, size);
          if (real) {
            ctx.drawImage(real, px - hw, py - hh, tw, th);
            return;
          }
          if (images.wants(works[i].img, size)) {
            const ox = px - cx, oy = py - cy;
            missing.push({ id: works[i].img, d2: ox * ox + oy * oy });
          }
        }

        if (wantHi && hires && hiAtlas) {
          const slot = hires.slots[i];
          hiWanted.add(hiAtlas.sheetOf(slot));
          const sheet = hiAtlas.get(slot);
          if (sheet) {
            const g = hires.meta.grid, ht = hires.meta.tile;
            const within = slot % hires.meta.per_sheet;
            const iw = ar >= 1 ? ht : ht * ar;
            const ih = ar >= 1 ? ht / ar : ht;
            ctx.drawImage(
              sheet,
              (within % g) * ht + (ht - iw) / 2,
              ((within / g) | 0) * ht + (ht - ih) / 2,
              iw, ih,
              px - hw, py - hh, tw, th,
            );
            return;
          }
        }

        // Its slot, not its index: the sheets are packed along a curve
        // through the map, so neighbouring works share a sheet.
        const slot = slots[i];
        const sheetIndex = (slot / per_sheet) | 0;
        const sheet = useThumbs
          ? (useMip ? mips[sheetIndex] : sheets[sheetIndex]) : undefined;
        if (sheet) {
          // The atlas cell is square with the work letterboxed inside it,
          // so read back only the occupied part. Drawing the whole cell
          // would paint its padding over the neighbouring tiles.
          const within = slot % per_sheet;
          const iw = ar >= 1 ? srcTile : srcTile * ar;
          const ih = ar >= 1 ? srcTile / ar : srcTile;
          ctx.drawImage(
            sheet,
            (within % grid) * srcTile + (srcTile - iw) / 2,
            ((within / grid) | 0) * srcTile + (srcTile - ih) / 2,
            iw, ih,
            px - hw, py - hh, tw, th,
          );
        } else {
          ctx.fillStyle = works[i].k ?? "#3a3a3f";
          ctx.fillRect(px - hw, py - hh, tw, th);
        }
      
      };

      // Only the works that could land in the frame. This walked all
      // 25,515 every frame and culled inside the loop, so zooming in cost
      // exactly as much as zooming out. While a facet change animates,
      // positions are between two layouts and the index does not describe
      // them, so those frames still scan.
      const q = wantReal ? prefetch : pad;
      const qx0 = (-q - cx) / s + vx, qx1 = (w + q - cx) / s + vx;
      const qy0 = (-q - cy) / s + vy, qy1 = (h + q - cy) / s + vy;
      if (gridIndex && !animRef.current) {
        const g = gridIndex;
        const lo = (v: number, o: number, c: number, n: number) =>
          Math.min(n - 1, Math.max(0, Math.floor((v - o) / c)));
        const gx0 = lo(qx0, g.x0, g.cw, g.nx), gx1 = lo(qx1, g.x0, g.cw, g.nx);
        const gy0 = lo(qy0, g.y0, g.ch, g.ny), gy1 = lo(qy1, g.y0, g.ch, g.ny);
        for (let gy = gy0; gy <= gy1; gy++) {
          const row = gy * g.nx;
          for (let c = row + gx0; c <= row + gx1; c++) {
            for (let k = g.start[c]; k < g.start[c + 1]; k++) visit(g.items[k]);
          }
        }
      } else {
        for (let i = 0; i < works.length; i++) visit(i);
      }

      // Nearest first. Capped per frame so a fast pan does not enqueue a
      // whole screen of work that has scrolled away by the time it lands.
      if (missing.length) {
        missing.sort((a, b) => a.d2 - b.d2);
        for (let n = 0; n < Math.min(missing.length, 32); n++) {
          images.fetch(missing[n].id, size);
        }
      }

      if (hiAtlas && hiWanted.size) hiAtlas.pump(hiWanted);

      // The uploaded picture, sitting among the works it resembles. Its
      // position is read from the live animated coordinates rather than
      // the target layout, so when the grouping changes it travels with
      // its neighbours instead of jumping when they arrive.
      pinRectRef.current.length = 0;
      for (let which = 0; which < pins.length; which++) {
        const p = pins[which];
        if (!p.matches.length) continue;
        // Same rule the panel uses: the region most of the neighbours
        // agree on, averaged over only the ones in it. Read from the live
        // animated coordinates, so it travels with them.
        const regionOf = layouts.facets[facet].region_of;
        const at = placeAmong(
          p.matches,
          (i) => (i * 2 + 1 < pos.length
            ? [pos[i * 2], pos[i * 2 + 1]] as [number, number] : null),
          regionOf && regionOf.length ? regionOf : null,
          // Only where the regions are styles. Under place or period the
          // picture has no claim of its own to make, so its neighbours
          // speak for it.
          facet === "style" ? p.style : null,
        );
        if (!at) continue;
        const x = (at[0] - vx) * s + cx;
        const y = (at[1] - vy) * s + cy;
        const im = p.img;
        // Marked the way a selected work is marked, and sized the way
        // one is: it belongs among its neighbours rather than looming
        // over them.
        const box = Math.max(size, SELECTED_MIN_PX * dpr);
        const ar2 = im.naturalWidth / Math.max(im.naturalHeight, 1) || 1;
        const pw = ar2 >= 1 ? box : box * ar2;
        const ph = ar2 >= 1 ? box / ar2 : box;

        pinRectRef.current.push({ x, y, w: pw, h: ph, which });
        // Only the one whose panel is open. Two halos at once says two
        // things are chosen, and nothing says which panel that is.
        if (activePin === which) glow(ctx, x, y, pw, ph, dpr);
        if (im.complete && im.naturalWidth) {
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(im, x - pw / 2, y - ph / 2, pw, ph);
        }

        ctx.font = `600 ${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.lineWidth = 4 * dpr;
        ctx.strokeStyle = "rgba(6,6,8,0.9)";
        const tag = pins.length > 1 ? `yours ${which + 1}` : "yours";
        ctx.strokeText(tag, x, y - ph / 2 - 8 * dpr);
        ctx.fillStyle = "#f5c451";
        ctx.fillText(tag, x, y - ph / 2 - 8 * dpr);
      }

      const highlight = selectedRef.current;
      selRectRef.current = null;
      if (highlight !== null) {
        const px = (pos[highlight * 2] - vx) * s + cx;
        const py = (pos[highlight * 2 + 1] - vy) * s + cy;
        // Its own size, or a floor — whichever is larger. Zoomed in it is
        // simply the tile the draw above already placed; zoomed out it is
        // held at the floor so it does not shrink into the field.
        const box = Math.max(size, SELECTED_MIN_PX * dpr);
        const ar = works[highlight].ar ?? 1;
        const bw = ar >= 1 ? box : box * ar;
        const bh = ar >= 1 ? box / ar : box;
        selRectRef.current = { x: px, y: py, w: bw, h: bh };

        glow(ctx, px, py, bw, bh, dpr);

        // Zoomed out the tile underneath is a speck, so draw the work
        // again at the floor size. Close in the tile is already there at
        // the best resolution loaded, and redrawing it from the atlas
        // would only make it blurrier.
        if (box > size) {
          ctx.imageSmoothingEnabled = true;
          const hslot = slots[highlight];
          const sh = sheets[(hslot / per_sheet) | 0];
          if (sh) {
            const within = hslot % per_sheet;
            const iw = ar >= 1 ? tile : tile * ar;
            const ih = ar >= 1 ? tile / ar : tile;
            ctx.drawImage(
              sh,
              (within % grid) * tile + (tile - iw) / 2,
              ((within / grid) | 0) * tile + (tile - ih) / 2,
              iw, ih,
              px - bw / 2, py - bh / 2, bw, bh,
            );
          } else {
            ctx.fillStyle = works[highlight].k ?? "#3a3a3f";
            ctx.fillRect(px - bw / 2, py - bh / 2, bw, bh);
          }
        }
      }

      // Region names, except where the rules already name the axis — on
      // the timeline they said the same thing twice and landed on top of
      // the works while doing it.
      const labelFade = Math.min(1, Math.max(0,
        (LABEL_MAX_SCALE - scale) / (LABEL_MAX_SCALE - LABEL_FADE_FROM)));
      if (labelFade > 0 && !(rules && rules.length)) {
        const regions = layouts.facets[facet].regions;
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        // A shadow rather than an outline. The shadow of filled text is a
        // blurred copy behind it, so the name keeps one colour and gains
        // no edge — it darkens what is under the letters just enough to
        // hold them. Set once for the whole pass: shadow state is sticky
        // on a context and would otherwise carry into the next frame.
        ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        ctx.shadowBlur = 7 * dpr;
        // Takes the shadow down with the letters, so the name thins out
        // whole rather than leaving its own shadow behind.
        ctx.globalAlpha = labelFade;
        for (const region of regions) {
          const px = (region.c[0] - vx) * s + cx;
          const py = (region.c[1] - vy) * s + cy;
          if (px < 0 || py < 0 || px > w || py > h) continue;
          const font = Math.min(34, Math.max(11, region.r * s * 0.24)) * dpr;
          if (font < 9 * dpr) continue;
          ctx.font = `600 ${font}px ui-sans-serif, system-ui, sans-serif`;
          // Drawn twice, so the shadow lands twice and actually reads.
          // The fill is dropped to 0.68 to compensate: two passes at that
          // composite to 1 - 0.32^2, which is the 0.9 the name should be.
          ctx.fillStyle = "rgba(255,255,255,0.68)";
          ctx.fillText(region.name, px, py);
          ctx.fillText(region.name, px, py);
        }
        ctx.restore();
      }
    };
    frame = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(frame); ro.disconnect(); };
  }, [works, layouts, facet, atlasMeta, sheets, mips, radius, fit, ready,
      images, hires, hiAtlas, focus, pins, activePin, slots, onNeedAtlas]);

  useEffect(() => {
    if (!focus) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    // Fit the region, with a little air around it.
    const target =
      seen(rect) / (focus.r * FOCUS_FILL);
    flyRef.current = {
      from: { ...viewRef.current },
      to: { x: focus.x + aimOff(target), y: focus.y, scale: target },
      start: performance.now(),
      ms: FLY_MS,
    };
  }, [focus, seen, aimOff]);

  // --- interaction -------------------------------------------------------
  const toWorld = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const { x, y, scale } = viewRef.current;
    return {
      x: (clientX - rect.left - rect.width / 2) / scale + x,
      y: (clientY - rect.top - rect.height / 2) / scale + y,
    };
  };

  // Wheel is bound natively rather than through React, because React
  // attaches wheel listeners passively and a passive listener cannot call
  // preventDefault. Without that call, ctrl+scroll and trackpad pinch fall
  // through to the browser and resize the whole page — which fights the
  // map's own zoom on the same gesture.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      flyRef.current = null;
      const rect = canvas.getBoundingClientRect();
      const view = viewRef.current;
      const at = (cx: number, cy: number) => ({
        x: (cx - rect.left - rect.width / 2) / view.scale + view.x,
        y: (cy - rect.top - rect.height / 2) / view.scale + view.y,
      });
      const before = at(e.clientX, e.clientY);
      // A pinch on a trackpad arrives as ctrl+wheel with small deltas, so
      // it needs a stronger factor to feel like the same gesture.
      const k = e.ctrlKey ? 0.012 : 0.0016;
      const next = view.scale * Math.exp(-e.deltaY * k);
      // The fitted view is the floor. Zooming out past the point where the
      // whole facet is on screen only shrinks it into the middle of an
      // empty canvas, and for the timeline — a ribbon a tenth as tall as
      // it is wide — that happened almost immediately.
      const floor = homeView(rect).scale;
      view.scale = Math.min(Math.max(next, floor), 260000);
      const after = at(e.clientX, e.clientY);
      view.x += before.x - after.x;
      view.y += before.y - after.y;
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [homeView]);

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    flyRef.current = null;      // any interaction cancels a flight
    dragRef.current = { x: e.clientX, y: e.clientY, moved: false };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) {
      // Smallest region under the cursor wins, so a small one nested
      // against a large one is still reachable.
      const canvas = canvasRef.current;
      if (!canvas) return;
      const box = canvas.getBoundingClientRect();
      const view = viewRef.current;
      const wx = (e.clientX - box.left - box.width / 2) / view.scale + view.x;
      const wy = (e.clientY - box.top - box.height / 2) / view.scale + view.y;
      let best: string | null = null, bestR = Infinity;
      for (const region of layouts.facets[facet].regions) {
        const within = region.o
          ? inside(region.o, wx, wy)
          : Math.hypot(wx - region.c[0], wy - region.c[1]) <= region.r;
        if (within && region.r < bestR) { bestR = region.r; best = region.name; }
      }
      if (best !== hoverRef.current) {
        hoverRef.current = best;
        canvas.style.cursor = best ? "pointer" : "grab";
      }
      return;
    }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    const view = viewRef.current;
    view.x -= dx / view.scale;
    view.y -= dy / view.scale;
    drag.x = e.clientX;
    drag.y = e.clientY;
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag || drag.moved) return;

    // An uploaded picture is drawn over the map and is larger than a
    // tile, so it gets first refusal on a click. Tested newest first,
    // which is the one drawn on top where two overlap.
    {
      const canvas = canvasRef.current!;
      const box = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const px = (e.clientX - box.left) * dpr;
      const py = (e.clientY - box.top) * dpr;
      const rects = pinRectRef.current;
      for (let n = rects.length - 1; n >= 0; n--) {
        const rect = rects[n];
        if (Math.abs(px - rect.x) <= rect.w / 2
            && Math.abs(py - rect.y) <= rect.h / 2) {
          onOpenPin(rect.which);
          return;
        }
      }
    }

    // The selected work is held at a legible size when the tiles around
    // it are specks, so it has to be hit-tested at that size too —
    // otherwise it is drawn where a click cannot reach it, and zoomed out
    // the click would go to the region instead.
    const sel = selRectRef.current;
    if (sel && selected !== null) {
      const canvas = canvasRef.current!;
      const box = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const px = (e.clientX - box.left) * dpr;
      const py = (e.clientY - box.top) * dpr;
      if (Math.abs(px - sel.x) <= sel.w / 2
          && Math.abs(py - sel.y) <= sel.h / 2) {
        onSelect(selected);
        return;
      }
    }

    const world = toWorld(e.clientX, e.clientY);

    // Smallest region containing the point, if any.
    let pick: { c: [number, number]; r: number } | null = null;
    for (const region of REGION_CLICK_FACETS.has(facet)
                         ? layouts.facets[facet].regions : []) {
      const within = region.o
        ? inside(region.o, world.x, world.y)
        : Math.hypot(world.x - region.c[0], world.y - region.c[1]) <= region.r;
      if (within && (!pick || region.r < pick.r)) {
        pick = { c: region.c, r: region.r };
      }
    }

    // Zoomed out far enough that tiles are specks: aiming at one is not a
    // gesture anyone can perform, so a click inside a region goes to the
    // region. Close in, picking a work is exactly what a click means.
    //
    // Unless you are already there. A large region fitted to the screen
    // still has tiles under that threshold, so going by tile size alone
    // meant every further click flew you to the region you were looking
    // at and the works inside it could never be reached. A click only
    // travels while there is somewhere to travel to.
    const tilePx = radius * 2 * viewRef.current.scale;
    const arrived = (at: { r: number }) => {
      const canvas = canvasRef.current;
      if (!canvas) return false;
      const box = canvas.getBoundingClientRect();
      const fitted = Math.min(box.width, box.height)
        / (Math.max(at.r, MIN_FOCUS_R) * FOCUS_FILL);
      return viewRef.current.scale >= fitted * 0.9;
    };
    if (pick && tilePx < WORK_CLICK_MIN_PX && !arrived(pick)) {
      onRegion(pick.c[0], pick.c[1], Math.max(pick.r, MIN_FOCUS_R));
      return;
    }

    const pos = posRef.current;
    // Nearest work within a few tile-widths, so a click near a gap does
    // nothing rather than selecting something across the map.
    const reach = Math.max(radius * 3, 40 / viewRef.current.scale);
    let best = -1, bestDist = reach * reach;
    for (let i = 0; i < works.length; i++) {
      const dx = pos[i * 2] - world.x, dy = pos[i * 2 + 1] - world.y;
      const d = dx * dx + dy * dy;
      if (d < bestDist) { bestDist = d; best = i; }
    }
    if (best >= 0) {
      onSelect(best);
      return;
    }

    // Empty space inside a region: go to the region instead of
    // deselecting — again, only if that is somewhere else.
    if (pick && !arrived(pick)) {
      onRegion(pick.c[0], pick.c[1], Math.max(pick.r, MIN_FOCUS_R));
      return;
    }
    onSelect(null);
  };

  return (
    <canvas
      ref={canvasRef}
      className="map"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => { dragRef.current = null; }}
    />
  );
}
