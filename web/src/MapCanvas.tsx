import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Bundle } from "./useData";
import type { Facet } from "./types";
import { ImageCache, TIER2_MIN_PX } from "./imageCache";
import { HiResAtlas } from "./hiresAtlas";

interface View { x: number; y: number; scale: number }

// Below this many pixels per tile there is no point paying for a texture
// lookup — a coloured square and a thumbnail are indistinguishable.
const THUMB_MIN_PX = 3.5;
const LABEL_MAX_SCALE = 4200;
const ANIM_MS = 750;
const FLY_MS = 900;

const easeInOut = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

export interface Focus { x: number; y: number; r: number; key: number }

export function MapCanvas({
  bundle, facet, selected, onSelect, focus,
}: {
  bundle: Bundle;
  facet: Facet;
  selected: number | null;
  onSelect: (index: number | null) => void;
  focus: Focus | null;
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
  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const flyRef = useRef<
    { from: View; to: View; start: number; ms: number } | null>(null);

  const { works, layouts, atlasMeta, sheets, facets, hires } = bundle;
  const radius = layouts.work_radius;
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

    // Pull back to the whole map as the regions re-form. Every facet
    // rearranges the entire corpus, so whatever you were looking at is
    // somewhere else afterwards — staying zoomed in just drops you in an
    // unfamiliar part of a map that has changed under you.
    const canvas = canvasRef.current;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      flyRef.current = {
        from: { ...viewRef.current },
        to: { x: 0.5, y: 0.5, scale: Math.min(rect.width, rect.height) * 0.92 },
        start: performance.now(),
        ms: ANIM_MS,
      };
    }
  }, [facet, flatten]);

  // Fit the map to the viewport once, and again whenever it resizes.
  const fit = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    viewRef.current.scale = Math.min(rect.width, rect.height) * 0.92;
    viewRef.current.x = 0.5;
    viewRef.current.y = 0.5;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let frame = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      if (viewRef.current.scale === 0) fit();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = () => {
      frame = requestAnimationFrame(draw);
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
        if (t >= 1) animRef.current = null;
      }

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.width, h = canvas.height;
      const { x: vx, y: vy, scale } = viewRef.current;
      const s = scale * dpr;
      const cx = w / 2, cy = h / 2;

      ctx.fillStyle = "#0d0d0f";
      ctx.fillRect(0, 0, w, h);

      const pos = posRef.current;
      const size = Math.max(1, radius * 2 * s);
      const useThumbs = size >= THUMB_MIN_PX && sheets.length > 0;
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
      const hiWanted = new Set<number>();
      // Collected during the draw and requested after it, nearest to the
      // centre of the screen first, so what you are looking at arrives
      // before what is at the edge.
      const missing: { id: string; d2: number }[] = [];

      for (let i = 0; i < works.length; i++) {
        const px = (pos[i * 2] - vx) * s + cx;
        const py = (pos[i * 2 + 1] - vy) * s + cy;
        const onScreen =
          px >= -pad && py >= -pad && px <= w + pad && py <= h + pad;
        if (!onScreen) {
          if (!wantReal) continue;
          if (px < -prefetch || py < -prefetch ||
              px > w + prefetch || py > h + prefetch) continue;
          if (images.wants(works[i].img, size)) {
            const ox = px - cx, oy = py - cy;
            // Ranked behind everything on screen, so visible tiles never
            // wait on a neighbour you have not reached yet.
            missing.push({ id: works[i].img, d2: ox * ox + oy * oy + 1e9 });
          }
          continue;
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
            continue;
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
            continue;
          }
        }

        const sheetIndex = (i / per_sheet) | 0;
        const sheet = useThumbs ? sheets[sheetIndex] : undefined;
        if (sheet) {
          // The atlas cell is square with the work letterboxed inside it,
          // so read back only the occupied part. Drawing the whole cell
          // would paint its padding over the neighbouring tiles.
          const within = i % per_sheet;
          const iw = ar >= 1 ? tile : tile * ar;
          const ih = ar >= 1 ? tile / ar : tile;
          ctx.drawImage(
            sheet,
            (within % grid) * tile + (tile - iw) / 2,
            ((within / grid) | 0) * tile + (tile - ih) / 2,
            iw, ih,
            px - hw, py - hh, tw, th,
          );
        } else {
          ctx.fillStyle = works[i].k ?? "#3a3a3f";
          ctx.fillRect(px - hw, py - hh, tw, th);
        }
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

      if (selected !== null) {
        const px = (pos[selected * 2] - vx) * s + cx;
        const py = (pos[selected * 2 + 1] - vy) * s + cy;
        const r = Math.max(size * 0.95, 9 * dpr);
        ctx.strokeStyle = "#f5c451";
        ctx.lineWidth = 2 * dpr;
        ctx.strokeRect(px - r, py - r, r * 2, r * 2);
      }

      if (scale < LABEL_MAX_SCALE) {
        const regions = layouts.facets[facet].regions;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        for (const region of regions) {
          const px = (region.c[0] - vx) * s + cx;
          const py = (region.c[1] - vy) * s + cy;
          if (px < 0 || py < 0 || px > w || py > h) continue;
          const font = Math.min(34, Math.max(11, region.r * s * 0.24)) * dpr;
          if (font < 9 * dpr) continue;
          ctx.font = `600 ${font}px ui-sans-serif, system-ui, sans-serif`;
          ctx.lineWidth = 4 * dpr;
          ctx.strokeStyle = "rgba(6,6,8,0.85)";
          ctx.strokeText(region.name, px, py);
          ctx.fillStyle = "rgba(255,255,255,0.92)";
          ctx.fillText(region.name, px, py);
        }
      }
    };
    frame = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(frame); ro.disconnect(); };
  }, [works, layouts, facet, atlasMeta, sheets, radius, selected, fit, ready,
      images, hires, hiAtlas, focus]);

  useEffect(() => {
    if (!focus) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    // Fit the region, with a little air around it.
    const target = Math.min(rect.width, rect.height) / (focus.r * 2.6);
    flyRef.current = {
      from: { ...viewRef.current },
      to: { x: focus.x, y: focus.y, scale: target },
      start: performance.now(),
      ms: FLY_MS,
    };
  }, [focus]);

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
      view.scale = Math.min(
        Math.max(next, Math.min(rect.width, rect.height) * 0.5), 260000,
      );
      const after = at(e.clientX, e.clientY);
      view.x += before.x - after.x;
      view.y += before.y - after.y;
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    flyRef.current = null;      // any interaction cancels a flight
    dragRef.current = { x: e.clientX, y: e.clientY, moved: false };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
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
    const world = toWorld(e.clientX, e.clientY);
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
    onSelect(best >= 0 ? best : null);
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
