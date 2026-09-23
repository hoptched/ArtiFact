import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Bundle } from "./useData";
import type { Facet } from "./types";
import { ImageCache, TIER2_MIN_PX } from "./imageCache";

interface View { x: number; y: number; scale: number }

// Below this many pixels per tile there is no point paying for a texture
// lookup — a coloured square and a thumbnail are indistinguishable.
const THUMB_MIN_PX = 3.5;
const LABEL_MAX_SCALE = 4200;
const ANIM_MS = 750;

const easeInOut = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

export function MapCanvas({
  bundle, facet, selected, onSelect,
}: {
  bundle: Bundle;
  facet: Facet;
  selected: number | null;
  onSelect: (index: number | null) => void;
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

  const { works, layouts, atlasMeta, sheets, facets } = bundle;
  const radius = layouts.work_radius;
  const images = useMemo(() => new ImageCache(facets.iiif), [facets.iiif]);

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

      ctx.imageSmoothingEnabled = size < tile;

      // Past this size an atlas sprite is visibly blocky, so ask IIIF for
      // the real thing. Only visible works are ever requested.
      const wantReal = size >= TIER2_MIN_PX;

      for (let i = 0; i < works.length; i++) {
        const px = (pos[i * 2] - vx) * s + cx;
        const py = (pos[i * 2 + 1] - vy) * s + cy;
        if (px < -pad || py < -pad || px > w + pad || py > h + pad) continue;

        // Every tier draws the work's true shape. `size` is the box it has
        // to fit inside, not the shape it takes: a 1:3 hanging scroll is a
        // sliver, a wide landscape is a band.
        const ar = works[i].ar ?? 1;
        const tw = ar >= 1 ? size : size * ar;
        const th = ar >= 1 ? size / ar : size;
        const hw = tw / 2, hh = th / 2;

        if (wantReal) {
          const real = images.get(works[i].img, size);
          if (real) {
            ctx.drawImage(real, px - hw, py - hh, tw, th);
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
  }, [works, layouts, facet, atlasMeta, sheets, radius, selected, fit, ready, images]);

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

  const onWheel = (e: React.WheelEvent) => {
    const before = toWorld(e.clientX, e.clientY);
    const view = viewRef.current;
    const next = view.scale * Math.exp(-e.deltaY * 0.0016);
    const rect = canvasRef.current!.getBoundingClientRect();
    view.scale = Math.min(Math.max(next, Math.min(rect.width, rect.height) * 0.5), 260000);
    const after = toWorld(e.clientX, e.clientY);
    view.x += before.x - after.x;
    view.y += before.y - after.y;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
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
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => { dragRef.current = null; }}
    />
  );
}
