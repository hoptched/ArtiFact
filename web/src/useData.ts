import { useEffect, useState } from "react";
import type { AtlasMeta, Facets, Layouts, Work } from "./types";

export interface Bundle {
  works: Work[];
  layouts: Layouts;
  facets: Facets;
  neighbors: Record<string, number[]>;
  atlasMeta: AtlasMeta;
  sheets: HTMLImageElement[];
  /** Quarter-size copies of the sheets, 16px to a tile. */
  mips: HTMLCanvasElement[];
  hires: { meta: AtlasMeta; slots: number[] } | null;
}

const base = import.meta.env.BASE_URL;

async function json<T>(path: string): Promise<T> {
  const res = await fetch(base + path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

/**
 * A quarter-size copy of a sheet: 16px to a tile instead of 64.
 *
 * Zoomed out, a tile is a handful of pixels but the browser still reads
 * the whole 64x64 cell and filters it down. At 3.5px — the smallest size
 * that still draws a thumbnail — that is 21,786 tiles sampling 4,096
 * source pixels each, 89 million a frame, which is why the map dragged
 * just above the threshold and was fine just below it, where cells are
 * flat colour.
 *
 * Halved twice rather than quartered in one step: a single large
 * reduction samples too sparsely to represent what it skips, and the
 * result crawls as you pan.
 */
function mipOf(img: HTMLImageElement, sheetPx: number): HTMLCanvasElement {
  let src: HTMLImageElement | HTMLCanvasElement = img;
  let px = sheetPx;
  for (let step = 0; step < 2; step++) {
    const half = document.createElement("canvas");
    half.width = half.height = px / 2;
    const c = half.getContext("2d");
    if (!c) break;
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = "high";
    c.drawImage(src, 0, 0, px / 2, px / 2);
    src = half;
    px /= 2;
  }
  return src as HTMLCanvasElement;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    // Resolved once it is decoded, not merely fetched, so the first frame
    // that draws a sheet does not also have to decompress it.
    img.onload = () => img.decode().then(() => resolve(img), () => resolve(img));
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

export function useBundle() {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // works and layouts are all the map needs to draw; the atlas
        // sheets arrive after and simply upgrade colour cells to
        // thumbnails, so nothing waits on 24 MB of JPEG.
        const [works, layouts, facets, atlasMeta] = await Promise.all([
          json<Work[]>("data/works.json"),
          json<import("./types").Layouts>("data/layouts.json"),
          json<import("./types").Facets>("data/facets.json"),
          json<AtlasMeta>("atlas/atlas.json"),
        ]);
        if (cancelled) return;

        // Mutated in place as sheets arrive, and never replaced. The draw
        // loop closes over this array and reads it fresh every frame, so
        // a sheet shows up on the next frame without a re-render. Handing
        // React a new array per sheet — which is what this did — meant 25
        // state updates, and since `sheets` is a dependency of the effect
        // that owns the render loop, 25 teardowns and restarts of it
        // during the one moment the map is busiest.
        const sheets: HTMLImageElement[] = [];
        const mips: HTMLCanvasElement[] = [];
        setBundle({ works, layouts, facets, neighbors: {}, atlasMeta, sheets,
                    mips, hires: null });

        // Only read when a detail panel opens, which cannot happen before
        // there is a map to click on. Awaiting it here put 3.5 MB of JSON
        // — fetch plus parse — in front of the first frame.
        json<Record<string, number[]>>("data/neighbors.json")
          .then((neighbors) => {
            if (!cancelled) setBundle((b) => (b ? { ...b, neighbors } : b));
          })
          .catch(() => { /* similar works stay empty */ });

        // Optional: the map works without it, just softer when zoomed in.
        Promise.all([
          json<AtlasMeta>("atlas/hires.json"),
          json<number[]>("data/hires_slots.json"),
        ])
          .then(([meta, slots]) => {
            if (!cancelled) {
              setBundle((b) => (b ? { ...b, hires: { meta, slots } } : b));
            }
          })
          .catch(() => { /* no high-resolution atlas built yet */ });

        // A few at a time rather than one after another. These were
        // awaited in sequence, so the whole atlas cost 25 round trips
        // end to end when the browser will gladly run six at once.
        let next = 0;
        const worker = async () => {
          for (let s = next++; s < atlasMeta.sheets; s = next++) {
            if (cancelled) return;
            try {
              const img = await loadImage(
                `${base}atlas/atlas_${String(s).padStart(3, "0")}.jpg`,
              );
              // Before the sheet is published, so no frame can find a
              // sheet without its reduction and fall back mid-pan.
              mips[s] = mipOf(img, atlasMeta.sheet_px);
              sheets[s] = img;
            } catch { /* one missing sheet leaves colour cells there */ }
          }
        };
        await Promise.all(
          Array.from({ length: Math.min(6, atlasMeta.sheets) }, worker),
        );
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return { bundle, error };
}
