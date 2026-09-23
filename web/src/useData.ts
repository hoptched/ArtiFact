import { useEffect, useState } from "react";
import type { AtlasMeta, Facets, Layouts, Work } from "./types";

export interface Bundle {
  works: Work[];
  layouts: Layouts;
  facets: Facets;
  neighbors: Record<string, number[]>;
  atlasMeta: AtlasMeta;
  sheets: HTMLImageElement[];
  hires: { meta: AtlasMeta; slots: number[] } | null;
}

const base = import.meta.env.BASE_URL;

async function json<T>(path: string): Promise<T> {
  const res = await fetch(base + path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
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
        // thumbnails, so nothing waits on 2.7 MB of JPEG.
        const [works, layouts, facets, atlasMeta] = await Promise.all([
          json<Work[]>("data/works.json"),
          json<import("./types").Layouts>("data/layouts.json"),
          json<import("./types").Facets>("data/facets.json"),
          json<AtlasMeta>("atlas/atlas.json"),
        ]);
        if (cancelled) return;
        const neighbors = await json<Record<string, number[]>>("data/neighbors.json");
        const sheets: HTMLImageElement[] = [];
        setBundle({ works, layouts, facets, neighbors, atlasMeta, sheets,
                    hires: null });

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

        for (let s = 0; s < atlasMeta.sheets; s++) {
          const img = await loadImage(
            `${base}atlas/atlas_${String(s).padStart(3, "0")}.jpg`,
          );
          if (cancelled) return;
          sheets[s] = img;
          setBundle((b) => (b ? { ...b, sheets: [...sheets] } : b));
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return { bundle, error };
}
