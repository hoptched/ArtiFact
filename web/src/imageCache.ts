import { iiifUrl } from "./types";

// Above this many pixels per tile, a 32px atlas sprite is visibly blocky
// and the work deserves a real image.
export const TIER2_MIN_PX = 26;

// Two buckets rather than a continuous size, so panning around at a given
// zoom reuses what is already cached instead of requesting a new width
// for every few pixels of scale.
// 800 so a deeply zoomed tile is still sharp. Confined sizing means a
// request larger than the source simply returns the source.
const BUCKETS = [200, 400, 800];

function bucketFor(px: number) {
  return BUCKETS.find((b) => b >= px * 1.5) ?? BUCKETS[BUCKETS.length - 1];
}

interface Entry {
  img: HTMLImageElement;
  width: number;
  ready: boolean;
}

/**
 * Viewport-driven IIIF loading with a bounded cache.
 *
 * Only what is on screen is ever requested, at most a few at a time, so
 * panning fast does not queue hundreds of images that scroll away before
 * they arrive. Nothing here blocks a frame: the draw loop asks for an
 * image and uses whatever is ready, falling back to the atlas sprite.
 */
export class ImageCache {
  private entries = new Map<string, Entry>();
  private inflight = 0;
  private queue: { key: string; url: string; width: number }[] = [];

  constructor(
    private template: string,
    private maxEntries = 400,
    private maxInflight = 12,
    // Bounded rather than cleared: a queue that is emptied every frame
    // never drains, because each dropped entry is re-requested on the
    // next frame and dropped again. Only the first few ever load.
    private maxQueue = 80,
  ) {}

  /** The best cached image for this work, or null. Requests one if absent. */
  get(imageId: string, pxOnScreen: number): HTMLImageElement | null {
    const width = bucketFor(pxOnScreen);
    const key = `${imageId}@${width}`;
    const hit = this.entries.get(key);
    if (hit) {
      // Refresh recency: Map preserves insertion order, so re-inserting
      // moves this entry to the newest end for eviction.
      this.entries.delete(key);
      this.entries.set(key, hit);
      return hit.ready ? hit.img : this.smaller(imageId, width);
    }
    this.request(key, iiifUrl(this.template, imageId, width), width);
    return this.smaller(imageId, width);
  }

  /** A smaller already-loaded size, so zooming shows something immediately. */
  private smaller(imageId: string, width: number): HTMLImageElement | null {
    for (const b of BUCKETS) {
      if (b >= width) break;
      const e = this.entries.get(`${imageId}@${b}`);
      if (e?.ready) return e.img;
    }
    return null;
  }

  private request(key: string, url: string, width: number) {
    if (this.entries.has(key)) return;
    const img = new Image();
    const entry: Entry = { img, width, ready: false };
    this.entries.set(key, entry);
    this.evict();

    const start = () => {
      this.inflight++;
      img.onload = () => { entry.ready = true; this.done(); };
      img.onerror = () => { this.entries.delete(key); this.done(); };
      img.src = url;
    };

    if (this.inflight < this.maxInflight) {
      start();
      return;
    }
    this.queue.push({ key, url, width });
    while (this.queue.length > this.maxQueue) {
      const dropped = this.queue.shift();
      if (dropped) this.entries.delete(dropped.key);
    }
  }

  private done(): void {
    this.inflight--;
    const next = this.queue.shift();
    if (!next) return;
    const entry = this.entries.get(next.key);
    if (!entry) { this.done(); return; }
    this.inflight++;
    entry.img.onload = () => { entry.ready = true; this.done(); };
    entry.img.onerror = () => { this.entries.delete(next.key); this.done(); };
    entry.img.src = next.url;
  }

  private evict() {
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }

  /**
   * Drop queued work that has scrolled out of view.
   *
   * Call this when the viewport actually changes, never per frame: every
   * dropped entry is re-requested on the next draw, so clearing each
   * frame starves the queue instead of draining it.
   */
  clearQueue() {
    for (const q of this.queue) this.entries.delete(q.key);
    this.queue.length = 0;
  }
}
