import type { AtlasMeta } from "./types";

/**
 * The high-resolution atlas, loaded a sheet at a time.
 *
 * At 192px a tile is sharp to roughly 170px on screen, which covers the
 * zoom range you actually browse at — but the whole atlas is ~23 MB and
 * ~1.5 GB decoded, so it cannot all be held. It is packed along a Hilbert
 * curve through the UMAP plane instead, which puts a screenful of the map
 * in about 3.5 sheets rather than all 92, making "load what is visible"
 * a real strategy rather than a euphemism for loading everything.
 *
 * Sheets are evicted least-recently-needed, so panning across the map
 * keeps memory flat instead of accumulating every sheet ever touched.
 */
export class HiResAtlas {
  private sheets = new Map<number, HTMLImageElement>();
  private ready = new Set<number>();
  private loading = new Set<number>();
  private lastNeeded = new Map<number, number>();
  private tick = 0;

  constructor(
    private base: string,
    public meta: AtlasMeta & { prefix?: string },
    private maxSheets = 14,
    private maxInflight = 3,
  ) {}

  /** Slot -> its sheet. */
  sheetOf(slot: number) {
    return (slot / this.meta.per_sheet) | 0;
  }

  /** The sheet image if it is loaded, else null. Marks it as wanted. */
  get(slot: number): HTMLImageElement | null {
    const s = this.sheetOf(slot);
    this.lastNeeded.set(s, ++this.tick);
    return this.ready.has(s) ? this.sheets.get(s)! : null;
  }

  /** Start loading any wanted sheet that is missing, nearest need first. */
  pump(wanted: Iterable<number>) {
    const want = [...new Set(wanted)].filter(
      (s) => !this.ready.has(s) && !this.loading.has(s),
    );
    for (const s of want) {
      if (this.loading.size >= this.maxInflight) break;
      this.load(s);
    }
    this.evict();
  }

  private load(sheet: number) {
    this.loading.add(sheet);
    const img = new Image();
    img.referrerPolicy = "no-referrer";
    const name = `${this.meta.prefix ?? "hires"}_${String(sheet).padStart(3, "0")}.jpg`;
    img.onload = () => {
      this.loading.delete(sheet);
      this.sheets.set(sheet, img);
      this.ready.add(sheet);
    };
    img.onerror = () => { this.loading.delete(sheet); };
    img.src = `${this.base}atlas/${name}`;
  }

  private evict() {
    while (this.ready.size > this.maxSheets) {
      let oldest = -1, oldestTick = Infinity;
      for (const s of this.ready) {
        const t = this.lastNeeded.get(s) ?? 0;
        if (t < oldestTick) { oldestTick = t; oldest = s; }
      }
      if (oldest < 0) break;
      this.ready.delete(oldest);
      this.sheets.delete(oldest);
    }
  }

  get loadedSheets() { return this.ready.size; }
}
