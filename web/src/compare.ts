import type { Work } from "./types";

export interface CompareMeta {
  count: number;
  dim: number;
  scale: number;
  fingerprint: string;
  backbone: string;
  image_size: number;
}

export interface Match {
  index: number;
  similarity: number;
}

export interface CompareResult {
  matches: Match[];
  style: { label: string; confidence: number } | null;
  vector: Float32Array;
}

const base = import.meta.env.BASE_URL;

/**
 * Compares an uploaded picture against the corpus, entirely in the browser.
 *
 * The site has no backend, so the same CLIP vision tower the pipeline used
 * runs client-side on a quantized export, and the corpus vectors are
 * searched locally. Everything here is loaded on first use and never
 * before: the encoder is ~53 MB and the vectors 4.7 MB, which is a fair
 * price for a feature someone asked for and an unreasonable one for a map
 * they only wanted to look at.
 *
 * The uploaded image never leaves the machine.
 */
export class Comparer {
  private meta: CompareMeta | null = null;
  private vectors: Int8Array | null = null;
  private norms: Float32Array | null = null;
  private head: {
    labels: string[]; coef: number[][]; intercept: number[];
    fingerprint: string;
  } | null = null;
  private extractor: unknown = null;
  private loading: Promise<void> | null = null;

  /** Fetch the model and the corpus vectors. Safe to call repeatedly. */
  load(onProgress?: (what: string) => void): Promise<void> {
    if (!this.loading) this.loading = this.doLoad(onProgress);
    return this.loading;
  }

  get ready() {
    return this.meta !== null && this.vectors !== null && this.extractor !== null;
  }

  private async doLoad(onProgress?: (what: string) => void) {
    onProgress?.("Loading the corpus…");
    const [meta, head, vecBuf, normBuf] = await Promise.all([
      fetch(base + "data/compare.json").then((r) => r.json()),
      fetch(base + "data/style_head.json").then((r) => r.json()),
      fetch(base + "data/embeddings_i8.bin").then((r) => r.arrayBuffer()),
      fetch(base + "data/embeddings_norms.bin").then((r) => r.arrayBuffer()),
    ]);
    this.meta = meta as CompareMeta;
    this.head = head;
    this.vectors = new Int8Array(vecBuf);
    this.norms = new Float32Array(normBuf);

    if (this.head!.fingerprint !== this.meta!.fingerprint) {
      throw new Error("style head and corpus vectors disagree");
    }

    onProgress?.("Loading the image model (~53 MB, once)…");
    const { pipeline } = await import("@huggingface/transformers");
    this.extractor = await pipeline(
      "image-feature-extraction",
      "Xenova/clip-vit-base-patch32",
      // q4f16 is 53 MB against 352 MB for the full weights. Measured on 24
      // corpus images it ranked each one first against itself and returned
      // about 70% of its true top-20 — plenty for "this looks like that",
      // and the quantization of the corpus vectors is not the weak link.
      { dtype: "q4f16", device: "wasm" },
    );
    onProgress?.("");
  }

  /** Encode one image and rank the corpus against it. */
  async compare(image: Blob, k = 24): Promise<CompareResult> {
    await this.load();
    const { RawImage } = await import("@huggingface/transformers");
    const raw = await RawImage.fromBlob(image);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const out = await (this.extractor as any)(raw, { pool: true, normalize: true });
    const q = Float32Array.from(out.data as Float32Array);

    let qn = 0;
    for (const v of q) qn += v * v;
    qn = Math.sqrt(qn) || 1;

    const { count, dim } = this.meta!;
    const vectors = this.vectors!, norms = this.norms!;
    // A flat scan: 9,130 dot products of 512 terms is a few milliseconds,
    // and an index would be more machinery than the problem deserves.
    const scores = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      let dot = 0;
      const off = i * dim;
      for (let d = 0; d < dim; d++) dot += q[d] * vectors[off + d];
      scores[i] = dot / (qn * norms[i]);
    }

    const order = Array.from(scores.keys())
      .sort((a, b) => scores[b] - scores[a])
      .slice(0, k);

    return {
      matches: order.map((index) => ({ index, similarity: scores[index] })),
      style: this.predictStyle(q, qn),
      vector: q,
    };
  }

  /** The same linear head the pipeline fitted, as a matrix multiply. */
  private predictStyle(q: Float32Array, qn: number) {
    const head = this.head;
    if (!head) return null;
    const logits = head.coef.map((row, c) => {
      let s = head.intercept[c];
      for (let d = 0; d < row.length; d++) s += row[d] * (q[d] / qn);
      return s;
    });
    const max = Math.max(...logits);
    const exp = logits.map((v) => Math.exp(v - max));
    const total = exp.reduce((a, b) => a + b, 0);
    let best = 0;
    for (let i = 1; i < exp.length; i++) if (exp[i] > exp[best]) best = i;
    return { label: head.labels[best], confidence: exp[best] / total };
  }
}

/** Where to draw the uploaded image on a given layout: with its neighbours. */
export function placeAmong(
  matches: Match[], xy: [number, number][], works: Work[],
): [number, number] | null {
  // Similarity-weighted centroid of the closest matches. The map has no way
  // to project a new point through UMAP, but sitting among the works it
  // most resembles is what the position would have meant anyway.
  const use = matches.slice(0, 8).filter((m) => works[m.index]);
  if (!use.length) return null;
  let wx = 0, wy = 0, total = 0;
  for (const m of use) {
    const w = Math.max(m.similarity, 0) ** 8;   // sharpen toward the best few
    wx += xy[m.index][0] * w;
    wy += xy[m.index][1] * w;
    total += w;
  }
  return total > 0 ? [wx / total, wy / total] : null;
}
