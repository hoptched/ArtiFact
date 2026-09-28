
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
  private model: unknown = null;
  private processor: unknown = null;
  private loading: Promise<void> | null = null;

  /** Fetch the model and the corpus vectors. Safe to call repeatedly. */
  load(onProgress?: (what: string) => void): Promise<void> {
    if (!this.loading) this.loading = this.doLoad(onProgress);
    return this.loading;
  }

  get ready() {
    return this.meta !== null && this.vectors !== null && this.model !== null;
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

    onProgress?.("Loading the image model (~173 MB, once)…");
    const { AutoProcessor, CLIPVisionModelWithProjection } =
      await import("@huggingface/transformers");
    // The same class the pipeline used, not the generic feature-extraction
    // pipeline: CLIP's vision export has no pooler, so asking that pipeline
    // to pool fails outright. This one emits image_embeds straight from the
    // projection head — the exact quantity the corpus vectors are.
    //
    // q4f16 is 173 MB against 1.2 GB for the full weights.
    // Must be the backbone the corpus was encoded with, or the vectors
    // do not share a space. ViT-L/14 scored 0.540 against B/32's 0.464 on
    // identical works with artists held out, which is worth 173 MB here.
    const repo = "Xenova/clip-vit-large-patch14";
    [this.processor, this.model] = await Promise.all([
      AutoProcessor.from_pretrained(repo),
      CLIPVisionModelWithProjection.from_pretrained(repo, {
        dtype: "q4f16", device: "wasm",
      }),
    ]);
    onProgress?.("");
  }

  /** Encode one image and rank the corpus against it. */
  async compare(image: Blob, k = 24): Promise<CompareResult> {
    await this.load();
    const { RawImage } = await import("@huggingface/transformers");
    const raw = await RawImage.fromBlob(image);
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const inputs = await (this.processor as any)(raw);
    const out = await (this.model as any)(inputs);
    /* eslint-enable @typescript-eslint/no-explicit-any */
    const embeds = out.image_embeds ?? out.last_hidden_state;
    if (!embeds) throw new Error("the model returned no image_embeds");
    const q = Float32Array.from(embeds.data as Float32Array);

    let qn = 0;
    for (const v of q) qn += v * v;
    qn = Math.sqrt(qn) || 1;

    const { count, dim } = this.meta!;
    const vectors = this.vectors!, norms = this.norms!;
    // A flat scan: 25,515 dot products of 768 terms is a few
    // milliseconds, and an index would be more machinery than the
    // problem deserves.
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

/** How many matches decide where the picture belongs. */
export const ANCHOR_K = 12;

/**
 * Where to draw the uploaded picture on a given layout.
 *
 * Not the centroid of its neighbours. For the Mona Lisa the twelve nearest
 * works split Italy 5, France 4, Belgium 2, Spain 1 — all Raphael-ish head
 * studies, scattered by the museum's attribution of where they were made —
 * and their weighted centroid lands inside Belgium, a region holding two of
 * the twelve. Averaging positions across separate clusters gives a point
 * belonging to none of them, and the similarities here run 0.91 to 0.93, so
 * no weighting rescues it.
 *
 * Instead: take the region most of the neighbours agree on, and average only
 * the ones that are in it. The map then says what the panel says, because
 * both are reading the same consensus.
 */
export function placeAmong(
  matches: Match[],
  pos: (index: number) => [number, number] | null,
  regionOf: string[] | null,
): [number, number] | null {
  const use = matches.slice(0, ANCHOR_K).filter((m) => pos(m.index));
  if (!use.length) return null;

  let chosen = use;
  if (regionOf) {
    const tally = new Map<string, number>();
    for (const m of use) {
      const r = regionOf[m.index];
      if (r) tally.set(r, (tally.get(r) ?? 0) + 1);
    }
    const top = [...tally].sort((a, b) => b[1] - a[1])[0];
    if (top) {
      const inRegion = use.filter((m) => regionOf[m.index] === top[0]);
      if (inRegion.length) chosen = inRegion;
    }
  }

  let wx = 0, wy = 0, total = 0;
  for (const m of chosen) {
    const p = pos(m.index)!;
    const w = Math.max(m.similarity, 0) ** 8;
    wx += p[0] * w; wy += p[1] * w; total += w;
  }
  return total > 0 ? [wx / total, wy / total] : null;
}
