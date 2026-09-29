export type Facet = "similarity" | "period" | "country" | "style";

export interface Work {
  id: number;
  t: string;            // title
  a: string | null;     // artist_display, keeps "After Raphael"
  img: string;          // IIIF image id
  d: string | null;     // date as the museum writes it
  pb: number[];         // every period bin the date range overlaps
  p: string;            // the midpoint bin's label
  prec: "exact" | "loose" | "vague";
  c: string;            // country
  s: string | null;     // predicted style, null outside the taxonomy
  sc: number | null;    // confidence
  xy: [number, number]; // UMAP position, similarity only
  type: string | null;
  k?: string;           // dominant colour
  ar?: number;          // aspect ratio, width / height
}

export interface Region {
  name: string;
  c: [number, number];
  r: number;
  n: number;
  /** Closed outline following the works inside, in the same 0..1 space. */
  o?: [number, number][];
  /** Hue in degrees, spread evenly but ordered by similarity. */
  h?: number;
}

export interface Layouts {
  work_radius: number;
  facets: Record<Facet, {
    xy: [number, number][];
    /** Tile radius this facet can carry; a ribbon differs from a disc. */
    work_radius?: number;
    /** Century boundaries along a timeline facet. */
    grid?: { x: number; label: string }[];
    regions: Region[];
    region_of: string[];
  }>;
}

export interface Facets {
  periods: { bin: number; label: string; count: number }[];
  countries: { name: string; count: number }[];
  styles: { name: string; count: number }[];
  style_labels: string[];
  outside_taxonomy_label: string;
  style_domain_note: string;
  iiif: string;
  counts: { works: number; neighbors_per_work: number };
}

export interface AtlasMeta {
  prefix?: string;
  tile: number;
  sheet_px: number;
  grid: number;
  per_sheet: number;
  sheets: number;
  count: number;
}

// The template carries {width} twice, for IIIF confined sizing (!w,h),
// so every occurrence has to be replaced.
export function iiifUrl(template: string, imageId: string, width: number) {
  return template
    .replaceAll("{image_id}", imageId)
    .replaceAll("{width}", String(width));
}

/** A classifier probability as a percentage. Rounded to whole points:
 *  the second decimal was reading as precision the number does not have,
 *  and nothing anyone does with it turns on a hundredth. */
export function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}
