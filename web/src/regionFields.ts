import type { Region } from "./types";

/**
 * Region fields, baked once per facet into small textures.
 *
 * These were filled and blurred from scratch every frame — one blurred
 * path per region, and a blur costs in proportion to the area it covers,
 * so the tinted facets grew heavy exactly where the regions are largest.
 * Nothing about a field changes as you move, though: it is fixed in map
 * space, and panning and zooming only decide where it lands. So it is
 * drawn once into an image and afterwards only blitted.
 *
 * The textures are small on purpose. A blur has no detail in it to lose,
 * which is what makes it the one thing that survives being scaled up.
 */

/** Longest side of a field texture, in pixels. */
const TEX_MAX = 256;
/** The blur eats inward as well as out, so the shape is grown first. */
const GLOW_GROW = 1.05;
/** Blur radius as a fraction of the region's radius. */
const BLUR_FRACTION = 0.13;
/** Gaussian tails, so the texture holds the whole falloff and not a cut. */
const TAIL = 3;

export interface Field {
  name: string;
  tex: HTMLCanvasElement;
  /** Where the texture belongs, in the map's own 0..1 space. */
  x0: number; y0: number; w: number; h: number;
}

/**
 * A region's tint.
 *
 * Not a categorical palette: at 43 style regions, assigning identity by
 * hue would mean cycling or generating colours, and no reader can tell 43
 * hues apart anyway. Identity stays with the label, which every region
 * already has.
 *
 * The hue is ranked by position in the layout and spread over the whole
 * wheel, so regions that look alike are tinted alike while the map still
 * uses more than one part of the spectrum. Lightness and chroma are held
 * constant, so no region reads as more important than another.
 *
 * Baked opaque: the alpha that matters is the blur's own, and how heavily
 * a field is drawn is decided when it is blitted.
 *
 * Chroma is pushed to the edge of what sRGB can show. At 0.13 the hues
 * were spread over the whole wheel but arrived as pastels once laid at
 * low alpha over a near-black map, so the wheel was there and could not
 * be seen. Where a hue cannot reach this chroma — the blues and violets
 * mostly — the browser maps it back into gamut, which costs those
 * regions some saturation and nothing else.
 */
function tint(hue: number) {
  return `oklch(0.68 0.28 ${hue.toFixed(0)})`;
}

export function buildFields(regions: Region[]): Field[] {
  const out: Field[] = [];
  for (const region of regions) {
    const [rcx, rcy] = region.c;
    const pts = region.o && region.o.length > 2
      ? region.o.map(([qx, qy]): [number, number] =>
          [rcx + (qx - rcx) * GLOW_GROW, rcy + (qy - rcy) * GLOW_GROW])
      : null;

    let x0: number, y0: number, x1: number, y1: number;
    if (pts) {
      x0 = y0 = Infinity; x1 = y1 = -Infinity;
      for (const [qx, qy] of pts) {
        if (qx < x0) x0 = qx;
        if (qx > x1) x1 = qx;
        if (qy < y0) y0 = qy;
        if (qy > y1) y1 = qy;
      }
    } else {
      const r = region.r * GLOW_GROW;
      x0 = rcx - r; x1 = rcx + r; y0 = rcy - r; y1 = rcy + r;
    }

    const blur = Math.max(region.r * BLUR_FRACTION, 1e-6);
    const pad = blur * TAIL;
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
    const w = x1 - x0, h = y1 - y0;
    if (!(w > 0) || !(h > 0)) continue;

    const scale = TEX_MAX / Math.max(w, h);
    const tex = document.createElement("canvas");
    tex.width = Math.max(1, Math.round(w * scale));
    tex.height = Math.max(1, Math.round(h * scale));
    const c = tex.getContext("2d");
    if (!c) continue;

    c.filter = `blur(${(blur * scale).toFixed(2)}px)`;
    c.fillStyle = tint(region.h ?? 0);
    c.beginPath();
    if (pts) {
      const at = (i: number): [number, number] => {
        const [qx, qy] = pts[(i + pts.length) % pts.length];
        return [(qx - x0) * scale, (qy - y0) * scale];
      };
      const [lx, ly] = at(-1);
      const [fx, fy] = at(0);
      c.moveTo((lx + fx) / 2, (ly + fy) / 2);
      for (let i = 0; i < pts.length; i++) {
        const [ax, ay] = at(i);
        const [bx, by] = at(i + 1);
        c.quadraticCurveTo(ax, ay, (ax + bx) / 2, (ay + by) / 2);
      }
    } else {
      c.arc((rcx - x0) * scale, (rcy - y0) * scale,
            region.r * GLOW_GROW * scale, 0, Math.PI * 2);
    }
    c.closePath();
    c.fill();

    out.push({ name: region.name, tex, x0, y0, w, h });
  }
  return out;
}
