import { useEffect, useRef, useState } from "react";
import type { Bundle } from "./useData";

/**
 * An image that tries again.
 *
 * The collection's pictures come from the museum's IIIF server, which
 * fails intermittently — a 500 or a Cloudflare 521, on a URL that serves
 * fine a second later. An <img> does not retry, so one blip left a
 * permanently broken cell until the panel was closed and reopened, and a
 * row of eight thumbnails gave that blip eight chances to happen.
 *
 * Remounting the element is what re-requests it: setting the same src on
 * a failed image does nothing, so the attempt number is its key. Backoff
 * is there because these failures come in runs — retrying instantly just
 * spends the attempts inside the same bad moment.
 */
const BACKOFF_MS = [400, 1200, 3000];

/**
 * The atlas cell for a work, drawn straight away.
 *
 * The museum generates each derivative on demand, so a thumbnail nobody
 * has asked for before can take seconds to come back — the request goes
 * out at once, the picture does not. But every work is already in the
 * 64px atlas, downloaded when the map loaded, so the right picture can
 * be on screen immediately and the network request only has to sharpen
 * it. Soft at this size, and soft is not what anyone was complaining
 * about.
 *
 * Cropped square from the centre, because the cell letterboxes the work
 * inside a square and the slot it goes into is object-fit: cover.
 */
function AtlasCell({ bundle, index }: { bundle: Bundle; index: number }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const { sheets, atlasMeta, works } = bundle;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const { tile, grid, per_sheet } = atlasMeta;
    const slot = bundle.slots[index];
    const sheet = sheets[(slot / per_sheet) | 0];
    const ctx = canvas.getContext("2d");
    if (!sheet || !ctx) return;
    const within = slot % per_sheet;
    const ar = works[index].ar ?? 1;
    const iw = ar >= 1 ? tile : tile * ar;
    const ih = ar >= 1 ? tile / ar : tile;
    const side = Math.min(iw, ih);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(
      sheet,
      (within % grid) * tile + (tile - iw) / 2 + (iw - side) / 2,
      ((within / grid) | 0) * tile + (tile - ih) / 2 + (ih - side) / 2,
      side, side,
      0, 0, canvas.width, canvas.height,
    );
  }, [bundle, index, sheets, atlasMeta, works]);

  return <canvas ref={ref} width={128} height={128} className="cell" />;
}

export function Thumb(
  { src, alt, bundle, index }:
  { src: string; alt: string; bundle?: Bundle; index?: number },
) {
  const [attempt, setAttempt] = useState(0);
  const [dead, setDead] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  // A new src is a different picture: drop the history with it, or a
  // work that failed leaves the next one it is replaced by marked dead.
  useEffect(() => {
    setAttempt(0);
    setDead(false);
    return () => window.clearTimeout(timer.current);
  }, [src]);

  if (dead) return bundle && index !== undefined
    ? <AtlasCell bundle={bundle} index={index} /> : null;
  return (
    <>
      {bundle && index !== undefined
        && <AtlasCell bundle={bundle} index={index} />}
      <img
      key={attempt}
      src={src}
      alt={alt}
      // AIC 403s a localhost referer; sending none works everywhere.
      referrerPolicy="no-referrer"
      onError={() => {
        const wait = BACKOFF_MS[attempt];
        if (wait === undefined) { setDead(true); return; }
        timer.current = window.setTimeout(
          () => setAttempt((a) => a + 1), wait);
      }}
      />
    </>
  );
}
