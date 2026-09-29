import { useEffect, useRef, useState } from "react";

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

export function Thumb({ src, alt }: { src: string; alt: string }) {
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

  if (dead) return null;
  return (
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
  );
}
