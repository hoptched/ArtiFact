import { useCallback, useRef, useState } from "react";
import type { Bundle } from "./useData";
import type { CompareResult } from "./compare";
import { Comparer, placeAmong, regionCentres, styleRegion }
  from "./compare";
import type { Facet } from "./types";

const comparer = new Comparer();

export interface Pin { url: string; result: CompareResult }

export function Compare({
  bundle, facet, onFocus, onPin, onWorking,
}: {
  bundle: Bundle;
  facet: Facet;
  onFocus: (x: number, y: number) => void;
  onPin: (pin: Pin) => void;
  /** What the comparison is doing, or "" when it has finished. The model
   *  runs on this thread, so the rest of the page has to stand aside. */
  onWorking: (stage: string) => void;
}) {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  // Held for the pin and the panel, which both draw from it, and shown
  // here only while the comparison runs.
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (file: File) => {
    // Every stage goes to both: the slot shows it, and the page uses it
    // to know it should stop taking input.
    const stage = (text: string) => { setStatus(text); onWorking(text); };
    setError(null);
    setBusy(true);
    stage("Waking the model…");
    const url = URL.createObjectURL(file);
    // Not revoked when the next picture arrives: the earlier ones stay
    // on the map and keep drawing from theirs.
    setObjectUrl(url);
    try {
      await comparer.load(stage);
      stage("Looking…");
      const res = await comparer.compare(file);
      onPin({ url, result: res });
      // The same rule the map draws the picture with. Aimed without it,
      // the flight went to where the neighbours are while the pin was
      // drawn where the style says, and the two were different places.
      const layout = bundle.layouts.facets[facet];
      const at = placeAmong(
        res.matches,
        (i) => (layout.xy[i] ?? null),
        layout.region_of?.length ? layout.region_of : null,
        styleRegion(bundle.layouts, facet, res.style?.label ?? null),
        regionCentres(bundle.layouts, facet),
      );
      if (at) onFocus(at[0], at[1]);
      stage("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      stage("");
    } finally {
      setBusy(false);
      // Here as well as on each path out, so no failure anyone has not
      // thought of can leave the page locked behind the overlay.
      stage("");
    }
  }, [bundle, facet, onFocus, onPin, onWorking]);

  return (
    <section className="compare">
      <h2>Add your own picture :)</h2>

      {/* Always opens the picker, and goes back to saying so the moment
          the comparison is done: the results live in the panel on the
          right and the picture itself is on the map, so the only job
          left here is taking the next one. Holding the last picture in
          the slot made it look occupied rather than ready. */}
      <button
        className="drop"
        // The heading above names it on screen; empty, it needs to name
        // itself to anything not reading the page by eye.
        aria-label="Drop an image here, or click to choose one"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f) void run(f);
        }}
      >
        {busy && objectUrl
          ? <img src={objectUrl} alt="the picture being compared" />
          : <span>we will try to guess...</span>}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void run(f);
          e.target.value = "";     // so the same file can be picked twice
        }}
      />

      {status && <p className="muted small">{status}</p>}
      {error && <p className="muted small">Could not compare: {error}</p>}

    </section>
  );
}
