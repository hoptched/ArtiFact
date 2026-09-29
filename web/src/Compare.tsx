import { useCallback, useRef, useState } from "react";
import type { Bundle } from "./useData";
import type { CompareResult } from "./compare";
import { Comparer, placeAmong } from "./compare";
import type { Facet } from "./types";

const comparer = new Comparer();

export interface Pin { url: string; result: CompareResult }

export function Compare({
  bundle, facet, onFocus, onPin, onOpen,
}: {
  bundle: Bundle;
  facet: Facet;
  onFocus: (x: number, y: number) => void;
  onPin: (pin: Pin) => void;
  onOpen: () => void;
}) {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  // Held for the pin and the panel, which both draw from it, and shown
  // here only while the comparison runs.
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (file: File) => {
    setError(null);
    setBusy(true);
    setResult(null);
    const url = URL.createObjectURL(file);
    // Not revoked when the next picture arrives: the earlier ones stay
    // on the map and keep drawing from theirs.
    setObjectUrl(url);
    try {
      await comparer.load(setStatus);
      setStatus("Looking…");
      const res = await comparer.compare(file);
      setResult(res);
      onPin({ url, result: res });
      const layout = bundle.layouts.facets[facet];
      const at = placeAmong(
        res.matches,
        (i) => (layout.xy[i] ?? null),
        layout.region_of?.length ? layout.region_of : null,
      );
      if (at) onFocus(at[0], at[1]);
      setStatus("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
    } finally {
      setBusy(false);
    }
  }, [bundle, facet, onFocus, onPin]);

  return (
    <section className="compare">
      <h2>Your own picture</h2>

      {/* Always opens the picker, and goes back to saying so the moment
          the comparison is done: the results live in the panel on the
          right and the picture itself is on the map, so the only job
          left here is taking the next one. Holding the last picture in
          the slot made it look occupied rather than ready. */}
      <button
        className="drop"
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
          : <span>Drop an image, or choose one</span>}
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

      {result && !status && (
        <p className="verdict">
          {result.style && (
            <>
              <b>{result.style.label}</b>{" "}
              <span className="muted">{result.style.confidence.toFixed(2)}</span>
            </>
          )}
          <button className="link" onClick={onOpen}>details</button>
        </p>
      )}
    </section>
  );
}
