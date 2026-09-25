import { useCallback, useRef, useState } from "react";
import type { Bundle } from "./useData";
import type { CompareResult } from "./compare";
import { Comparer, placeAmong } from "./compare";
import { iiifUrl } from "./types";
import type { Facet } from "./types";

const comparer = new Comparer();

export interface Pin { url: string; result: CompareResult }

export function Compare({
  bundle, facet, onSelect, onFocus, onPin, onOpen,
}: {
  bundle: Bundle;
  facet: Facet;
  onSelect: (index: number) => void;
  onFocus: (x: number, y: number) => void;
  onPin: (pin: Pin | null) => void;
  onOpen: () => void;
}) {
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (file: File) => {
    setError(null);
    setBusy(true);
    setResult(null);
    const url = URL.createObjectURL(file);
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return url; });
    onPin(null);
    try {
      await comparer.load(setStatus);
      setStatus("Looking…");
      const res = await comparer.compare(file);
      setResult(res);
      onPin({ url, result: res });
      const xy = bundle.layouts.facets[facet].xy;
      const at = placeAmong(res.matches, xy, bundle.works);
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

      <button
        className="drop"
        disabled={busy}
        onClick={() => (result ? onOpen() : inputRef.current?.click())}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f) void run(f);
        }}
      >
        {preview
          ? <img src={preview} alt="your picture" />
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
        }}
      />

      {status && <p className="muted small">{status}</p>}
      {error && <p className="muted small">Could not compare: {error}</p>}

      {result && (
        <>
          <button className="reopen" onClick={onOpen}>See the full comparison</button>
          {result.style && (
            <p className="verdict">
              Closest style: <b>{result.style.label}</b>{" "}
              <span className="muted">
                {result.style.confidence.toFixed(2)}
              </span>
            </p>
          )}
          <div className="matches">
            {result.matches.slice(0, 9).map((m) => {
              const w = bundle.works[m.index];
              return (
                <button
                  key={w.id}
                  title={`${w.t} — ${(m.similarity * 100).toFixed(0)}% alike`}
                  onClick={() => onSelect(m.index)}
                >
                  <img
                    src={iiifUrl(bundle.facets.iiif, w.img, 200)}
                    alt={w.t}
                    loading="lazy"
                    referrerPolicy="no-referrer"
                  />
                </button>
              );
            })}
          </div>
          <p className="muted small">
            Encoded in your browser. The picture is never uploaded.
          </p>
        </>
      )}
    </section>
  );
}
