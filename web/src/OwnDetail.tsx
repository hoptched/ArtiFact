import type { Bundle } from "./useData";
import type { CompareResult } from "./compare";
import { iiifUrl } from "./types";
import { Thumb } from "./Thumb";

/** The most common value among the closest matches, and how dominant it is. */
function consensus<T>(
  matches: CompareResult["matches"], pick: (i: number) => T | T[] | null,
): { value: T; share: number } | null {
  const tally = new Map<T, number>();
  let total = 0;
  for (const m of matches.slice(0, 12)) {
    const got = pick(m.index);
    if (got === null || got === undefined) continue;
    for (const v of Array.isArray(got) ? got : [got]) {
      tally.set(v, (tally.get(v) ?? 0) + 1);
      total++;
    }
  }
  if (!total) return null;
  const [value, n] = [...tally].sort((a, b) => b[1] - a[1])[0];
  return { value, share: n / total };
}

export function OwnDetail({
  bundle, url, result, onSelect, onClose,
}: {
  bundle: Bundle;
  url: string;
  result: CompareResult;
  onSelect: (index: number) => void;
  onClose: () => void;
}) {
  const { works, facets } = bundle;
  const period = consensus(result.matches, (i) => works[i].p);
  const place = consensus(result.matches, (i) => works[i].c);
  const best = result.matches[0];

  return (
    <aside className="detail">
      <button className="close" onClick={onClose} aria-label="Close">×</button>

      <div className="herobox">
        <img src={url} alt="your picture" />
      </div>

      <h2>Your picture</h2>
      <p className="artist">Compared against {works.length.toLocaleString()} works</p>

      <dl>
        <dt>Style</dt>
        <dd>
          {result.style ? (
            <>
              <span className="predicted">{result.style.label}</span>
              <span className="muted"> predicted</span>
              <span className="conf">
                <em>{result.style.confidence.toFixed(2)}</em>
              </span>
            </>
          ) : <span className="withheld">not predicted</span>}
        </dd>

        {/* Not claims about the picture — claims about its neighbours. A
            photograph of a field is not from 1850 because the works it
            resembles are. */}
        <dt>Nearest are</dt>
        <dd>
          {period
            ? <>{period.value} <span className="muted">
                ({Math.round(period.share * 100)}% of the closest)</span></>
            : <span className="muted">—</span>}
        </dd>

        <dt>Mostly from</dt>
        <dd>
          {place
            ? <>{place.value} <span className="muted">
                ({Math.round(place.share * 100)}%)</span></>
            : <span className="muted">—</span>}
        </dd>

        <dt>Closest</dt>
        <dd>
          {best
            ? <>{(best.similarity * 100).toFixed(1)}%{" "}
                <span className="muted">alike</span></>
            : <span className="muted">—</span>}
        </dd>
      </dl>

      <h3>Most like yours</h3>
      <div className="similar">
        {result.matches.slice(0, 8).map((m) => {
          const w = works[m.index];
          if (!w) return null;
          return (
            <button
              key={w.id}
              onClick={() => onSelect(m.index)}
              title={`${w.t} — ${(m.similarity * 100).toFixed(0)}% alike`}
              style={{ background: w.k ?? undefined }}
            >
              <Thumb src={iiifUrl(facets.iiif, w.img, 200)} alt={w.t}
                     bundle={bundle} index={m.index} />
            </button>
          );
        })}
      </div>
      <p className="muted small note-foot">
        Encoded in your browser against the same model the collection was.
        The picture is never uploaded.
      </p>
    </aside>
  );
}
