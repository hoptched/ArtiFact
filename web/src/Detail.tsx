import type { Bundle } from "./useData";
import { iiifUrl } from "./types";

const CONFIDENCE_STEPS = 3;

function Confidence({ value }: { value: number }) {
  const filled = Math.max(1, Math.round(value * CONFIDENCE_STEPS));
  return (
    <span className="conf" title={`confidence ${value.toFixed(2)}`}>
      {Array.from({ length: CONFIDENCE_STEPS }, (_, i) => (
        <i key={i} className={i < filled ? "on" : ""} />
      ))}
      <em>{value.toFixed(2)}</em>
    </span>
  );
}

export function Detail({
  bundle, index, onSelect, onClose,
}: {
  bundle: Bundle;
  index: number;
  onSelect: (index: number) => void;
  onClose: () => void;
}) {
  const { works, facets, neighbors } = bundle;
  const work = works[index];
  const indexOfId = new Map(works.map((w, i) => [w.id, i]));
  const similar = (neighbors[String(work.id)] ?? []).slice(0, 8);

  return (
    <aside className="detail">
      <button className="close" onClick={onClose} aria-label="Close">×</button>

      {/* The image sizes itself; the box around it absorbs the slack and
          sits empty, so there is no coloured block under a short work. */}
      <div className="herobox">
        <img
          src={iiifUrl(facets.iiif, work.img, 843)}
          alt={work.t}
          loading="eager"
          // AIC 403s a localhost referer; sending none works everywhere.
          referrerPolicy="no-referrer"
        />
      </div>

      <h2>{work.t}</h2>
      {work.a && <p className="artist">{work.a.split("\n")[0]}</p>}

      <dl>
        <dt>Date</dt>
        <dd>
          {work.d ?? "—"}
          <span className="muted"> · {work.p} ({work.prec})</span>
          {work.pb.length > 1 && (
            <span className="muted"> · spans {work.pb.length} periods</span>
          )}
        </dd>

        <dt>Place</dt>
        <dd>{work.c}</dd>

        <dt>Style</dt>
        <dd>
          {work.s === null ? (
            <>
              <span className="withheld">Outside the taxonomy</span>
              <span className="muted note">
                {" "}The classifier covers European painting and Japanese
                ukiyo-e. It has no label that fits this work, so none is shown.
              </span>
            </>
          ) : (
            <>
              <span className="predicted">{work.s}</span>
              <span className="muted"> predicted</span>
              {work.sc !== null && <Confidence value={work.sc} />}
            </>
          )}
        </dd>

        {work.type && (<><dt>Type</dt><dd>{work.type}</dd></>)}
      </dl>

      {/* Neighbours load after the map, so this is briefly absent rather
          than a heading over an empty row. */}
      {similar.length > 0 && <>
      <h3>Visually similar</h3>
      <div className="similar">
        {similar.map((id) => {
          const i = indexOfId.get(id);
          if (i === undefined) return null;
          const other = works[i];
          return (
            <button key={id} onClick={() => onSelect(i)} title={other.t}>
              <img
                src={iiifUrl(facets.iiif, other.img, 200)}
                alt={other.t}
                loading="lazy"
                referrerPolicy="no-referrer"
              />
            </button>
          );
        })}
      </div>
      <p className="muted small note-foot">
        Neighbours come from image similarity alone — no title, date or place
        was used to find them.
      </p>
      </>}
    </aside>
  );
}
