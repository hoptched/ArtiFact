import type { Bundle } from "./useData";
import { iiifUrl } from "./types";
import { Thumb } from "./Thumb";
import type { Facet } from "./types";

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
  bundle, index, onSelect, onGoTo, onClose,
}: {
  bundle: Bundle;
  index: number;
  onSelect: (index: number) => void;
  /** Follow one of this work's facts onto the map. */
  onGoTo: (facet: Facet) => void;
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
        <Thumb src={iiifUrl(facets.iiif, work.img, 843)} alt={work.t} eager />
      </div>

      <h2>{work.t}</h2>
      {work.a && <p className="artist">{work.a.split("\n")[0]}</p>}

      {/* Each fact is a way into the map: the grouping it belongs to,
          centred on this work. */}
      <dl>
        <dt>Date</dt>
        <dd>
          <button className="jump" onClick={() => onGoTo("period")}
                  title="Find this work on the timeline">
            {work.d ?? "—"}
          </button>
          <span className="muted"> · {work.p} ({work.prec})</span>
          {work.pb.length > 1 && (
            <span className="muted"> · spans {work.pb.length} periods</span>
          )}
        </dd>

        <dt>Place</dt>
        <dd>
          <button className="jump" onClick={() => onGoTo("country")}
                  title={`Find this work among works from ${work.c}`}>
            {work.c}
          </button>
        </dd>

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
              <button className="jump predicted" onClick={() => onGoTo("style")}
                      title={`Find this work among ${work.s}`}>
                {work.s}
              </button>
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
      <h3>
        <button className="jump" onClick={() => onGoTo("similarity")}
                title="Find this work among its neighbours on the map">
          Visually similar
        </button>
      </h3>
      <div className="similar">
        {similar.map((id) => {
          const i = indexOfId.get(id);
          if (i === undefined) return null;
          const other = works[i];
          return (
            <button key={id} onClick={() => onSelect(i)} title={other.t}>
              <Thumb src={iiifUrl(facets.iiif, other.img, 200)}
                     alt={other.t} />
            </button>
          );
        })}
      </div>
      </>}
    </aside>
  );
}
