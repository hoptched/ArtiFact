import type { Bundle } from "./useData";
import { aicUrl, iiifUrl, percent } from "./types";
import { Thumb } from "./Thumb";
import type { Facet } from "./types";

function Confidence({ value }: { value: number }) {
  // 'Predicted' belongs with the number rather than with the label: it
  // is the number that says how far to trust it, and the two should
  // carry the same weight rather than the word sitting up beside the
  // style in a larger face.
  return <span className="conf"><em>predicted · {percent(value)}</em></span>;
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
      {/* The picture is the way through to the museum's own page, which
          has the provenance, the medium and the rest of what is not
          worth repeating here. */}
      <div className="herobox">
        <a href={aicUrl(work.id)} target="_blank" rel="noreferrer"
           title="See this work at the Art Institute of Chicago">
          <Thumb src={iiifUrl(facets.iiif, work.img, 843)} alt={work.t} />
        </a>
      </div>

      <h2>{work.t}</h2>
      {work.a && <p className="artist">{work.a.split("\n")[0]}</p>}

      {/* Each fact is a way into the map: the grouping it belongs to,
          centred on this work. */}
      <dl>
        <dt>Date</dt>
        <dd>
          {/* The museum's own wording, and nothing else. The bin and the
              precision behind it are how the timeline is built, not what
              anyone came to the panel to read. */}
          <button className="jump" onClick={() => onGoTo("period")}
                  title="Find this work on the timeline">
            {work.d ?? "—"}
          </button>
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
            <span className="withheld">Outside the taxonomy</span>
          ) : (
            <>
              <button className="jump predicted" onClick={() => onGoTo("style")}
                      title={`Find this work among ${work.s}`}>
                {work.s}
              </button>
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
            <button key={id} onClick={() => onSelect(i)} title={other.t}
                    style={{ background: other.k ?? undefined }}>
              <Thumb src={iiifUrl(facets.iiif, other.img, 200)}
                     alt={other.t} bundle={bundle} index={i} />
            </button>
          );
        })}
      </div>
      </>}
    </aside>
  );
}
