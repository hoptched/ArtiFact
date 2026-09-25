import { useState } from "react";
import { MapCanvas } from "./MapCanvas";
import type { Focus } from "./MapCanvas";
import { Detail } from "./Detail";
import { Compare } from "./Compare";
import { useBundle } from "./useData";
import type { Facet } from "./types";

const facetNote: Record<Facet, string> = {
  similarity: "Position is the image alone. Neighbours look alike.",
  period: "A continuous timeline. Bins are ordered by how their works "
    + "look, not by date — that order turns out to be chronological — and "
    + "within each, works run oldest to newest along the arc.",
  country: "Regions are present-day countries, placed by how their works look.",
  style: "Regions are predicted styles. 387 works sit outside the taxonomy.",
};

const FACETS: { key: Facet; label: string }[] = [
  // Similarity first and default: it is the only arrangement where being
  // next to something means the two works look alike.
  { key: "similarity", label: "Similarity" },
  { key: "period", label: "Period" },
  { key: "country", label: "Place" },
  { key: "style", label: "Style" },
];

export default function App() {
  const { bundle, error } = useBundle();
  const [facet, setFacet] = useState<Facet>("similarity");
  const [selected, setSelected] = useState<number | null>(null);
  const [focus, setFocus] = useState<Focus | null>(null);
  const regions = bundle?.layouts.facets[facet].regions ?? [];

  if (error) return <div className="status">Could not load the map: {error}</div>;
  if (!bundle) return <div className="status">Loading 9,101 works…</div>;

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>ArtiFact</h1>
          <p>
            {bundle.works.length.toLocaleString()} public-domain works from
            the Art Institute of Chicago, arranged by how they look.
          </p>
        </header>

        <section className="group">
          <h2>Group by</h2>
          <div className="switcher">
            {FACETS.map((f) => (
              <button
                key={f.key}
                className={f.key === facet ? "on" : ""}
                onClick={() => setFacet(f.key)}
              >
                {f.label}
              </button>
            ))}
          </div>
          <p className="hint">{facetNote[facet]}</p>
        </section>

        {regions.length > 0 && (
          <section className="regions">
            <h2>Jump to</h2>
            <ul>
              {regions.map((r) => (
                <li key={r.name}>
                  <button
                    onClick={() =>
                      setFocus({ x: r.c[0], y: r.c[1], r: r.r, key: Date.now() })
                    }
                  >
                    <span>{r.name}</span>
                    <em>{r.n.toLocaleString()}</em>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <Compare
          bundle={bundle}
          facet={facet}
          onSelect={setSelected}
          onFocus={(x, y) => setFocus({ x, y, r: 0.06, key: Date.now() })}
        />

        <button
          className="reset"
          onClick={() => setFocus({ x: 0.5, y: 0.5, r: 0.5, key: Date.now() })}
        >
          Whole map
        </button>
      </aside>

      <div className="stage">
        <MapCanvas
          bundle={bundle}
          facet={facet}
          selected={selected}
          onSelect={setSelected}
          focus={focus}
        />
      </div>

      {selected !== null && (
        <Detail
          bundle={bundle}
          index={selected}
          onSelect={setSelected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
