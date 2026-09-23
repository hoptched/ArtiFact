import { useState } from "react";
import { MapCanvas } from "./MapCanvas";
import { Detail } from "./Detail";
import { useBundle } from "./useData";
import type { Facet } from "./types";

const facetNote: Record<Facet, string> = {
  similarity: "Position is the image alone. Neighbours look alike.",
  period: "Regions are 50-year bins, placed by how their works look — "
    + "which recovers chronology on its own.",
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

  if (error) return <div className="status">Could not load the map: {error}</div>;
  if (!bundle) return <div className="status">Loading 9,101 works…</div>;

  return (
    <div className="app">
      <MapCanvas
        bundle={bundle}
        facet={facet}
        selected={selected}
        onSelect={setSelected}
      />

      <header className="chrome">
        <h1>ArtiFact</h1>
        <p>
          {bundle.works.length.toLocaleString()} public-domain works from the
          Art Institute of Chicago, arranged by how they look.
        </p>
        <p className="hint">
          {facetNote[facet]}
        </p>
      </header>

      <nav className="chrome switcher">
        <span className="muted small">Group by</span>
        {FACETS.map((f) => (
          <button
            key={f.key}
            className={f.key === facet ? "on" : ""}
            onClick={() => setFacet(f.key)}
          >
            {f.label}
          </button>
        ))}
      </nav>

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
