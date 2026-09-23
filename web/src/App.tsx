import { useState } from "react";
import { MapCanvas } from "./MapCanvas";
import { Detail } from "./Detail";
import { useBundle } from "./useData";
import type { Facet } from "./types";

const FACETS: { key: Facet; label: string }[] = [
  { key: "period", label: "Period" },
  { key: "country", label: "Place" },
  { key: "style", label: "Style" },
];

export default function App() {
  const { bundle, error } = useBundle();
  const [facet, setFacet] = useState<Facet>("period");
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
