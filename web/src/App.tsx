import { useCallback, useEffect, useState } from "react";
import { MapCanvas } from "./MapCanvas";
import type { Focus, MapPin } from "./MapCanvas";
import type { Pin } from "./Compare";
import { Detail } from "./Detail";
import { Compare } from "./Compare";
import { OwnDetail } from "./OwnDetail";
import { useBundle } from "./useData";
import type { Facet } from "./types";

const facetNote: Record<Facet, string> = {
  similarity: "Position is the image alone. Neighbours look alike, and "
    + "there are no regions to click.",
  period: "A continuous timeline. Bins are ordered by how their works "
    + "look, not by date — that order turns out to be chronological — and "
    + "within each, works run oldest to newest along the arc.",
  country: "Regions are present-day countries, placed by how their works look.",
  style: "Regions are predicted styles, plus one for the works no "
    + "style in the taxonomy fits. Click a region to go there.",
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
  const [pin, setPin] = useState<Pin | null>(null);
  const [mapPin, setMapPin] = useState<MapPin | null>(null);
  const [showOwn, setShowOwn] = useState(false);

  const flyTo = useCallback(
    (x: number, y: number) => setFocus({ x, y, r: 0.06, key: Date.now() }), []);

  // Only one thing is ever the selected thing. The uploaded picture and a
  // work from the collection wear the same halo and open panels that
  // occupy the same place, so choosing either has to release the other.
  const selectWork = useCallback((i: number | null) => {
    setSelected(i);
    if (i !== null) setShowOwn(false);
  }, []);
  const openPin = useCallback(() => {
    setShowOwn(true);
    setSelected(null);
  }, []);

  // Follow one of a work's own facts onto the map: switch to that
  // grouping and travel to where this work lands in it. The works are
  // still moving into place while the flight runs, and both finish at
  // the same coordinates, so the view arrives with the work rather than
  // waiting for it.
  const goTo = useCallback((f: Facet) => {
    if (!bundle || selected === null) return;
    const layout = bundle.layouts.facets[f];
    const at = layout.xy[selected];
    if (!at) return;
    const r = (layout.work_radius ?? bundle.layouts.work_radius) * 3.8;
    setFacet(f);
    setFocus({ x: at[0], y: at[1], r, key: Date.now() });
  }, [bundle, selected]);

  // The map draws a decoded image, so hold the pin back until it has
  // loaded rather than flashing an empty frame where the picture goes.
  useEffect(() => {
    if (!pin) { setMapPin(null); return; }
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (live) setMapPin({ img, matches: pin.result.matches });
    };
    img.src = pin.url;
    return () => { live = false; };
  }, [pin]);

  if (error) return <div className="status">Could not load the map: {error}</div>;
  if (!bundle) return <div className="status">Loading the collection…</div>;

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>ArtiFact</h1>
          <p>
            A collection of {bundle.works.length.toLocaleString()}{" "}
            public-domain works from the Art Institute of Chicago.
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

        <Compare
          bundle={bundle}
          facet={facet}
          onFocus={flyTo}
          onPin={(p) => { setPin(p); setShowOwn(p !== null); }}
          onOpen={() => setShowOwn(true)}
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
          onSelect={selectWork}
          focus={focus}
          pin={mapPin}
          pinActive={showOwn}
          onOpenPin={openPin}
          onRegion={(x, y, r) =>
            setFocus({ x, y, r: Math.max(r, 0.02), key: Date.now() })}
        />
      </div>

      {showOwn && pin && (
        <OwnDetail
          bundle={bundle}
          url={pin.url}
          result={pin.result}
          onSelect={(i) => { setShowOwn(false); setSelected(i); }}
          onClose={() => setShowOwn(false)}
        />
      )}

      {selected !== null && !showOwn && (
        <Detail
          bundle={bundle}
          index={selected}
          onSelect={selectWork}
          onGoTo={goTo}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
