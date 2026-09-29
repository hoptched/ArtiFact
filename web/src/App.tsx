import { useCallback, useEffect, useState } from "react";
import { MapCanvas } from "./MapCanvas";
import type { Focus, MapPin } from "./MapCanvas";
import type { Pin } from "./Compare";
import { Detail } from "./Detail";
import { Compare } from "./Compare";
import { OwnDetail } from "./OwnDetail";
import { About } from "./About";
import { Logo } from "./Logo";
import { useBundle } from "./useData";
import type { Facet } from "./types";

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
  // Every picture anyone has compared this session, oldest first. They
  // stay on the map: the point of putting one there is to see where it
  // falls, and that is a comparison you want to make more than once.
  const [pins, setPins] = useState<Pin[]>([]);
  const [mapPins, setMapPins] = useState<MapPin[]>([]);
  // Which one's panel is open, by index, or null for none.
  const [openPin, setOpenPin] = useState<number | null>(null);

  const flyTo = useCallback(
    (x: number, y: number) => setFocus({ x, y, r: 0.06, key: Date.now() }), []);

  // Only one thing is ever the selected thing. The uploaded picture and a
  // work from the collection wear the same halo and open panels that
  // occupy the same place, so choosing either has to release the other.
  const selectWork = useCallback((i: number | null) => {
    setSelected(i);
    if (i !== null) setOpenPin(null);
  }, []);
  const showPin = useCallback((which: number) => {
    setOpenPin(which);
    setSelected(null);
  }, []);
  const addPin = useCallback((p: Pin) => {
    setOpenPin(pins.length);      // where this one is about to land
    setPins((all) => [...all, p]);
    setSelected(null);
  }, [pins.length]);

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
    // The radius to frame, as a multiple of a work's own. Larger means
    // further out: at 3.8 the work filled about 200px and its neighbours
    // were mostly off screen, which told you where it was but not what
    // it had landed among. At 15 it is around 50px, with a few hundred
    // works around it — its own size on the map, in company.
    const r = (layout.work_radius ?? bundle.layouts.work_radius) * 15;
    setFacet(f);
    setFocus({ x: at[0], y: at[1], r, key: Date.now() });
  }, [bundle, selected]);

  // The map draws decoded images, so a picture is held back until its
  // own has loaded rather than flashing an empty frame where it goes.
  // Rebuilt from the whole list so the array the map reads stays in step
  // with it, and so the indices the two sides use mean the same thing.
  useEffect(() => {
    let live = true;
    Promise.all(pins.map((p) => new Promise<MapPin>((resolve) => {
      const img = new Image();
      const done = () => resolve({ img, matches: p.result.matches });
      img.onload = done;
      // Resolved on failure too, so the array stays the same length as
      // the list and an index means the same picture on both sides. The
      // map skips one that did not decode.
      img.onerror = done;
      img.src = p.url;
    }))).then((loaded) => { if (live) setMapPins(loaded); });
    return () => { live = false; };
  }, [pins]);

  if (error) return <div className="status">Could not load the map: {error}</div>;
  if (!bundle) return <div className="status">Loading the collection…</div>;

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>
            <a href="https://commons.wikimedia.org/wiki/File:Minerva,_by_Rembrandt_(1635).jpg#Summary"
               target="_blank" rel="noreferrer"
               title="Minerva, by Rembrandt (1635)">
              <Logo />
            </a>
          </h1>
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
                aria-pressed={f.key === facet}
              >
                <span className="led" />{f.label}
              </button>
            ))}
          </div>
        </section>

        <Compare
          bundle={bundle}
          facet={facet}
          onFocus={flyTo}
          onPin={addPin}
          onOpen={() => setOpenPin(pins.length - 1)}
        />

        <button
          className="reset"
          onClick={() => setFocus({ x: 0.5, y: 0.5, r: 0.5, key: Date.now() })}
        >
          Whole map
        </button>
        <About works={bundle.works.length} />
      </aside>

      <div className="stage">
        <MapCanvas
          bundle={bundle}
          facet={facet}
          selected={selected}
          onSelect={selectWork}
          focus={focus}
          pins={mapPins}
          activePin={openPin}
          onOpenPin={showPin}
          onRegion={(x, y, r) => setFocus({ x, y, r, key: Date.now() })}
        />
      </div>

      {openPin !== null && pins[openPin] && (
        <OwnDetail
          bundle={bundle}
          url={pins[openPin].url}
          result={pins[openPin].result}
          onSelect={selectWork}
          onClose={() => setOpenPin(null)}
        />
      )}

      {selected !== null && openPin === null && (
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
