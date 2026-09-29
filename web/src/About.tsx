import { useEffect, useState } from "react";

/**
 * What this is and where it came from.
 *
 * It exists mostly for one sentence: the style labels are predictions,
 * made by a model trained on someone else's paintings, and nothing else
 * in the interface says whose.
 */
export function About({ works }: { works: number }) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open]);

  return (
    <>
      <button className="about-link" onClick={() => setOpen(true)}>About</button>

      {open && (
        // Fixed, so it escapes the sidebar's overflow: hidden rather than
        // being clipped to the column the link sits in.
        <div className="about-backdrop" onClick={() => setOpen(false)}>
          <div className="about" onClick={(e) => e.stopPropagation()}
               role="dialog" aria-label="About Minerva">
            <button className="close" onClick={() => setOpen(false)}
                    aria-label="Close">×</button>
            <h2>Minerva</h2>

            <p>
              {works.toLocaleString()} works in the public domain from the Art
              Institute of Chicago, laid out so that works which look alike sit
              near each other.
            </p>

            <h3>The map</h3>
            <p>
              Every work is turned into a vector by CLIP ViT-L/14, and that
              vector decides where it sits. Under <b>Similarity</b> it is the
              only thing deciding: there are no regions, and being next to
              something means the two look alike. Under <b>Period</b>,{" "}
              <b>Place</b> and <b>Style</b> a work's group picks its region and
              the vector places it inside, so a work near the edge of its
              region resembles what lies over the border.
            </p>

            <h3>Where the labels come from</h3>
            <p>
              Date and place come from the museum's own catalogue. Style comes
              from a model. The museum's style field holds mostly centuries and
              cultures rather than movements, and fewer than a fifth of works
              carry one, so the 42 style labels here are predicted by a linear
              classifier trained on{" "}
              <a href="https://huggingface.co/datasets/Artificio/WikiArt"
                 target="_blank" rel="noreferrer">Artificio/WikiArt</a>{" "}
              and applied to this collection. Scored with every painter in the
              test fold unseen during training, it gets 49% right. Treat a
              style as a guess with a number attached, and the confidence
              beside it as that guess's own estimate. The map pools the
              smallest labels, so it shows 23 regions rather than 42.
            </p>
            <p>
              A movement can only be predicted for a work made after it began.
              Anything dated earlier falls through to the model's next choice,
              which lifted about 2,400 labels the calendar ruled out.
            </p>
            <p>
              Works from traditions the classifier covers poorly, mostly the
              Indian subcontinent, the Himalaya and Mesoamerica, are marked{" "}
              <i>outside the taxonomy</i> and held apart on the map, rather
              than given the nearest label that fits badly.
            </p>

            <h3>Your own picture</h3>
            <p>
              Encoded in your browser by the same model and compared against
              all {works.toLocaleString()} vectors. It stays on your machine.
            </p>

            {/* The heading goes, its gap stays: the credits still want
                separating from the section above them. */}
            <p className="muted small credits">
              Images and metadata from the{" "}
              <a href="https://api.artic.edu/docs/" target="_blank"
                 rel="noreferrer">Art Institute of Chicago</a>, used under CC0.
              Style training data from WikiArt, used for training only.
            </p>
            <p className="signoff">
              made by{" "}
              <a href="https://github.com/hoptched/" target="_blank"
                 rel="noreferrer">joshua soo</a> 🐻
            </p>
          </div>
        </div>
      )}
    </>
  );
}
