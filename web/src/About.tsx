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
              {works.toLocaleString()} public-domain works from the Art
              Institute of Chicago, laid out so that works which look alike
              sit near each other.
            </p>

            <h3>The map</h3>
            <p>
              Every work is turned into a vector by CLIP ViT-L/14, and that
              vector decides where it sits. Under <b>Similarity</b> it is the
              only thing deciding: there are no regions, and being next to
              something means the two look alike. Under <b>Period</b>,{" "}
              <b>Place</b> and <b>Style</b> a work's group picks its region
              and the vector places it inside — so a work near the edge of
              its region resembles what is over the border.
            </p>

            <h3>Where the labels come from</h3>
            <p>
              Date and place are the museum's own. <b>Style is not.</b> The
              museum's style field is mostly centuries and cultures rather
              than movements, and fewer than a fifth of works carry one, so
              the 42 style labels here are predicted by a linear classifier
              trained on{" "}
              <a href="https://huggingface.co/datasets/Artificio/WikiArt"
                 target="_blank" rel="noreferrer">Artificio/WikiArt</a>{" "}
              and applied to this collection. Scored with every painter in
              the test fold unseen during training, it is right 49% of the
              time across those 42 labels. Treat a style as a guess with a
              number attached, and the confidence beside it as that guess's
              own estimate. The map pools the smallest labels, so it shows
              28 regions rather than 42.
            </p>
            <p>
              Works the classifier has no label for are marked{" "}
              <i>outside the taxonomy</i> rather than given the nearest
              label that fits badly.
            </p>

            <h3>Your own picture</h3>
            <p>
              Encoded in your browser by the same model, compared against
              all {works.toLocaleString()} vectors, and never uploaded
              anywhere.
            </p>

            <h3>Credits</h3>
            <p className="muted small">
              Images and metadata from the{" "}
              <a href="https://api.artic.edu/docs/" target="_blank"
                 rel="noreferrer">Art Institute of Chicago</a>, used under
              CC0. Style training data from WikiArt, used for training only
              and not redistributed here.
            </p>
          </div>
        </div>
      )}
    </>
  );
}
