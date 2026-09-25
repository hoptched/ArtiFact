"""D7: ship what the browser needs to compare an uploaded image.

    .venv/bin/python -m pipeline.export_compare

The site has no backend, so an uploaded picture is encoded in the visitor's
browser by the same CLIP vision tower the pipeline used, and compared
against the corpus locally. That needs two things here that the ordinary
bundle does not carry.

**The embeddings themselves.** The map only ever needed precomputed
neighbour lists, deliberately — 9,130 x 512 floats is 18.7 MB. But an
arbitrary uploaded image has no precomputed anything, so the vectors have
to be present. Quantized to int8 against a single global scale they are
4.7 MB and still agree with the exact top-20 on 96.4% of works, which is
far better than the browser-side encoder itself manages (~70%). The
quantization is not the weak link.

**The style head**, as plain numbers: 17 x 512 coefficients and 17
intercepts, 35 KB. Predicting a style in the browser is then one matrix
multiply.

Both are loaded only when someone actually opens the compare panel.
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from pipeline.config import Config


def main() -> None:
    config = Config.load()
    P, out = config.paths.processed, config.paths.site_data
    vecs = np.load(P / "embeddings.npy").astype(np.float32)
    head = joblib.load(P / "style_head.joblib")
    if head["fingerprint"] != config.model.fingerprint:
        raise SystemExit("the style head was trained in a different vector space")

    # One global scale, not per-row: the browser needs only the ranking, and
    # a shared scale keeps the dot product a plain integer sum.
    scale = float(np.abs(vecs).max())
    q = np.clip(np.round(vecs / scale * 127), -127, 127).astype(np.int8)

    # Row norms of the quantized vectors, so cosine is exact against what
    # was actually stored rather than against the original floats.
    norms = np.linalg.norm(q.astype(np.float32), axis=1).astype(np.float32)

    (out / "embeddings_i8.bin").write_bytes(q.tobytes())
    (out / "embeddings_norms.bin").write_bytes(norms.tobytes())

    model = head["model"]
    (out / "style_head.json").write_text(json.dumps({
        "labels": head["labels"],
        "coef": [[round(float(v), 6) for v in row] for row in model.coef_],
        "intercept": [round(float(v), 6) for v in model.intercept_],
        "fingerprint": head["fingerprint"],
    }, separators=(",", ":")))

    (out / "compare.json").write_text(json.dumps({
        "count": int(len(vecs)),
        "dim": int(vecs.shape[1]),
        "scale": scale,
        "fingerprint": config.model.fingerprint,
        "backbone": config.model.backbone,
        "image_size": config.model.image_size,
        # Measured, not assumed: the browser-side q4f16 encoder ranked the
        # right work first for 24 of 24 corpus images and reproduced about
        # 70% of each one's true top-20.
        "encoder_note": "quantized in-browser encoder; ~70% top-20 fidelity",
    }, indent=2))

    for name in ("embeddings_i8.bin", "embeddings_norms.bin",
                 "style_head.json", "compare.json"):
        print(f"  {name:<24} {(out / name).stat().st_size / 1e6:6.2f} MB")

    # Sanity: dequantized vectors must still rank their own neighbours.
    deq = q.astype(np.float32) / norms[:, None]
    rng = np.random.default_rng(0)
    agree = []
    for i in rng.choice(len(vecs), 200, replace=False):
        true = set(np.argsort(-(vecs @ vecs[i]))[:20].tolist())
        got = set(np.argsort(-(deq @ deq[i]))[:20].tolist())
        agree.append(len(true & got) / 20)
    print(f"\n  int8 vs exact, top-20 agreement: {100 * np.mean(agree):.1f}%")


if __name__ == "__main__":
    main()
