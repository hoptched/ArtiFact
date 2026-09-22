# ArtiFact

Label artworks by time period, location and art style, and make them browsable.

An offline Python pipeline produces frozen data artifacts; a static site reads
only those artifacts. See [PLAN.md](PLAN.md) for the design and deliverables.

## Setup

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install -e .
.venv/bin/python -m pipeline.hello        # prints the resolved config
```

Heavier stages pull their own dependencies, so the base install stays small:

```bash
python3 -m pip --python .venv/bin/python install -e ".[harvest]"   # D1, D2
python3 -m pip --python .venv/bin/python install -e ".[embed]"     # D3
python3 -m pip --python .venv/bin/python install -e ".[train]"     # D4, D5
```

**Why `--without-pip` and `pip --python`:** this box has no `ensurepip`
(Ubuntu ships it in a separate `python3.12-venv` package), so a normal
`python3 -m venv` produces an env with no working pip. Driving the system pip
at the venv's interpreter sidesteps that and still gives a fully isolated env.
Running `sudo apt install python3.12-venv` once makes the ordinary
`python3 -m venv .venv` + `.venv/bin/pip` route work instead.

## Configuration

Everything tunable lives in [config.yaml](config.yaml) — backbone, corpus cap,
image size, taxonomy scheme, output paths. No stage hardcodes these.

Changing `model.backbone` or `model.image_size` changes the embedding
fingerprint, which marks existing vectors as incomparable rather than letting
a remote D3 run silently drift from the local corpus.
