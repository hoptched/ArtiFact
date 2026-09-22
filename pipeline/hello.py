"""Prints the resolved config. The D0 acceptance check."""

from __future__ import annotations

from pipeline.config import ROOT, Config


def main() -> None:
    config = Config.load()
    config.paths.ensure()

    print("ArtiFact — resolved config\n")
    print(f"  root            {ROOT}")
    print(f"  corpus          {config.corpus.source}, cap {config.corpus.max_works:,} works")
    print(f"  images          {config.corpus.iiif_width}px via IIIF, never stored")
    print(f"  backbone        {config.model.backbone}")
    print(f"  embedding id    {config.model.fingerprint}")
    print(f"  period scheme   {config.taxonomy.period_scheme}")
    print(f"  site bundle     top-{config.site.neighbors_per_work} neighbors, "
          f"<= {config.site.max_bundle_mb} MB")
    print()

    for name, path in [
        ("raw", config.paths.raw),
        ("interim", config.paths.interim),
        ("processed", config.paths.processed),
        ("site data", config.paths.site_data),
    ]:
        contents = list(path.iterdir())
        state = f"{len(contents)} item(s)" if contents else "empty"
        print(f"  {name:<10} {path.relative_to(ROOT)}  ({state})")

    print("\nD0 ok. Next: D1 — harvest the AIC corpus.")


if __name__ == "__main__":
    main()
