"""Typed access to config.yaml.

Every stage loads its settings from here rather than hardcoding them. The
embedding fingerprint exists so a D3 run on remote compute and the local
corpus can be checked for drift instead of silently disagreeing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class CorpusConfig:
    source: str
    max_works: int
    iiif_width: int
    max_span_years: int
    painting_like_only: bool
    painting_like_types: list[str]
    print_cap_per_artist: int
    print_dedupe_title: bool


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    image_size: int
    batch_size: int
    embed_dim: int

    @property
    def fingerprint(self) -> str:
        """Identifies an embedding run. Vectors built under a different
        fingerprint are not comparable and must not be mixed."""
        return hashlib.sha256(
            f"{self.backbone}@{self.image_size}".encode()
        ).hexdigest()[:12]


@dataclass(frozen=True)
class TaxonomyConfig:
    period_scheme: str
    unknown_label: str


@dataclass(frozen=True)
class SiteConfig:
    neighbors_per_work: int
    max_bundle_mb: int


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    site_data: Path

    def ensure(self) -> None:
        for path in (self.raw, self.interim, self.processed, self.site_data):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    corpus: CorpusConfig
    model: ModelConfig
    taxonomy: TaxonomyConfig
    site: SiteConfig
    paths: Paths

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or DEFAULT_CONFIG
        if not path.exists():
            raise FileNotFoundError(f"no config at {path}")
        raw = yaml.safe_load(path.read_text())

        valid_schemes = {"century_bins", "fifty_year_bins", "named_eras"}
        scheme = raw["taxonomy"]["period_scheme"]
        if scheme not in valid_schemes:
            raise ValueError(
                f"period_scheme {scheme!r} must be one of {sorted(valid_schemes)}"
            )

        return cls(
            corpus=CorpusConfig(**raw["corpus"]),
            model=ModelConfig(**raw["model"]),
            taxonomy=TaxonomyConfig(**raw["taxonomy"]),
            site=SiteConfig(**raw["site"]),
            paths=Paths(**{k: ROOT / v for k, v in raw["paths"].items()}),
        )
