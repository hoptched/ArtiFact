"""The style taxonomy for D8: Artificio/WikiArt's 137 labels into 45 classes.

Three kinds of merge, and one kind of removal.

**Variants into their parent.** Analytical and Synthetic Cubism are phases
of Cubism, not rivals to it; Neo-Rococo is Rococo; Tenebrism is a Baroque
lighting convention. These carry no visual distinction a linear head over
frozen features could hold onto, and splitting them only starves the
parent.

**Near-synonyms into one name.** Art Informel, Tachisme, Lyrical
Abstraction and Spatialism are four national names for the same postwar
gestural abstraction. Keeping them apart would be asking the model to
learn which country a critic was writing in.

**Traditions that belong together.** Shin-hanga and Sosaku-hanga are the
two halves of the 20th-century Japanese print revival; the Persian,
Ottoman, Timurid and Safavid material is one manuscript tradition split by
dynasty. Each is small alone and coherent together.

**Dropped:** labels that name a medium, a century or nothing in
particular, and the long tail below what a class can be learned from.
Their works leave the training set rather than joining a bucket, because
a class called "other" teaches the model to answer "other".
"""

from __future__ import annotations

# Every WikiArt label -> the class it trains as. Anything absent is dropped.
STYLE_MAP: dict[str, str] = {}


def _group(name: str, *labels: str) -> None:
    for label in labels:
        STYLE_MAP[label] = name


# --- before the Renaissance ------------------------------------------------
_group("Medieval", "Byzantine", "Romanesque", "Gothic",
       "International Gothic", "Mosan art", "Neo-Byzantine")
_group("Proto Renaissance", "Proto Renaissance")

# --- Renaissance -----------------------------------------------------------
_group("Early Renaissance", "Early Renaissance")
_group("High Renaissance", "High Renaissance", "Renaissance")
_group("Northern Renaissance", "Northern Renaissance")
_group("Mannerism", "Mannerism (Late Renaissance)")

# --- Baroque to Neoclassical ----------------------------------------------
_group("Baroque", "Baroque", "Tenebrism", "Neo-baroque")
_group("Rococo", "Rococo", "Neo-Rococo")
_group("Neoclassicism", "Neoclassicism", "Classicism")
_group("Academicism", "Academicism", "Verism")
_group("Romanticism", "Romanticism", "Neo-Romanticism", "Biedermeier")

# --- the nineteenth century -----------------------------------------------
_group("Realism", "Realism", "American Realism", "Analytical Realism")
_group("Naturalism", "Naturalism", "Costumbrismo")
_group("Orientalism", "Orientalism")
_group("Impressionism", "Impressionism", "Intimism")
_group("Post-Impressionism", "Post-Impressionism", "Cloisonnism", "Synthetism")
_group("Pointillism", "Pointillism", "Divisionism")
_group("Tonalism", "Tonalism", "Luminism")
_group("Symbolism", "Symbolism")
_group("Art Nouveau", "Art Nouveau (Modern)", "Modernismo", "Japonism")

# --- east Asian traditions -------------------------------------------------
_group("Ukiyo-e", "Ukiyo-e", "Yamato-e")
_group("Japanese prints, modern", "Shin-hanga", "Sōsaku hanga", "Nihonga")
_group("Ink and wash", "Ink and wash painting", "Nanga (Bunjinga)",
       "Gongbi", "Zen", "Joseon Dynasty")

# --- Persian and Ottoman manuscript ---------------------------------------
_group("Persian & Ottoman", "Ottoman Period", "Safavid Period",
       "Timurid Period", "Ilkhanid", "Nas-Taliq")

# --- early modernism -------------------------------------------------------
_group("Fauvism", "Fauvism")
_group("Expressionism", "Expressionism", "Figurative Expressionism",
       "Cubo-Expressionism")
_group("Neo-Expressionism", "Neo-Expressionism")
_group("Cubism", "Cubism", "Analytical Cubism", "Synthetic Cubism",
       "Mechanistic Cubism", "Tubism", "Cubo-Futurism", "Orphism", "Purism")
_group("Futurism", "Futurism", "Rayonism")
_group("Geometric abstraction", "Constructivism", "Suprematism",
       "Neoplasticism", "Concretism", "Neo-Concretism")
_group("Precisionism", "Precisionism")
_group("Art Deco", "Art Deco")
_group("Dada", "Dada", "Neo-Dada", "Mail Art", "Lettrism")
_group("Surrealism", "Surrealism", "Metaphysical art", "Automatic Painting",
       "Fantastic Realism", "Transautomatism")
_group("Magic Realism", "Magic Realism")
_group("Naive art", "Naïve Art (Primitivism)", "Art Brut", "Outsider art",
       "Primitivism", "Native Art", "Kitsch")

# --- mid-century abstraction ----------------------------------------------
_group("Abstract Expressionism", "Abstract Expressionism", "Action painting")
_group("Lyrical Abstraction", "Art Informel", "Tachisme",
       "Lyrical Abstraction", "Spatialism", "Miserablism")
_group("Abstract art", "Abstract Art")
_group("Colour Field", "Color Field Painting", "Post-Painterly Abstraction",
       "Hard Edge Painting", "Synchromism")
_group("Minimalism", "Minimalism", "Post-Minimalism", "Light and Space",
       "Kinetic Art", "New Casualism")

# --- postwar figuration ----------------------------------------------------
_group("Pop Art", "Pop Art", "Nouveau Réalisme", "Street art")
_group("Op Art", "Op Art")
_group("Social Realism", "Social Realism", "Socialist Realism", "Muralism")
_group("Regionalism", "Regionalism")
_group("Photorealism", "Contemporary Realism", "New Realism", "Photorealism",
       "Hyper-Realism", "Poster Art Realism")

# Deliberately unmapped, and so dropped: Conceptual Art, Feminist Art,
# Existential Art, Perceptism, Environmental (Land) Art, Indian Space
# painting, Neo-Figurative Art, New European Painting, Spectralism,
# Cartographic Art. Together about 260 works whose only shared property is
# being hard to place, which is not something to teach a classifier.

STYLE_LABELS: list[str] = sorted(set(STYLE_MAP.values()))
