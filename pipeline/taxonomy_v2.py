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
# Academicism folds into Romanticism. Trained apart, the model sent 1,079
# Romanticism works to Academicism and scored 0.214 and 0.376 on the two —
# it cannot separate salon painting from the Romantic mainstream, and art
# history draws the line by institution rather than by appearance.
_group("Romanticism", "Romanticism", "Neo-Romanticism", "Biedermeier",
       "Academicism", "Verism")

# --- the nineteenth century -----------------------------------------------
# Tonalism and Luminism fold in. Kept apart they became a sink: 587 works
# scoring 0.185 while pulling in 923 Realism and 981 Romanticism
# predictions, because balanced weighting rewards a small class for
# guessing and "muted tonal landscape" is not a style so much as a mood
# several styles pass through.
_group("Realism", "Realism", "American Realism", "Analytical Realism",
       "Tonalism", "Luminism")
_group("Naturalism", "Naturalism", "Costumbrismo")
_group("Orientalism", "Orientalism")
_group("Impressionism", "Impressionism", "Intimism")
# Pointillism and Divisionism fold in. Split out, Pointillism had 839
# training works against Impressionism's 10,752, and balanced weighting
# rewarded it for guessing at broken colour: the independent AIC test sent
# 11 of 56 Post-Impressionist works to it, and a late Monet came back
# "Pointillism 0.82". The 17-class taxonomy merged them and I separated
# them in D8 without evidence they should be apart. This is that bill.
_group("Post-Impressionism", "Post-Impressionism", "Cloisonnism",
       "Synthetism", "Pointillism", "Divisionism")
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
# Magic Realism is dropped, not merged. 1,002 works — ample — and it still
# scored 0.097, because it is a category of subject and intent rather than
# of appearance: a Magic Realist painting looks like whatever its painter's
# realism looks like. No amount of data teaches a classifier to see it.
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


# --- when a movement could first have happened -----------------------------
#
# The corpus is public domain and so about 95% pre-1900, while this taxonomy
# reaches well into the 20th century. Softmax must answer, so it answers with
# whatever is nearest — and the result was 20th-century labels on 18th-century
# works. Measured over the built site before this gate existed: every work
# called Surrealism, Social Realism, Regionalism or Photorealism predated the
# movement, 99% of Dada did, and 75% of "Japanese prints, modern" were Edo
# prints a century before Shin-hanga.
#
# A work cannot belong to a movement that had not started. Compared against
# the latest year the work could have been made, so a label is ruled out only
# when even that is too early, and a work with no date is never gated.
#
# Movements whose start is genuinely gradual or contested are left out: this
# is for ruling out the impossible, not for arbitrating the edges.
STYLE_EARLIEST: dict[str, int] = {
    "Japanese prints, modern": 1880,   # Nihonga, the earliest of the three
    "Impressionism": 1860,
    "Naturalism": 1860,
    "Post-Impressionism": 1886,
    "Symbolism": 1880,
    "Art Nouveau": 1890,
    "Fauvism": 1904,
    "Expressionism": 1905,
    "Cubism": 1907,
    "Futurism": 1909,
    "Art Deco": 1910,
    "Dada": 1916,
    "Social Realism": 1920,
    "Surrealism": 1924,
    "Regionalism": 1930,
    "Abstract Expressionism": 1943,
    "Colour Field": 1947,
    "Op Art": 1955,
    "Pop Art": 1955,
    "Minimalism": 1960,
    "Photorealism": 1960,
    "Lyrical Abstraction": 1945,
}
