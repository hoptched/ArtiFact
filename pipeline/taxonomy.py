"""The three browsing axes: period, country, style.

Kept separate from `normalize.py` so the lookups can be read, argued with
and corrected without touching the code that applies them. PLAN.md calls
D2 the critical path precisely because these tables decide whether the
site is browsable, so they are data, not logic.
"""

from __future__ import annotations

# --- Period ---------------------------------------------------------------
# Fifty-year bins, chosen 2026-09-22 over named eras. The corpus is ~10%
# Chinese, Japanese, Indian, Tibetan and Iranian, and "Baroque" is
# meaningless for a Qing scroll or a Tibetan thangka. Bins are mechanical
# and never wrong, at the cost of a lopsided grid: 1800-1899 holds 45% of
# the corpus.
BIN_WIDTH = 50


def period_bin(date_start: int, date_end: int, max_span: int) -> int | None:
    """The bin a work belongs to, or None if its date is too vague.

    A work is placed by the midpoint of its range, so a piece dated
    1790-1810 lands in 1800-1849 rather than being arbitrarily assigned to
    whichever endpoint came first.
    """
    if date_start is None or date_end is None:
        return None
    if date_end < date_start:
        date_start, date_end = date_end, date_start
    if date_end - date_start > max_span:
        return None
    midpoint = (date_start + date_end) // 2
    return (midpoint // BIN_WIDTH) * BIN_WIDTH


def period_bins(date_start: int, date_end: int, max_span: int) -> list[int]:
    """Every bin the work's date range overlaps.

    A work dated "18th century" (1700-1799) genuinely belongs to both
    1700-1749 and 1750-1799, and the museum is telling us it does not know
    which. Assigning it to one by midpoint invented precision that was not
    there and skewed the axis badly: 100% of whole-century records landed
    in the first half, leaving 1700-1749 at 37.6% guesswork while
    1750-1799 sat at 4.5%. Returning both is the honest answer.
    """
    if date_start is None or date_end is None:
        return []
    if date_end < date_start:
        date_start, date_end = date_end, date_start
    if date_end - date_start > max_span:
        return []
    first = (date_start // BIN_WIDTH) * BIN_WIDTH
    last = (date_end // BIN_WIDTH) * BIN_WIDTH
    return list(range(first, last + BIN_WIDTH, BIN_WIDTH))


# How much to trust a date. The corpus splits roughly 47/28/25 across
# these, so the site must be able to tell them apart.
PRECISION_EXACT_MAX = 10
PRECISION_LOOSE_MAX = 50


def date_precision(date_start: int, date_end: int) -> str:
    span = abs(date_end - date_start)
    if span <= PRECISION_EXACT_MAX:
        return "exact"
    if span <= PRECISION_LOOSE_MAX:
        return "loose"
    return "vague"


def period_label(bin_start: int) -> str:
    if bin_start < 0:
        return f"{abs(bin_start + BIN_WIDTH - 1)}–{abs(bin_start)} BCE"
    return f"{bin_start}–{bin_start + BIN_WIDTH - 1}"


# --- Country --------------------------------------------------------------
# Every one of the 129 distinct place_of_origin values in the painting-like
# corpus is mapped here by hand; nothing falls through to Unknown by
# accident. Resolved to present-day countries, per the 2026-09-22 decision.
#
# Six names are ambiguous between England and New England. All six were
# checked against the actual records and all six are American: "Bath" is
# The Schooner Jane of Bath, Maine; "Lancaster" is Jacob Eichholtz, the
# Lancaster, Pennsylvania portraitist; "Greenwich" is Twachtman in
# Connecticut; "Ipswich" is Arthur Wesley Dow; "Gloucester" is Childe
# Hassam's New England Headlands; "Boston" is Hunt and Smibert.
PLACE_TO_COUNTRY: dict[str, str] = {
    # France
    "France": "France", "Paris": "France", "Lyon": "France",
    "Brittany": "France", "Giverny": "France", "Trouville": "France",
    "Saint-Rémy-de-Provence": "France",
    # Italy
    "Italy": "Italy", "Venice": "Italy", "Florence": "Italy",
    "Northern Italy": "Italy", "Rome": "Italy", "Genoa": "Italy",
    "Naples": "Italy", "Bologna": "Italy", "Central Italy": "Italy",
    "Perugia": "Italy", "Frascati": "Italy", "Umbria": "Italy",
    "Feltre": "Italy", "Veneto": "Italy", "Urbino": "Italy", "Siena": "Italy",
    # United Kingdom (Ireland is its own modern country)
    "England": "United Kingdom", "United Kingdom": "United Kingdom",
    "Scotland": "United Kingdom", "Great Britain": "United Kingdom",
    "London": "United Kingdom", "Wales": "United Kingdom",
    "Ireland": "Ireland",
    # Low Countries, split as modern states
    "Holland": "Netherlands", "Netherlands": "Netherlands",
    "Northern Netherlands": "Netherlands", "Dordrecht": "Netherlands",
    "Delft": "Netherlands",
    "Flanders": "Belgium", "Belgium": "Belgium", "Bruges": "Belgium",
    # German-speaking
    "Germany": "Germany", "Munich": "Germany", "Berlin": "Germany",
    "Southern Germany": "Germany", "Bavaria": "Germany", "Rhine": "Germany",
    "Austria": "Austria", "Switzerland": "Switzerland",
    # United States
    "United States": "United States", "Philadelphia": "United States",
    "New York": "United States", "New York City": "United States",
    "Lancaster": "United States", "Roxbury": "United States",
    "Massachusetts": "United States", "Boston": "United States",
    "New Hampshire": "United States", "Newport": "United States",
    "Long Island": "United States", "New England": "United States",
    "Florida": "United States", "Saint Louis": "United States",
    "Baltimore": "United States", "Niagara Falls": "United States",
    "Prouts Neck": "United States", "Nantucket Island": "United States",
    "York Harbor": "United States", "Gloucester": "United States",
    "Connecticut": "United States", "Bennington": "United States",
    "Virginia": "United States", "New Jersey": "United States",
    "Wyoming": "United States", "Pennsylvania": "United States",
    "Montana": "United States", "Greenwich": "United States",
    "Ipswich": "United States", "Bath": "United States",
    # Iberia
    "Spain": "Spain", "Seville": "Spain", "Catalonia": "Spain",
    # Nordics and central Europe
    "Sweden": "Sweden", "Denmark": "Denmark", "Norway": "Norway",
    "Finland": "Finland", "Hungary": "Hungary", "Russia": "Russia",
    "Bohemia": "Czech Republic", "Czech Republic": "Czech Republic",
    "Greece": "Greece", "Kríti": "Greece", "Corfu": "Greece",
    # East Asia
    "China": "China", "Guangdong": "China", "Northern China": "China",
    "Japan": "Japan", "Ueno": "Japan", "Korea": "Korea",
    "Mongolia": "Mongolia",
    # Tibet is deliberately not folded into China. The corpus holds 35
    # Tibetan works, all religious painting, and the art-historical unit is
    # what a browsing filter should show.
    "Tibet": "Tibet", "Nepal": "Nepal",
    # South and West Asia
    "India": "India", "Rajasthan": "India", "Murshidabad": "India",
    "Andhra Pradesh": "India", "Jaipur": "India", "Bundi": "India",
    "Jodhpur": "India", "Deccan": "India", "Kota": "India",
    "Lucknow": "India", "Avadh": "India",
    "Lahore": "Pakistan", "Herat": "Afghanistan",
    "Iran": "Iran", "Shiraz": "Iran", "Isfahan": "Iran",
    "Bukhara": "Uzbekistan", "Turkey": "Turkey",
    # Africa, Oceania, Americas
    "Egypt": "Egypt", "Ethiopia": "Ethiopia", "Australia": "Australia",
    "Teotihuacán": "Mexico", "Peruvian North Coast": "Peru",
}

# Real values that name something larger than a country. Kept distinct from
# Unknown: "somewhere in Europe" is information, and PLAN.md says the site
# shows Unknown rather than hiding it, so it should not swallow these.
VAGUE_PLACES: dict[str, str] = {
    "Europe": "Europe (unspecified)",
    "Western Europe": "Europe (unspecified)",
    "Northern Europe": "Europe (unspecified)",
    "Middle East": "Middle East (unspecified)",
}

UNKNOWN_PLACES = {"Unknown Place", "Unknown", "", None}


def country_for(place: str | None, unknown_label: str) -> str:
    if place in UNKNOWN_PLACES:
        return unknown_label
    if place in VAGUE_PLACES:
        return VAGUE_PLACES[place]
    return PLACE_TO_COUNTRY.get(place, unknown_label)


# --- Style ----------------------------------------------------------------
# WikiArt ships 27 style classes (verified against the HuggingFace dataset
# info for huggan/wikiart, not recalled). This maps them onto the label set
# the head is trained on.
#
# Seven postwar classes are dropped rather than merged. The AIC corpus is
# public domain and therefore ~95% pre-1900; a head that has never seen Pop
# Art cannot predict Pop Art for an 1870 landscape. Dropping beats merging
# because these movements have no plausible instance in the target corpus.
WIKIART_STYLE_MAP: dict[str, str] = {
    "Early_Renaissance": "Early Renaissance",
    "High_Renaissance": "High Renaissance",
    "Northern_Renaissance": "Northern Renaissance",
    "Mannerism_Late_Renaissance": "Mannerism",
    "Baroque": "Baroque",
    "Rococo": "Rococo",
    "Romanticism": "Romanticism",
    "Realism": "Realism",
    "Impressionism": "Impressionism",
    "Post_Impressionism": "Post-Impressionism",
    "Pointillism": "Post-Impressionism",      # a technique within it
    "Symbolism": "Symbolism",
    "Art_Nouveau": "Art Nouveau",
    "Expressionism": "Expressionism",
    "Fauvism": "Fauvism",
    "Cubism": "Cubism",
    "Analytical_Cubism": "Cubism",            # phases, not separate movements
    "Synthetic_Cubism": "Cubism",
    "Naive_Art_Primitivism": "Naive Art",
    "Ukiyo_e": "Ukiyo-e",
}

DROPPED_WIKIART_STYLES = {
    "Abstract_Expressionism", "Action_painting", "Color_Field_Painting",
    "Contemporary_Realism", "Minimalism", "New_Realism", "Pop_Art",
}

STYLE_LABELS: list[str] = sorted(set(WIKIART_STYLE_MAP.values()))


# --- Where the style taxonomy applies -------------------------------------
# The 17 labels are 16 European movements plus Ukiyo-e. They describe
# Western painting c.1400-1900 and, for Ukiyo-e, the Japanese woodblock
# tradition. They describe nothing else.
#
# Measured on the AIC corpus 2026-09-23: Chinese works were predicted
# Ukiyo-e 161 times out of 193, Tibetan thangkas 12 of 19, at a mean
# confidence of 0.73. Ukiyo-e is simply the only non-European class
# available, so everything East Asian falls into it. A confidence
# threshold cannot catch this — the model is confidently wrong, not
# uncertain — so the domain has to be declared rather than inferred.
#
# Japan is deliberately in: 144 of 162 Japanese works were predicted
# Ukiyo-e at 0.814 confidence, which is correct.
STYLE_DOMAIN_COUNTRIES: frozenset[str] = frozenset({
    # Western Europe and its settler traditions
    "France", "Italy", "United Kingdom", "Ireland", "Netherlands", "Belgium",
    "Germany", "Austria", "Switzerland", "Spain", "Greece",
    "Sweden", "Denmark", "Norway", "Finland", "Hungary", "Czech Republic",
    "Russia", "United States", "Australia",
    "Europe (unspecified)",
    # The one non-European tradition the taxonomy actually covers
    "Japan",
})

OUTSIDE_TAXONOMY = "Outside the taxonomy"


def style_applies(country: str) -> bool:
    """Whether a style prediction for this work can mean anything.

    A country the taxonomy does not cover gets no style rather than a
    confident falsehood. Period and country are observed facts and remain
    browsable either way.
    """
    return country in STYLE_DOMAIN_COUNTRIES
