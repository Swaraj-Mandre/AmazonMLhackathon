"""Text normalisation for business names and addresses.

Everything here is a deterministic, self-contained transformation of the strings
in the provided data. No lookups, no external resources, the fair-play rules
forbid them, and the abbreviation and transliteration tables below are ordinary
code, not data pulled from anywhere.

Three jobs, in order of how much score they are worth:

1. **Transliteration.** Source 1 is always Latin script, but roughly 10% of the
   Indian links in the training data point at a Devanagari record. Character
   n-grams score exactly zero across scripts, so those matches are invisible to
   blocking unless both sides are brought into one alphabet first.
2. **Abbreviation folding.** ``Pvt Ltd`` and ``Private Limited`` have to collapse
   to the same token sequence, likewise ``Rd``/``Road`` and ``&``/``and``.
3. **Structural extraction.** Street numbers and postcodes are far more
   discriminative than any word in an address, so they are pulled out into their
   own field rather than left to drown in the token soup.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Devanagari -> Latin
# --------------------------------------------------------------------------
# A plain character map in the ITRANS/ISO-15919 spirit. It is not a scholarly
# transliteration and does not need to be: the only requirement is that the same
# Devanagari word always produces the same Latin string, and that the result
# looks close enough to the English spelling for character n-grams to overlap.
# Matras are folded to their bare vowel and the inherent 'a' is dropped, which
# costs some fidelity but keeps names short and comparable.

_DEVANAGARI_MAP = {
    # independent vowels
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    # candra/short vowels, common in transliterated loanwords ("property")
    "ऑ": "o", "ऒ": "o", "ऍ": "e", "ऎ": "e",
    # vowel signs (matras)
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॊ": "o", "ॅ": "e", "ॆ": "e",
    # consonants
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # nukta forms, common in loanwords and therefore in business names
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    # nasalisation and aspiration marks
    "ं": "n", "ँ": "n", "ः": "h", "ऽ": "",
    # virama suppresses the inherent vowel; we drop vowels anyway
    "्": "",
    # digits
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def has_devanagari(text: str) -> bool:
    """True when the string contains any Devanagari codepoint."""
    return bool(_DEVANAGARI_RE.search(text))


def transliterate(text: str) -> str:
    """Map Devanagari characters to a Latin approximation, leaving the rest alone.

    Two-codepoint nukta sequences (क + ़) are handled by composing first, so the
    combined forms in the table are reachable whichever way the source encoded
    them.
    """
    if not has_devanagari(text):
        return text
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in _DEVANAGARI_MAP:
            out.append(_DEVANAGARI_MAP[ch])
        elif "ऀ" <= ch <= "ॿ":
            out.append("")  # unmapped Devanagari mark: drop rather than keep noise
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# Abbreviation tables
# --------------------------------------------------------------------------
# Both sides of each pair collapse to one canonical token. Longer keys are
# applied first so "private limited" is not half-rewritten by "limited".

_LEGAL_FORMS = {
    "corporation": "corp", "incorporated": "inc", "company": "co",
    "limited": "ltd", "private": "pvt", "public": "pub",
    "llp": "llp", "llc": "llc", "plc": "plc",
    "pvtltd": "pvt ltd", "prvt": "pvt", "priv": "pvt",
    "brothers": "bros", "brother": "bros",
    "associates": "assoc", "associate": "assoc", "association": "assoc",
    "international": "intl", "industries": "inds", "industry": "inds",
    "enterprises": "ent", "enterprise": "ent",
    "manufacturing": "mfg", "manufacturers": "mfg", "manufacturer": "mfg",
    "services": "svc", "service": "svc", "solutions": "sol", "solution": "sol",
    "technologies": "tech", "technology": "tech",
    "trading": "trdg", "traders": "trdg", "trader": "trdg",
}

_ADDRESS_FORMS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "court": "ct", "place": "pl",
    "square": "sq", "highway": "hwy", "parkway": "pkwy", "circle": "cir",
    "terrace": "ter", "trail": "trl", "way": "way", "route": "rt",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "apt", "suite": "ste", "floor": "flr", "building": "bldg",
    "block": "blk", "sector": "sec", "phase": "ph", "plot": "plot",
    "opposite": "opp", "near": "near", "behind": "behind",
    "nagar": "nagar", "colony": "colony", "marg": "marg", "chowk": "chowk",
    "saint": "st", "mount": "mt", "fort": "ft",
}

# Legal suffixes carry almost no discriminative power, nearly every Indian
# business ends in "pvt ltd", so a separate "core name" strips them entirely.
_STRIPPABLE_SUFFIXES = {
    "corp", "inc", "co", "ltd", "pvt", "pub", "llp", "llc", "plc",
    "the", "and", "&",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_LEADING_JUNK_RE = re.compile(r"^[\W_]+|[\W_]+$", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"\d+")


def _fold(text: str) -> str:
    """Lowercase, strip accents, and reduce to alphanumerics plus single spaces.

    Accent stripping is what makes the French records comparable, ``Président``
    and ``President`` must not be two different tokens, and it is script-safe
    because the decomposition only removes combining marks.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    # "&" would otherwise be erased by punctuation stripping, leaving
    # "Smith & Sons" and "Smith and Sons" with different token counts.
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _apply_table(tokens: list[str], table: dict[str, str]) -> list[str]:
    """Rewrite each token through an abbreviation table, dropping empties."""
    out = []
    for tok in tokens:
        mapped = table.get(tok, tok)
        out.extend(t for t in mapped.split() if t)
    return out


def normalize_name(raw: str) -> str:
    """Canonical form of a business name.

    Transliterates, folds case and accents, then rewrites legal forms so that
    ``Sharma Medical Stores Private Limited`` and ``Sharma Medical Store Pvt Ltd``
    converge. Leading junk such as the ``--`` and ``<<`` prefixes that appear in
    the raw data is removed before anything else.
    """
    if not raw:
        return ""
    text = transliterate(raw)
    text = _LEADING_JUNK_RE.sub("", text)
    tokens = _fold(text).split()
    return " ".join(_apply_table(tokens, _LEGAL_FORMS))


def core_name(raw: str) -> str:
    """The name with generic legal and filler tokens removed.

    ``pvt``/``ltd``/``inc`` appear on a large share of records, so leaving them
    in inflates similarity between unrelated businesses. The stripped form is
    the stronger blocking key; the full form is kept as a feature.
    """
    tokens = [t for t in normalize_name(raw).split() if t not in _STRIPPABLE_SUFFIXES]
    return " ".join(tokens) or normalize_name(raw)


def normalize_address(raw: str) -> str:
    """Canonical form of an address, with directional and street-type words folded."""
    if not raw:
        return ""
    text = transliterate(raw)
    tokens = _fold(text).split()
    return " ".join(_apply_table(tokens, _ADDRESS_FORMS))


def address_numbers(raw: str) -> list[str]:
    """Every digit run in an address, longest first.

    House numbers and postcodes are the most discriminative part of an address:
    two records sharing "411001" and "12" are far more likely to be the same
    business than two sharing the word "road". Kept as an ordered list so a
    feature can ask about the longest number (usually the postcode) separately.
    """
    if not raw:
        return []
    nums = _NUMBER_RE.findall(transliterate(raw))
    return sorted(set(nums), key=lambda n: (-len(n), n))


def name_tokens(raw: str) -> frozenset[str]:
    """Token set of the core name, for Jaccard-style comparisons."""
    return frozenset(core_name(raw).split())


def _self_test() -> None:
    assert normalize_name("Sharma Medical Stores Private Limited") == \
           normalize_name("Sharma Medical Stores Pvt. Ltd."), normalize_name("Sharma Medical Stores Pvt. Ltd.")
    assert core_name("B+ Retail Inc") == "b retail"
    assert normalize_name("-- Holloway Peak Inc Seafood").startswith("holloway")
    assert normalize_name("<< Team Ecole") == "team ecole"

    # Accent folding is what makes the French records comparable.
    assert normalize_address("Boulevard du Président Franklin Roosevelt") == \
           normalize_address("Boulevard du President Franklin Roosevelt")

    # Street-type and directional folding.
    assert normalize_address("1795 Westchester Drive") == normalize_address("1795 Westchester Dr")
    assert normalize_address("105 ELM ST, MORGANTON, NC") == normalize_address("105 Elm Street, Morganton, NC")

    # "&" and "and" converge.
    assert normalize_name("Smith & Sons") == normalize_name("Smith and Sons")

    # Transliteration: Devanagari must reach the Latin alphabet, and must be
    # stable across calls.
    deva = "आदित्य प्रॉपर्टीज"
    lat = transliterate(deva)
    assert not has_devanagari(lat), lat
    assert lat == transliterate(deva)
    assert normalize_name(deva) == normalize_name(deva)
    assert has_devanagari(deva) and not has_devanagari("Prime Money")

    # Numbers come out longest-first so the postcode leads.
    assert address_numbers("G-3/571, GULMOHAR COLONY, BHOPAL, 462001") == ["462001", "571", "3"]
    assert address_numbers("") == []

    assert name_tokens("Prime Money Pvt Ltd") == frozenset({"prime", "money"})
    assert normalize_name("") == "" and normalize_address("") == ""

    print(f"normalize.py: all checks passed  (sample transliteration: {lat!r})")


if __name__ == "__main__":
    _self_test()
