"""Build the Phase 0 and Phase 1 notes as a plain black-and-white .docx.

Deliberately no colour anywhere: the built-in Heading styles in Word are blue,
so headings here are ordinary paragraphs with a bold, slightly larger run. Tables
use Table Grid, which is black lines and no shading.

Run:  python docs/build_phase_notes_doc.py
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

OUT = Path(__file__).resolve().parent / "Phase0_Phase1_Notes.docx"

BLACK = RGBColor(0, 0, 0)
BODY_FONT = "Calibri"
MONO_FONT = "Consolas"


def setup(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(11)
    style.font.color.rgb = BLACK
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)


def heading(doc: Document, text: str, size: int = 14, space_before: int = 14):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    return p


def body(doc: Document, text: str, bold_prefix: str | None = None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.color.rgb = BLACK
    r = p.add_run(text)
    r.font.color.rgb = BLACK
    return p


def bullet(doc: Document, text: str, bold_prefix: str | None = None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.color.rgb = BLACK
    r = p.add_run(text)
    r.font.color.rgb = BLACK
    return p


def numbered(doc: Document, text: str, bold_prefix: str | None = None):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.color.rgb = BLACK
    r = p.add_run(text)
    r.font.color.rgb = BLACK
    return p


def mono(doc: Document, lines: list[str]):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    for i, line in enumerate(lines):
        run = p.add_run(line)
        run.font.name = MONO_FONT
        run.font.size = Pt(9.5)
        run.font.color.rgb = BLACK
        if i < len(lines) - 1:
            run.add_break()
    return p


def table(doc: Document, header: list[str], rows: list[list[str]],
          widths: list[float] | None = None, bold_rows: set[int] | None = None):
    bold_rows = bold_rows or set()
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = True

    for cell, text in zip(t.rows[0].cells, header):
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = BLACK

    for idx, row in enumerate(rows):
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(text)
            run.font.size = Pt(10)
            run.font.color.rgb = BLACK
            if idx in bold_rows:
                run.bold = True

    if widths:
        for row in t.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Inches(w)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def build() -> Path:
    doc = Document()
    setup(doc)

    # Title block
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run("Phase 0 and Phase 1: what we actually did")
    run.bold = True
    run.font.size = Pt(17)
    run.font.color.rgb = BLACK

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run("Amazon ML Challenge 2026  |  Team sahara  |  25 September 2026")
    run.font.size = Pt(10)
    run.italic = True
    run.font.color.rgb = BLACK

    body(doc, "Every number in this document came from a run on the real data. "
              "Nothing here is an estimate. Where a setting has a name in the "
              "code, that name is given so you can find it.")

    # ---------------------------------------------------------------- the task
    heading(doc, "The task in one paragraph", space_before=10)
    body(doc, "Amazon gave us three lists of businesses. Source 1 is the master "
              "list with 1.73 million rows. Source 2 and Source 3 are the same "
              "businesses recorded by other systems, 4.89 million and 5.08 "
              "million rows, spelled differently. Each row has only an ID, a "
              "business name, an address and a country. For every Source 1 row we "
              "have to say which Source 2 and Source 3 rows are the same real "
              "business. The score is F0.5, which means precision counts double, "
              "so a wrong guess hurts us twice as much as a missed one.")

    # -------------------------------------------------------------------- P0
    heading(doc, "Phase 0: getting the data into a usable shape")
    body(doc, "Nothing clever in this phase. It exists so that Phase 1 does not "
              "run out of memory, and so that we are working from measured facts "
              "instead of guesses.")

    heading(doc, "1. Converted the TSV files to Parquet", size=12, space_before=10)
    body(doc, "Read in chunks of 200,000 rows (CHUNK_ROWS), compressed with zstd, "
              "written to data/interim/. Six cache files. Parquet lets us load "
              "one column without reading the others, and load one country "
              "without reading the rest of the file.")

    heading(doc, "2. Cleaned every name and address once, up front", size=12,
            space_before=10)
    body(doc, "The cleaned versions are stored in the cache, so a run never has "
              "to clean anything again. Four columns were added to every row:")
    table(doc,
          ["Column", "What it holds"],
          [["core_name", "the cleaned name, and the only thing Phase 1 compares"],
           ["norm_addr", "cleaned address, kept for later phases"],
           ["addr_nums", "digit runs pulled out, longest first, so the postcode leads"],
           ["was_transliterated", "flag, was this row originally in Devanagari"]],
          widths=[1.7, 4.8])

    heading(doc, "3. What the cleaning does, in order", size=12, space_before=6)
    bullet(doc, "Devanagari to Latin using our own hand written table. "
                "The Hindi form of Ram Marketing Private Limited comes out as "
                "\u201cram marketing praivet limited\u201d.")
    bullet(doc, "Unicode NFKD accent folding. Needed for the French rows.")
    bullet(doc, "The ampersand becomes the word \u201cand\u201d. This has to happen "
                "before punctuation is stripped, otherwise the ampersand is simply "
                "deleted. We got that wrong the first time and the self test caught it.")
    bullet(doc, "Lowercase, strip punctuation, collapse repeated spaces.")
    bullet(doc, "Fold legal forms, so pvt, pvt. and private all become one token.")
    bullet(doc, "core_name then removes the words pvt, ltd, inc, llc and "
                "“the” entirely, because they identify nothing.")

    heading(doc, "4. Measured the data instead of assuming", size=12, space_before=10)
    body(doc, "Five findings from full passes over the training data. The first "
              "two are the reason the rest of the pipeline is shaped the way it is.")
    table(doc,
          ["What we checked", "Result"],
          [["Matches that cross a country label", "0 out of 1,038,755 links checked"],
           ["Source 2 or 3 rows owned by more than one Source 1 row",
            "0 out of 7,638,365 links"],
           ["Rows with no match at all, called singletons", "5.58 percent"],
           ["Mean matches per Source 1 row", "3.46, highest seen was 11"],
           ["Indian matches that need transliteration", "about 10 percent"]],
          widths=[4.0, 2.5])

    heading(doc, "5. Wrote the scoring function and checked it", size=12,
            space_before=6)
    body(doc, "metric.py is the only place F0.5 is calculated anywhere in the "
              "project. We ran it against the worked example printed in the "
              "problem statement and got 0.714286, which matches their answer "
              "exactly. Without this we would have no idea whether our own "
              "numbers mean anything.")

    heading(doc, "6. Split the training data by entity, not by pair", size=12,
            space_before=10)
    body(doc, "15 percent held out (VALIDATION_FRACTION), random seed 20260925. "
              "Splitting by pair instead would put the same business on both "
              "sides of the split, and every score after that would be inflated "
              "and useless.")

    doc.add_page_break()

    # -------------------------------------------------------------------- P1
    heading(doc, "Phase 1: the baseline", space_before=0)
    body(doc, "One signal, one threshold, no machine learning. The point of a "
              "baseline is to give every later idea something honest to beat.")

    heading(doc, "The exact parameters", size=12, space_before=10)
    body(doc, "Blocking key is the country field. Records are only ever compared "
              "inside the same country, which gives six blocks, two countries "
              "times three sources. The text vectorizer settings, from "
              "VECTORIZER_KWARGS in blocking.py:")
    mono(doc, [
        "analyzer      = \"word\"",
        "token_pattern = r\"\\S+\"        # split on whitespace, nothing else",
        "lowercase     = False         # already lowercased in phase 0",
        "sublinear_tf  = True          # 1+log(tf), stops repeats dominating",
        "dtype         = np.float32    # half the memory of float64",
        "min_df        = 1",
        "max_df        = 0.2           # ignore words in over 20% of rows",
    ])
    body(doc, "Fitted on a random sample of 1,000,000 names per block, seed 0. "
              "IDF is a smooth statistic that a million rows estimates perfectly "
              "well. Fitting on all 12 million would cost several gigabytes of "
              "vocabulary for no measurable gain.")
    body(doc, "Candidate generation and acceptance:")
    mono(doc, [
        "top_k           = 5        # TOP_K_PER_CANDIDATE",
        "min_score       = 0.60     # BLOCK_MIN, the blocking drop line",
        "CANDIDATE_CHUNK = 6_000    # rows held in memory at a time",
        "accept          = 0.92     # the final decision line",
        "exclusivity     = off",
    ])

    heading(doc, "What we compare, exactly", size=12, space_before=10)
    body(doc, "Only one thing. core_name, turned into a TF-IDF weighted bag of "
              "whitespace separated words, compared by cosine similarity.")
    body(doc, "We do not use the address in Phase 1. Not because it is useless, "
              "but because it needs its own feature work and we wanted a clean "
              "single signal baseline first.")
    body(doc, "Here is why this shrinks the problem so much. Take two names:")
    mono(doc, [
        "ram marketing     ->  tokens {ram, marketing}",
        "ram mktg          ->  tokens {ram, mktg}",
    ])
    body(doc, "They share the token ram, which is rare and therefore carries a "
              "high IDF weight, so the cosine is decent and the pair survives. "
              "Now take a different pair:")
    mono(doc, [
        "ram marketing ltd   ->  ltd is stripped by core_name",
        "shyam trading ltd   ->  ltd is stripped by core_name",
    ])
    body(doc, "After stripping they share nothing at all, so the cosine is "
              "exactly zero, and sparse matrix maths never even calculates it. "
              "That single fact is how 17.3 trillion possible pairs becomes 36 "
              "million actual comparisons.")

    heading(doc, "What we drop, and on what basis", size=12, space_before=10)
    body(doc, "Four separate drops, applied in this order:")
    table(doc,
          ["Drop rule", "Basis for it"],
          [["Different country, never compared",
            "Measured 0 violations in 1,038,755 links. Loses nothing."],
           ["Words appearing in over 20 percent of rows are ignored",
            "They carry no information. ltd matching ltd is not evidence."],
           ["Keep only the top 5 names per candidate, and only above cosine 0.60",
            "Mean true match count is 3.46, so 5 leaves headroom. This is the "
            "lossy step."],
           ["Of what survives, accept only above cosine 0.92",
            "Swept nine values on held out data. 0.92 won."]],
          widths=[3.0, 3.5])
    body(doc, "Drop three and drop four are different cutoffs doing different "
              "jobs. 0.60 means worth looking at. 0.92 means I will bet on it.")
    body(doc, "Direction matters, and this is the part that bit us. We scan from "
              "candidate to record, so for each Source 2 or 3 row we find its "
              "best Source 1 names, not the other way round. We do it that way "
              "because each Source 2 or 3 row has at most one owner, which keeps "
              "memory bounded. The cost is that a Source 1 row can be starved if "
              "it never lands in anybody's top five.")

    heading(doc, "Why the accept threshold is so high", size=12, space_before=10)
    body(doc, "F0.5 weights precision double, so the sweep pushes us to be "
              "picky. The score rises and then falls, which means 0.92 is a real "
              "peak and not just the edge of the range we happened to test. We "
              "ran it out to 0.99 on purpose to prove that.")
    table(doc,
          ["Accept threshold", "Local F0.5"],
          [["0.45", "0.3149"],
           ["0.75", "0.4969"],
           ["0.85", "0.5519"],
           ["0.92", "0.5659  (best)"],
           ["0.95", "0.5652"],
           ["0.99", "0.5567"]],
          widths=[2.0, 2.0],
          bold_rows={3})

    heading(doc, "Results", size=12, space_before=6)
    table(doc,
          ["Metric", "Value"],
          [["Local F0.5 on held out data", "0.5659"],
           ["Candidate recall", "0.7123"],
           ["Score if we predicted nothing for everyone", "0.0561"],
           ["Test set run time", "2,365 seconds, about 40 minutes"],
           ["Candidate pairs generated", "36,067,493"],
           ["Memory those pairs occupied", "757 MB, at 21 bytes per pair"],
           ["Matches predicted", "12,918,475"],
           ["Rows predicted as having no match", "327,203, which is 18.9 percent"],
           ["Amazon official validator", "PASS"]],
          widths=[3.6, 2.9])

    # ------------------------------------------------------------- problems
    heading(doc, "The two honest problems")
    body(doc, "Both of these are written down rather than hidden, because they "
              "tell us what to work on next.")

    numbered(doc, "We are calling roughly three times too many businesses a "
                  "no match. That is a direct consequence of the accept "
                  "threshold being cautious at 0.92, and it is a clean sign "
                  "that there is score still sitting on the table.",
             bold_prefix="Predicted singletons are 18.9 percent, but the true "
                         "rate is 5.58 percent. ")
    numbered(doc, "The blocking step is throwing away 29 percent of the correct "
                  "answers before scoring even begins. No model can recover "
                  "them, so a perfect classifier bolted on top of this still "
                  "could not beat about 0.71. Fixing this is worth more than "
                  "everything else on our list, and the fix is already written "
                  "up: scan in both directions and combine the results.",
             bold_prefix="Candidate recall is 0.7123. ")

    # --------------------------------------------------------------- sources
    heading(doc, "Where to check any of this")
    table(doc,
          ["Claim", "File"],
          [["All parameter values",
            "code/business_entity_resolution/src/blocking.py, run_baseline.py"],
           ["Cleaning and transliteration rules",
            "code/business_entity_resolution/src/normalize.py"],
           ["Scoring function and its self test",
            "code/business_entity_resolution/src/metric.py"],
           ["Dataset statistics",
            "code/business_entity_resolution/src/profile_data.py"],
           ["Every run, including the failed ones", "experiments/experiments.md"],
           ["Next steps per phase", "handoff/"]],
          widths=[2.6, 3.9])

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
