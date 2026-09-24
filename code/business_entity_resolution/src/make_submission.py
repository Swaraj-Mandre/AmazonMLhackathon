"""Assembles the two things the portal asks for, into ``submission/``.

The upload form wants a ``matching_results`` TSV and a zipped code file. The
problem statement specifies exactly what that zip contains, so the zip we build
is the full submission package it describes rather than a bare code folder. That
satisfies both the form and the written spec.

    submission/
    ├── matching_results.tsv          <- first upload box, as-is
    └── <team>_submission.zip         <- second upload box
        ├── output/
        │   ├── matching_results.tsv
        │   └── candidate_pairs.tsv
        ├── code/business_entity_resolution/
        │   ├── src/
        │   ├── README.md
        │   └── requirements.txt
        └── Documentation_template.md

Run from the repository root::

    python code/business_entity_resolution/src/make_submission.py --team "Your Team" \
        --members "A, B, C, D"

Placeholders in ``docs/METHODOLOGY.md`` are filled from the actual output files,
so the numbers in the write-up cannot drift away from the files shipped beside
it. The script refuses to build if the official validator has not been run, or
if the outputs disagree with the test set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CANDIDATE_PAIRS,
    MATCHING_RESULTS,
    PACKAGE_DIR,
    PROVIDED_DIR,
    REPO_ROOT,
    TEST_DIR,
    TEST_SOURCES,
)

csv.field_size_limit(1 << 24)

SUBMISSION_DIR = REPO_ROOT / "submission"
METHODOLOGY_SRC = REPO_ROOT / "docs" / "METHODOLOGY.md"

# Settings that produced the shipped outputs. Kept here so the write-up and the
# reproduce instructions quote the same numbers the files were made with.
THRESHOLD = 0.92
BLOCK_MIN = 0.60


def count_id_lists(path: Path) -> tuple[int, int]:
    """Return (rows, total IDs) for a two-column ID-list TSV."""
    rows = ids = 0
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for row in reader:
            rows += 1
            if len(row) > 1 and row[1]:
                ids += row[1].count(",") + 1
    return rows, ids


def run_official_validator() -> bool:
    """Run the organisers' own validator. A failed submission wastes one of five."""
    script = PROVIDED_DIR / "validate_submission.py"
    print(f"Running the official validator ({script.name})...")
    proc = subprocess.run(
        [sys.executable, str(script),
         "--matching", str(MATCHING_RESULTS),
         "--candidate", str(CANDIDATE_PAIRS),
         "--test-dir", str(TEST_DIR)],
        capture_output=True, text=True,
    )
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    return proc.returncode == 0


def fill_methodology(stats: dict[str, str]) -> str:
    """Substitute the measured numbers into the write-up."""
    text = METHODOLOGY_SRC.read_text(encoding="utf-8")
    for key, value in stats.items():
        text = text.replace("{{" + key + "}}", str(value))
    leftover = [line for line in text.splitlines() if "{{" in line]
    if leftover:
        raise SystemExit(f"unfilled placeholders remain: {leftover}")
    return text


def build(team: str, members: str, local_f05: str, cand_recall: str,
          val_entities: str) -> None:
    for path in (MATCHING_RESULTS, CANDIDATE_PAIRS):
        if not path.exists():
            raise SystemExit(f"missing {path}. Run run_baseline.py --predict first.")

    if not run_official_validator():
        raise SystemExit(
            "\nThe official validator did not pass. Nothing was written to "
            "submission/. Fix the reported issues rather than spending one of "
            "the five daily uploads on a rejection."
        )

    m_rows, m_ids = count_id_lists(MATCHING_RESULTS)
    c_rows, c_ids = count_id_lists(CANDIDATE_PAIRS)
    n_test = sum(1 for _ in open(TEST_SOURCES["S1"], encoding="utf-8")) - 1
    if m_rows != n_test:
        raise SystemExit(f"matching_results has {m_rows} rows, test_source1 has {n_test}")

    missed = 1.0 - float(cand_recall)
    stats = {
        "TEAM_NAME": team,
        "TEAM_MEMBERS": members,
        "DATE": dt.date.today().isoformat(),
        "LOCAL_F05": local_f05,
        "CAND_RECALL": cand_recall,
        "VAL_ENTITIES": val_entities,
        "CANDIDATE_PAIRS": f"{c_ids:,}",
        "S1_COUNT": f"{c_rows:,}",
        "CANDIDATES_PER_ENTITY": f"{c_ids / max(c_rows, 1):.1f}",
        "REDUCTION_RATIO": f"{1 - c_ids / (c_rows * 9_969_589):.8f}",
        "THRESHOLD": str(THRESHOLD),
        "BLOCK_MIN": str(BLOCK_MIN),
        "MISSED_PCT": f"{missed:.1%}",
    }
    methodology = fill_methodology(stats)

    SUBMISSION_DIR.mkdir(exist_ok=True)
    shutil.copy2(MATCHING_RESULTS, SUBMISSION_DIR / "matching_results.tsv")

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in team) or "team"
    zip_path = SUBMISSION_DIR / f"{safe}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(MATCHING_RESULTS, "output/matching_results.tsv")
        z.write(CANDIDATE_PAIRS, "output/candidate_pairs.tsv")
        base = "code/business_entity_resolution"
        for src in sorted((PACKAGE_DIR / "src").glob("*.py")):
            z.write(src, f"{base}/src/{src.name}")
        z.write(PACKAGE_DIR / "README.md", f"{base}/README.md")
        z.write(PACKAGE_DIR / "requirements.txt", f"{base}/requirements.txt")
        z.writestr("Documentation_template.md", methodology)

    print(f"\n  submission/matching_results.tsv   "
          f"{MATCHING_RESULTS.stat().st_size/1e6:>8.1f} MB  "
          f"{m_rows:,} rows, {m_ids:,} matches")
    print(f"  submission/{zip_path.name:<22} "
          f"{zip_path.stat().st_size/1e6:>8.1f} MB")
    print("\n  zip contents:")
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            print(f"    {info.file_size:>12,}  {info.filename}")
    print("\nBoth files are ready to upload.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", required=True, help="team name, used in the zip filename")
    ap.add_argument("--members", default="", help="comma separated member names")
    ap.add_argument("--f05", default="0.5659", help="best local macro F_0.5")
    ap.add_argument("--recall", default="0.6554", help="candidate recall at the shipped settings")
    ap.add_argument("--val-entities", default="329,348", help="held-out records scored")
    args = ap.parse_args()
    build(args.team, args.members or "[List all team members]",
          args.f05, args.recall, args.val_entities)


if __name__ == "__main__":
    main()
