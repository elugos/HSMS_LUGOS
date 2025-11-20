#!/usr/bin/env python3
"""
openie_batch.py
----------------
Batch-extract subject–relation–object triples from a folder of CSV files using Stanford OpenIE,
then dedupe by subject keeping the row whose `relation` has the longest string.

Output columns (fixed):
    ["GlobalEventID", "date", "source_index", "subject", "relation", "object"]

Requirements:
    - Python 3.8+
    - stanford-openie (`pip install stanford-openie`)
    - pandas
    - Java (JRE/JDK) available on PATH (`java -version`)
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

try:
    from openie import StanfordOpenIE
except Exception as e:
    raise SystemExit(
        "Could not import 'openie'. Install it with:\n\n"
        "    pip install stanford-openie\n\n"
        f"Original error: {e}"
    )


# ----------------------------
# Utilities
# ----------------------------
def sentence_split(text: str) -> List[str]:
    """Lightweight sentence splitter (no extra deps)."""
    if not text:
        return []
    t = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def annotate_text(client: StanfordOpenIE, text: str) -> List[Dict[str, str]]:
    """Run OpenIE on a single text and return triples."""
    triples = client.annotate(text)
    return [
        {"subject": t.get("subject", ""), "relation": t.get("relation", ""), "object": t.get("object", "")}
        for t in triples
    ]


def _safe_get(df: pd.DataFrame, idx: Any, col: str) -> Any:
    """Safely extract a value; return '' if missing/NaN."""
    if col not in df.columns:
        return ""
    val = df.at[idx, col]
    if pd.isna(val):
        return ""
    return val


# ----------------------------
# Your requested dedupe function
# ----------------------------
def dedupe_relation_longest_object(df: pd.DataFrame,
                                  subject_col: str = "subject",
                                  relation_col: str = "relation") -> pd.DataFrame:
    """
    Deduplicate by `subject_col`, keeping the row where `relation_col` has the longest string length.
    Ties keep the first occurrence.
    """
    df = df.copy()
    df["_len"] = df[relation_col].fillna("").astype(str).str.len()
    idx = df.groupby(subject_col)["_len"].idxmax()
    return df.loc[idx].drop(columns=["_len"]).reset_index(drop=True)


# ----------------------------
# File processing
# ----------------------------
def process_file(
    client: StanfordOpenIE,
    in_path: Path,
    out_dir: Path,
    text_col: str = "text",
    split_sentences_flag: bool = False,
    max_chars: int | None = None,
    encoding_candidates: Iterable[str] = ("utf-8", "latin-1"),
) -> int:
    """
    Process one CSV file: extract triples for each non-empty row in `text_col`,
    then DEDUPE by subject keeping the longest `relation`, and write <stem>_triples.csv.

    Output columns fixed to:
        ["GlobalEventID","date","source_index","subject","relation","object"]
    """
    df = None
    last_err = None
    for enc in encoding_candidates:
        try:
            df = pd.read_csv(in_path, encoding=enc, low_memory=False)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        print(f"[SKIP] {in_path.name}: failed to read CSV ({last_err})")
        return 0

    if text_col not in df.columns:
        print(f"[SKIP] {in_path.name}: missing column '{text_col}'")
        return 0

    rows_out: List[Dict[str, Any]] = []
    for idx, raw in df[text_col].items():
        if pd.isna(raw):
            continue
        text = str(raw).strip()
        if not text:
            continue

        if max_chars and len(text) > max_chars:
            text = text[:max_chars]

        geid = _safe_get(df, idx, "GlobalEventID")
        date_val = _safe_get(df, idx, "date")

        try:
            if split_sentences_flag:
                for sent in sentence_split(text):
                    for t in annotate_text(client, sent):
                        rows_out.append({
                            "GlobalEventID": geid,
                            "date": date_val,
                            "source_index": idx,
                            "subject": t["subject"],
                            "relation": t["relation"],
                            "object": t["object"],
                        })
            else:
                for t in annotate_text(client, text):
                    rows_out.append({
                        "GlobalEventID": geid,
                        "date": date_val,
                        "source_index": idx,
                        "subject": t["subject"],
                        "relation": t["relation"],
                        "object": t["object"],
                    })
        except Exception as e:
            rows_out.append({
                "GlobalEventID": geid,
                "date": date_val,
                "source_index": idx,
                "subject": "",
                "relation": f"ERROR: {e}",
                "object": "",
            })

    # Build DataFrame with fixed column order
    out_cols = ["GlobalEventID", "date", "source_index", "subject", "relation", "object"]
    triples_df = pd.DataFrame(rows_out, columns=out_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_triples.csv"

    # Write final (deduped) CSV
    triples_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] {in_path.name} -> {out_path.name} ({len(triples_df)} rows after dedupe)")
    return len(triples_df)


def main():
    ap = argparse.ArgumentParser(description="Batch Stanford OpenIE over CSV folder (dedupe by longest relation per subject).")
    ap.add_argument("--input", "-i", type=Path, required=True, help="Input directory of CSV files.")
    ap.add_argument("--output", "-o", type=Path, required=True, help="Directory to write *_triples.csv files.")
    ap.add_argument("--text-col", default="title", help="Name of the text column in input CSVs (default: text).")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern for input files (default: *.csv).")
    ap.add_argument("--split-sentences", action="store_true", help="Split texts into sentences before annotation.")
    ap.add_argument("--max-chars", type=int, default=None, help="Truncate each text to at most this many characters.")
    args = ap.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    glob_pat: str = args.glob

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Support both .csv and .CSV
    files = sorted(set(input_dir.glob(glob_pat)) | set(input_dir.glob(glob_pat.upper())))
    if not files:
        raise SystemExit(f"No files matched '{glob_pat}' (or '{glob_pat.upper()}') in {input_dir}")

    with StanfordOpenIE() as client:
        total = 0
        for p in files:
            total += process_file(
                client=client,
                in_path=p,
                out_dir=output_dir,
                text_col=args.text_col,
                split_sentences_flag=args.split_sentences,
                max_chars=args.max_chars,
            )
    print(f"Done. Total rows written: {total}")


if __name__ == "__main__":
    main()
