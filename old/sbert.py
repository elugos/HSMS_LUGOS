#!/usr/bin/env python3
"""
sbert.py
--------
Encode OpenIE triples with Sentence-BERT (all-MiniLM-L6-v2).

- Accepts a single CSV file OR a directory of CSV files (case-insensitive .csv/.CSV).
- Outputs CSV files with embeddings serialized as JSON lists.
- Keeps ALL original columns and adds:
    - sbert_text      (joined [subject, relation, object])
    - embedding_json  (JSON list of floats)

Usage examples:
    # Single file -> single CSV
    python sbert.py --input /path/file_triples.csv --output /path/file_triples_sbert.csv

    # Folder -> outputs one CSV per input into outdir
    python sbert.py --input /path/folder --output /path/outdir

Options you may tweak:
    --cols subject relation object   # which columns to embed
    --joiner " | "                   # how to join them
    --batch-size 256
    --device cpu|cuda
    --normalize                      # L2-normalize embeddings
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sentence_transformers import SentenceTransformer


def read_csv_robust(path: Path, encodings: Iterable[str] = ("utf-8", "latin-1")) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} ({last_err})")


def build_texts(df: pd.DataFrame, cols: List[str], joiner: str) -> pd.Series:
    parts = [df[c].fillna("").astype(str).str.strip() if c in df.columns
             else pd.Series([""] * len(df), index=df.index) for c in cols]
    s = parts[0]
    for p in parts[1:]:
        s = s + joiner + p
    # collapse multiple joiners caused by empties, and strip
    j = joiner.strip()
    if j:
        s = s.str.replace(rf"(?:\s*{j}\s*)+", f" {j} ", regex=True)
    return s.str.strip()


def process_one_file(
    model: SentenceTransformer,
    in_csv: Path,
    out_csv: Path,
    cols: List[str],
    joiner: str,
    batch_size: int,
    normalize: bool,
):
    df = read_csv_robust(in_csv)

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"[WARN] {in_csv.name}: missing columns {missing_cols}; they will be treated as empty strings.")

    sbert_text = build_texts(df, cols, joiner)

    embeddings = model.encode(
        sbert_text.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    out_df = df.copy()
    out_df["sbert_text"] = sbert_text
    out_df["embedding_json"] = [json.dumps(vec.tolist()) for vec in embeddings]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] {in_csv.name} -> {out_csv.name} (rows: {len(out_df)})")


def main():
    ap = argparse.ArgumentParser(description="Encode triples CSV(s) with Sentence-BERT (CSV in, CSV out).")
    ap.add_argument("--input", "-i", type=Path, required=True,
                    help="Path to a CSV file OR a directory of CSV files.")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Output CSV path (if input is a file) OR output directory (if input is a folder).")
    ap.add_argument("--cols", nargs="+", default=["subject", "relation", "object"],
                    help="Columns to join for embedding input (default: subject relation object).")
    ap.add_argument("--joiner", default=" ", help="String to join input parts (default: ' | ').")
    ap.add_argument("--batch-size", type=int, default=256, help="Encode batch size (default: 256).")
    ap.add_argument("--device", default=None, help="Force device (e.g., 'cpu', 'cuda'); defaults to auto.")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings.")
    args = ap.parse_args()

    # Load model once
    model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    if args.input.is_dir():
        # Process all .csv/.CSV in the folder
        files = sorted(set(args.input.glob("*.csv")) | set(args.input.glob("*.CSV")))
        if not files:
            raise SystemExit(f"No CSV files found in directory: {args.input}")
        if (args.output.exists() and args.output.is_file()) or (not args.output.suffix == "" and not args.output.exists()):
            # If output looks like a file path, that's ambiguous for dir input
            raise SystemExit("--output must be a DIRECTORY when --input is a DIRECTORY.")
        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            out_csv = out_dir / f"{f.stem}_sbert.csv"
            print(f"Processing {f} ...")
            process_one_file(model, f, out_csv, args.cols, args.joiner, args.batch_size, args.normalize)
    else:
        # Single file
        in_csv = args.input
        # If --output is a directory, write alongside with _sbert.csv name
        if args.output.is_dir():
            out_csv = args.output / f"{in_csv.stem}_sbert.csv"
        else:
            out_csv = args.output if args.output.suffix.lower() == ".csv" else args.output.with_suffix(".csv")
        process_one_file(model, in_csv, out_csv, args.cols, args.joiner, args.batch_size, args.normalize)


if __name__ == "__main__":
    main()
