#!/usr/bin/env python3
"""
sbert.py
--------
Encode OpenIE triples with Sentence-BERT (all-MiniLM-L6-v2).

*** Updated ***
- In addition to the combined embedding_json column,
  we now also compute separate embeddings for:
      • embedding_subject_json
      • embedding_relation_json
      • embedding_object_json
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

    j = joiner.strip()
    if j:
        s = s.str.replace(rf"(?:\s*{j}\s*)+", f" {j} ", regex=True)
    return s.str.strip()


def encode_single_column(model, series: pd.Series, batch_size: int, normalize: bool):
    """Encode a pandas series containing raw text."""
    return model.encode(
        series.fillna("").astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )


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
        print(f"[WARN] {in_csv.name}: missing columns {missing_cols}; treating as empty.")

    # Combined SBERT text
    sbert_text = build_texts(df, cols, joiner)

    # ================================
    # NEW: embed each column separately
    # ================================
    subj_series = df["subject"].fillna("").astype(str) if "subject" in df.columns else pd.Series([""] * len(df))
    rel_series  = df["relation"].fillna("").astype(str) if "relation" in df.columns else pd.Series([""] * len(df))
    obj_series  = df["object"].fillna("").astype(str) if "object" in df.columns else pd.Series([""] * len(df))

    emb_subj = encode_single_column(model, subj_series, batch_size, normalize)
    emb_rel  = encode_single_column(model, rel_series, batch_size, normalize)
    emb_obj  = encode_single_column(model, obj_series, batch_size, normalize)

    # Original combined embedding
    embeddings = model.encode(
        sbert_text.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    # Build output DataFrame
    out_df = df.copy()
    out_df["sbert_text"] = sbert_text
    out_df["embedding_json"] = [json.dumps(vec.tolist()) for vec in embeddings]

    # NEW columns
    out_df["embedding_subject_json"] = [json.dumps(vec.tolist()) for vec in emb_subj]
    out_df["embedding_relation_json"] = [json.dumps(vec.tolist()) for vec in emb_rel]
    out_df["embedding_object_json"] = [json.dumps(vec.tolist()) for vec in emb_obj]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] {in_csv.name} -> {out_csv.name} (rows: {len(out_df)})")


def main():
    ap = argparse.ArgumentParser(description="Encode triples CSV(s) with Sentence-BERT (CSV in, CSV out).")
    ap.add_argument("--input", "-i", type=Path, required=True,
                    help="Path to a CSV file OR a directory of CSV files.")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Output CSV path (file) OR output directory (folder).")
    ap.add_argument("--cols", nargs="+", default=["subject", "relation", "object"],
                    help="Columns to join for embedding input (default: subject relation object).")
    ap.add_argument("--joiner", default=" | ", help="Joiner between text components (default: ' | ').")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    if args.input.is_dir():
        files = sorted(set(args.input.glob("*.csv")) | set(args.input.glob("*.CSV")))
        if not files:
            raise SystemExit(f"No CSV files found in directory: {args.input}")

        if args.output.exists() and args.output.is_file():
            raise SystemExit("--output must be a DIRECTORY when --input is a DIRECTORY.")

        args.output.mkdir(parents=True, exist_ok=True)
        for f in files:
            out_csv = args.output / f"{f.stem}_sbert.csv"
            print(f"Processing {f} ...")
            process_one_file(model, f, out_csv, args.cols, args.joiner, args.batch_size, args.normalize)

    else:
        in_csv = args.input
        if args.output.is_dir():
            out_csv = args.output / f"{in_csv.stem}_sbert.csv"
        else:
            out_csv = args.output if args.output.suffix.lower() == ".csv" else args.output.with_suffix(".csv")

        process_one_file(model, in_csv, out_csv, args.cols, args.joiner, args.batch_size, args.normalize)


if __name__ == "__main__":
    main()
