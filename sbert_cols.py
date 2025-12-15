#!/usr/bin/env python3
"""
sbert.py
--------
Encode text from arbitrary CSV columns using Sentence-BERT.

- Accepts a single CSV file OR a directory of CSV files.
- Outputs CSV files with embeddings serialized as JSON lists.
- Keeps ALL original columns and adds:
    - sbert_text
    - embedding_json

Examples:
    python sbert.py --input data.csv --output data_sbert.csv --cols abstract
    python sbert.py --input data.csv --output out.csv --cols title abstract claims
    python sbert.py --input data.csv --output out.csv --cols ALL
    python sbert.py --input data.csv --output out.csv --text-col full_text
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer


def read_csv_robust(path: Path, encodings: Iterable[str] = ("utf-8", "latin-1")) -> pd.DataFrame:
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError("Failed to read CSV {} ({})".format(path, last_err))

def make_suffix(cols, text_col):
    if text_col is not None:
        return text_col

    if len(cols) == 1 and cols[0].upper() == "ALL":
        return "ALL"

    return "_".join(cols)


def build_texts(
    df: pd.DataFrame,
    cols: List[str],
    joiner: str,
    text_col: Optional[str] = None,
) -> pd.Series:

    # Case 1: pre-built text column
    if text_col is not None:
        if text_col not in df.columns:
            raise ValueError("--text-col '{}' not found in CSV".format(text_col))
        return df[text_col].fillna("").astype(str).str.strip()

    # Case 2: ALL columns
    if len(cols) == 1 and cols[0].upper() == "ALL":
        cols = list(df.columns)

    # Case 3: selected columns
    parts = []
    for c in cols:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str).str.strip())
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))

    s = parts[0]
    for p in parts[1:]:
        s = s + joiner + p

    j = joiner.strip()
    if j:
        s = s.str.replace(r"(?:\s*{}\s*)+".format(j), " {} ".format(j), regex=True)

    return s.str.strip()


def process_one_file(
    model: SentenceTransformer,
    in_csv: Path,
    out_csv: Path,
    cols: List[str],
    joiner: str,
    batch_size: int,
    normalize: bool,
    text_col: Optional[str],
):
    df = read_csv_robust(in_csv)

    sbert_text = build_texts(df, cols, joiner, text_col)

    embeddings = model.encode(
        sbert_text.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    suffix = make_suffix(cols, text_col)

    text_col_name = "sbert_text_{}".format(suffix)
    embed_col_name = "embedding_json_{}".format(suffix)

    out_df = df.copy()
    out_df[text_col_name] = sbert_text
    out_df[embed_col_name] = [json.dumps(vec.tolist()) for vec in embeddings]


    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print("[OK] {} -> {} (rows: {})".format(in_csv.name, out_csv.name, len(out_df)))


def main():
    ap = argparse.ArgumentParser(description="Encode CSV text columns with Sentence-BERT")
    ap.add_argument("--input", "-i", type=Path, required=True,
                    help="CSV file OR directory of CSV files")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Output CSV file OR output directory")
    ap.add_argument("--cols", nargs="+", default=["subject", "relation", "object"],
                    help="Columns to embed (use ALL for all columns)")
    ap.add_argument("--text-col", default=None,
                    help="Single pre-built text column (overrides --cols)")
    ap.add_argument("--joiner", default=" ",
                    help="String to join columns")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--normalize", action="store_true")

    args = ap.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    if args.input.is_dir():
        if args.output.exists() and args.output.is_file():
            raise SystemExit("--output must be a directory when --input is a directory")

        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(list(args.input.glob("*.csv")))

        if not files:
            raise SystemExit("No CSV files found in {}".format(args.input))

        for f in files:
            out_csv = args.output / (f.stem + "_sbert.csv")
            process_one_file(
                model,
                f,
                out_csv,
                args.cols,
                args.joiner,
                args.batch_size,
                args.normalize,
                args.text_col,
            )
    else:
        if args.output.is_dir():
            out_csv = args.output / (args.input.stem + "_sbert.csv")
        else:
            out_csv = args.output

        process_one_file(
            model,
            args.input,
            out_csv,
            args.cols,
            args.joiner,
            args.batch_size,
            args.normalize,
            args.text_col,
        )


if __name__ == "__main__":
    main()
