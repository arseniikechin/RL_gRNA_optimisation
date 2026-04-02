#!/usr/bin/env python3
"""
For `data/eval_100_optimized.csv` this script computes:
  1) Doench 2016 (on-target) for initial and optimized guides
  2) Off-targets (<= max_mismatches, PAM NGG/CC) + CFD
  3) CRISPR-BERT (aggregated mean/max/count per gRNA spacer)

Off-target search:
  default genome FASTA: `data/genes_with_flankers.fna`

Expected input CSV columns:
  - sgRNA              (20 nt, initial spacer)
  - sgRNA_optimized    (20 nt, optimized spacer)
  - pam                (3 nt, e.g. TGG/CGG/AGG)
  - window             (30 nt, 4 + 20 + 3 + 3; used for Doench)

Output:
  same CSV with additional columns:
    doench_initial, doench_optimized
    n_off_targets_initial, max_cfd_initial, mean_cfd_initial
    n_off_targets_optimized, max_cfd_optimized, mean_cfd_optimized
    crisprbert_mean_initial, crisprbert_max_initial, crisprbert_n_initial
    crisprbert_mean_optimized, crisprbert_max_optimized, crisprbert_n_optimized
"""

import argparse
import os
import sys
from statistics import mean
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# sys.path: add repository root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


def cfd_score(guide_20: str, off_target_20: str) -> float:
    """
    Simple CFD score: 1.0 for perfect match with a 0.25 penalty per mismatch.
    Matches the `cfd_score` behavior from `diverse_doench_offtarget_cfd_bert.py`.
    """
    guide_20 = (guide_20 or "").upper()[:20]
    off_target_20 = (off_target_20 or "").upper()[:20]
    if len(guide_20) != 20 or len(off_target_20) != 20:
        return 0.0
    penalty_per_mm = 0.25
    score = 1.0
    for i in range(20):
        if guide_20[i] != off_target_20[i]:
            score *= (1.0 - penalty_per_mm)
    return round(score, 6)


def _validate_window_doench(window: str) -> bool:
    if not window:
        return False
    s = str(window).upper().replace("U", "T").strip()
    return len(s) == 30 and set(s) <= set("ACGT")


def _validate_spacer_20(seq: str) -> bool:
    if not seq:
        return False
    s = str(seq).upper().replace("U", "T").strip()
    return len(s) == 20 and set(s) <= set("ACGT")


def run_doench_2016(df: pd.DataFrame) -> pd.DataFrame:
    from modules.doench_r import init_r_env, generate_features_in_r, predict_from_r_model

    if "window" not in df.columns:
        raise ValueError("Column 'window' is required for Doench 2016")
    if "sgRNA_optimized" not in df.columns:
        raise ValueError("Column 'sgRNA_optimized' is required for Doench 2016")

    valid_acgt = set("ACGT")
    windows_initial: List[str] = []
    windows_optimized: List[str] = []
    valid_mask: List[bool] = []

    for _, row in df.iterrows():
        w = (row.get("window") or "").upper().replace("U", "T").strip()
        g_opt = (row.get("sgRNA_optimized") or "").upper().replace("U", "T").strip()[:20]
        ok = len(w) == 30 and set(w) <= valid_acgt and len(g_opt) == 20 and set(g_opt) <= valid_acgt
        valid_mask.append(ok)
        windows_initial.append(w if ok else "A" * 30)
        windows_optimized.append((w[:4] + g_opt + w[24:30]) if ok else "A" * 30)

    init_r_env()

    print("Doench 2016: initial...")
    features_initial = generate_features_in_r(windows_initial)
    scores_initial = predict_from_r_model(features_initial)

    print("Doench 2016: optimized...")
    features_optimized = generate_features_in_r(windows_optimized)
    scores_optimized = predict_from_r_model(features_optimized)

    df["doench_initial"] = [round(float(s), 4) if ok else None for s, ok in zip(scores_initial, valid_mask)]
    df["doench_optimized"] = [round(float(s), 4) if ok else None for s, ok in zip(scores_optimized, valid_mask)]

    n_invalid = sum(1 for ok in valid_mask if not ok)
    if n_invalid:
        print(f"  Warning: {n_invalid} rows with invalid window/spacer -> Doench = NaN")

    return df


def run_offtarget_and_cfd(
    df: pd.DataFrame,
    genome_path: str,
    max_mismatches: int = 4,
    crisprbert_max_offtargets_per_guide: int = 2000,
    skip_cfd: bool = False,
) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    from RL.grna_rl_adapters import load_genome, find_off_targets_in_genome

    for col in ("sgRNA", "sgRNA_optimized", "pam"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required for off-target/CFD")

    print(f"Loading genome FASTA: {genome_path}")
    genome_seq = load_genome(genome_path)
    print(f"  Genome length (concatenated): {len(genome_seq)}")

    n_off_initial: List[int] = []
    max_cfd_initial: List[float] = []
    mean_cfd_initial: List[float] = []

    n_off_optimized: List[int] = []
    max_cfd_optimized: List[float] = []
    mean_cfd_optimized: List[float] = []

    pairs_initial: List[Dict] = []
    pairs_optimized: List[Dict] = []

    for _, row in df.iterrows():
        pam = (row.get("pam") or "GGG").upper()[:3].ljust(3, "G")
        g0 = (row.get("sgRNA") or "").upper().replace("U", "T")[:20]
        g1 = (row.get("sgRNA_optimized") or "").upper().replace("U", "T")[:20]

        if len(g0) != 20 or len(g1) != 20:
            n_off_initial.append(0)
            max_cfd_initial.append(0.0)
            mean_cfd_initial.append(0.0)
            n_off_optimized.append(0)
            max_cfd_optimized.append(0.0)
            mean_cfd_optimized.append(0.0)
            continue

        # exclude_23 = guide spacer + PAM (23 nt)
        exclude_i = g0 + pam
        exclude_o = g1 + pam

        offs_i = find_off_targets_in_genome(
            g0,
            genome_seq=genome_seq,
            max_mismatches=max_mismatches,
            pam_suffixes=("GG",),
            exclude_23=exclude_i,
        )
        offs_o = find_off_targets_in_genome(
            g1,
            genome_seq=genome_seq,
            max_mismatches=max_mismatches,
            pam_suffixes=("GG",),
            exclude_23=exclude_o,
        )

        n_off_initial.append(len(offs_i))
        n_off_optimized.append(len(offs_o))

        # CFD
        if skip_cfd:
            max_cfd_initial.append(0.0)
            mean_cfd_initial.append(0.0)
            max_cfd_optimized.append(0.0)
            mean_cfd_optimized.append(0.0)
        else:
            if offs_i:
                cfd_vals_i = [cfd_score(g0, ot[:20]) for ot in offs_i]
                max_cfd_initial.append(round(max(cfd_vals_i), 6))
                mean_cfd_initial.append(round(mean(cfd_vals_i), 6))
            else:
                max_cfd_initial.append(0.0)
                mean_cfd_initial.append(0.0)

            if offs_o:
                cfd_vals_o = [cfd_score(g1, ot[:20]) for ot in offs_o]
                max_cfd_optimized.append(round(max(cfd_vals_o), 6))
                mean_cfd_optimized.append(round(mean(cfd_vals_o), 6))
            else:
                max_cfd_optimized.append(0.0)
                mean_cfd_optimized.append(0.0)

        # CRISPR-BERT pairs: take first K off-targets per guide
        k_i = offs_i[:crisprbert_max_offtargets_per_guide] if crisprbert_max_offtargets_per_guide else offs_i
        k_o = offs_o[:crisprbert_max_offtargets_per_guide] if crisprbert_max_offtargets_per_guide else offs_o

        guide23_i = g0 + pam
        guide23_o = g1 + pam
        for ot in k_i:
            pairs_initial.append({"sgRNA": guide23_i, "off_target": ot, "label": -1})
        for ot in k_o:
            pairs_optimized.append({"sgRNA": guide23_o, "off_target": ot, "label": -1})

    df["n_off_targets_initial"] = n_off_initial
    df["max_cfd_initial"] = max_cfd_initial
    df["mean_cfd_initial"] = mean_cfd_initial

    df["n_off_targets_optimized"] = n_off_optimized
    df["max_cfd_optimized"] = max_cfd_optimized
    df["mean_cfd_optimized"] = mean_cfd_optimized

    return df, pairs_initial, pairs_optimized


def run_crisprbert_for_pairs(
    pairs_list: List[Dict],
    weights_path: str,
    n_predict: int = -1,
) -> Optional[pd.DataFrame]:
    if not pairs_list:
        return None

    from modules.core import (
        prepare_crisprbert_df,
        add_rnn_encoded_column,
        add_bert_encoding_columns,
        run_crisprbert_prediction,
    )
    from models.CRISPR_BERT.model import build_bert

    df_pairs = pd.DataFrame(pairs_list)
    df_cb = prepare_crisprbert_df(df_pairs[["sgRNA", "off_target", "label"]])
    df_cb = add_rnn_encoded_column(df_cb, column="sequence", n_nucl=24)
    df_cb = add_bert_encoding_columns(df_cb, column="sequence", n_nucl=24)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"CRISPR-BERT weights not found: {weights_path}")

    model = build_bert()
    model.load_weights(weights_path)

    df_pred = run_crisprbert_prediction(df_cb, model, n_predict=n_predict)
    df_pred["sgRNA_core"] = df_pred["sgRNA"].str[:20]

    agg = df_pred.groupby("sgRNA_core")["prediction"].agg(["mean", "max", "count"]).reset_index()
    agg = agg.rename(
        columns={
            "mean": "crisprbert_mean",
            "max": "crisprbert_max",
            "count": "crisprbert_n",
        }
    )
    return agg


def main():
    parser = argparse.ArgumentParser(description="Doench + off-target + CFD + CRISPR-BERT for eval_100_optimized.csv")
    parser.add_argument("--input", default="data/eval_100_optimized.csv", help="Input CSV")
    parser.add_argument("--output", default="data/eval_100_optimized_metrics_doench_offtarget_cfd_crisprbert.csv", help="Output CSV")
    parser.add_argument("--genome", default="data/genes_with_flankers.fna", help="Genome FASTA for off-target search")
    parser.add_argument("--max-mismatches", type=int, default=4, help="Max mismatches for off-target search")
    parser.add_argument("--crisprbert-max-offtargets-per-guide", type=int, default=2000, help="Cap off-targets per guide for CRISPR-BERT")

    parser.add_argument("--skip-doench", action="store_true", help="Skip Doench 2016")
    parser.add_argument("--skip-offtarget-cfd", action="store_true", help="Skip off-target + CFD")
    parser.add_argument("--skip-crisprbert", action="store_true", help="Skip CRISPR-BERT")
    parser.add_argument("--skip-cfd", action="store_true", help="Do not compute CFD (keep only n_off_targets and BERT pairs)")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if not args.skip_doench:
        print("=== Doench 2016 ===")
        df = run_doench_2016(df)

    pairs_initial: List[Dict] = []
    pairs_optimized: List[Dict] = []

    if not args.skip_offtarget_cfd or not args.skip_crisprbert:
        print("=== Off-target + CFD ===")
        df, pairs_initial, pairs_optimized = run_offtarget_and_cfd(
            df=df,
            genome_path=args.genome,
            max_mismatches=args.max_mismatches,
            crisprbert_max_offtargets_per_guide=args.crisprbert_max_offtargets_per_guide,
            skip_cfd=args.skip_cfd or args.skip_offtarget_cfd,
        )
    else:
        # If off-target is fully skipped and BERT is not needed either
        pairs_initial = []
        pairs_optimized = []

    if not args.skip_crisprbert:
        print("=== CRISPR-BERT ===")
        repo_root = os.path.abspath(os.path.dirname(__file__))
        weights_path = os.path.join(repo_root, "models", "CRISPR_BERT", "weight", "I1.h5")

        agg_i = run_crisprbert_for_pairs(pairs_initial, weights_path=weights_path, n_predict=-1)
        agg_o = run_crisprbert_for_pairs(pairs_optimized, weights_path=weights_path, n_predict=-1)

        df["_g20_i"] = df["sgRNA"].astype(str).str.upper().str.replace("U", "T").str[:20]
        df["_g20_o"] = df["sgRNA_optimized"].astype(str).str.upper().str.replace("U", "T").str[:20]

        if agg_i is not None:
            ai = agg_i.rename(
                columns={
                    "crisprbert_mean": "crisprbert_mean_initial",
                    "crisprbert_max": "crisprbert_max_initial",
                    "crisprbert_n": "crisprbert_n_initial",
                }
            )
            ai = ai.rename(columns={"sgRNA_core": "_g20_i"})
            df = df.merge(ai[["_g20_i", "crisprbert_mean_initial", "crisprbert_max_initial", "crisprbert_n_initial"]], on="_g20_i", how="left")
        else:
            df["crisprbert_mean_initial"] = np.nan
            df["crisprbert_max_initial"] = np.nan
            df["crisprbert_n_initial"] = 0

        if agg_o is not None:
            ao = agg_o.rename(
                columns={
                    "crisprbert_mean": "crisprbert_mean_optimized",
                    "crisprbert_max": "crisprbert_max_optimized",
                    "crisprbert_n": "crisprbert_n_optimized",
                }
            )
            ao = ao.rename(columns={"sgRNA_core": "_g20_o"})
            df = df.merge(ao[["_g20_o", "crisprbert_mean_optimized", "crisprbert_max_optimized", "crisprbert_n_optimized"]], on="_g20_o", how="left")
        else:
            df["crisprbert_mean_optimized"] = np.nan
            df["crisprbert_max_optimized"] = np.nan
            df["crisprbert_n_optimized"] = 0

        df["crisprbert_n_initial"] = df["crisprbert_n_initial"].fillna(0).astype(int)
        df["crisprbert_n_optimized"] = df["crisprbert_n_optimized"].fillna(0).astype(int)
        df = df.drop(columns=["_g20_i", "_g20_o"], errors="ignore")

    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

