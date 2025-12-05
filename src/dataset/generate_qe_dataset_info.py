#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QE Dataset Info Generator
---------------------------------------------
Extracts and merges raw dataset_info.csv files into a QE dataset_info.csv.

Author: Extracted from train_ct_qe_tool.py
"""

from __future__ import annotations

import os
import ast
import logging
import argparse
import pandas as pd
import numpy as np
from typing import List, Optional

# Define global labels
LABELS = ["none", "low", "medium", "high"]
LABEL_SET = set(LABELS)

logger = logging.getLogger(__name__)

def clean_sev(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, float) and np.isnan(x):
        return "none"
    s = str(x).strip().lower()
    if s in ("", "nan", "null", "none"):
        return "none"
    return s


def parse_degradations(x):
    # raw CSV typically stores as "['ldct','svct']"
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return None
    return None


def severities_to_degradations(ldct: str, lact: str, svct: str) -> str:
    d = []
    if ldct != "none":
        d.append("ldct")
    if lact != "none":
        d.append("lact")
    if svct != "none":
        d.append("svct")
    return str(d)  # "['ldct','svct']" or "[]"


def build_qe_dataset_info(
    raw_csvs: List[str],
    out_csv: str,
    dedup_clean: bool = True,
    max_clean: Optional[int] = None,
) -> str:
    """
    Merge multiple raw dataset_info.csv into a QE dataset_info with columns:
      image_paths, ldct_severities, lact_severities, svct_severities, degradations
    plus appended clean rows from label_paths of single-degradation samples.

    Returns: out_csv path
    """
    dfs = []
    for p in raw_csvs:
        df = pd.read_csv(p)
        df["__src__"] = p
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    required = ["image_paths", "label_paths", "ldct_severities", "lact_severities", "svct_severities", "degradations"]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw dataset_info: {missing}")

    for c in ["ldct_severities", "lact_severities", "svct_severities"]:
        df_all[c] = df_all[c].apply(clean_sev)
        bad = ~df_all[c].isin(LABEL_SET)
        if bad.any():
            raise ValueError(f"Invalid values in {c}: {df_all.loc[bad, c].head(10).tolist()}")

    # base QE rows
    qe_df = df_all[["image_paths", "ldct_severities", "lact_severities", "svct_severities"]].copy()
    qe_df["__is_clean_added__"] = False

    # find single-degradation rows -> add clean images from label_paths
    df_all["degradations_parsed"] = df_all["degradations"].apply(parse_degradations)
    is_single = df_all["degradations_parsed"].apply(lambda x: isinstance(x, list) and len(x) == 1)
    clean_paths = df_all.loc[is_single, "label_paths"].astype(str).tolist()
    if dedup_clean:
        clean_paths = sorted(set(clean_paths))
    if max_clean is not None:
        clean_paths = clean_paths[:max_clean]

    clean_df = pd.DataFrame({
        "image_paths": clean_paths,
        "ldct_severities": ["none"] * len(clean_paths),
        "lact_severities": ["none"] * len(clean_paths),
        "svct_severities": ["none"] * len(clean_paths),
        "__is_clean_added__": [True] * len(clean_paths),
    })

    qe_df = pd.concat([qe_df, clean_df], ignore_index=True)

    # Method B: regenerate degradations from severities (self-consistent)
    qe_df["degradations"] = qe_df.apply(
        lambda r: severities_to_degradations(r["ldct_severities"], r["lact_severities"], r["svct_severities"]),
        axis=1,
    )

    qe_df = qe_df.dropna(subset=["image_paths"]).reset_index(drop=True)
    qe_df["image_paths"] = qe_df["image_paths"].astype(str)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    qe_df.to_csv(out_csv, index=False)
    logger.info(
        f"QE dataset_info saved: {out_csv} | raw_rows={len(df_all)} | added_clean={len(clean_df)} | final_rows={len(qe_df)}"
    )
    return out_csv

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate QE Dataset Info")
    
    parser.add_argument("--dataset", type=str, choices=["deeplesion", "mayo"], default="deeplesion",
                        help="选择要处理的数据集: deeplesion 或 mayo")
    parser.add_argument("--raw_dataset_info_root", type=str, default="/data/hyq/codes/AgenticCT/data",
                        help="原始dataset_info所在根目录，脚本将按数据集、退化类型及数据划分自动拼接")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="all",
                        help="选择处理的数据划分: train/val/test 或 all(三者合并)")
    parser.add_argument("--qe_dataset_info_out", type=str, default=None,
                        help="输出QE的dataset_info.csv路径；未设置时按数据集和划分默认路径生成")
    parser.add_argument("--dedup_clean", action="store_true", help="Deduplicate clean(label_paths) when preprocessing")
    parser.add_argument("--max_clean", type=int, default=None, help="Max clean samples appended (default: all)")

    args = parser.parse_args()
    
    dataset = args.dataset
    root = args.raw_dataset_info_root.rstrip("/")
    split = args.split

    def split_paths(s):
        base = f"{root}/{dataset}"
        if s == "all":
            return [
                f"{base}/ldct/train_dataset_info.csv",
                f"{base}/lact/train_dataset_info.csv",
                f"{base}/svct/train_dataset_info.csv",
                f"{base}/ldct/val_dataset_info.csv",
                f"{base}/lact/val_dataset_info.csv",
                f"{base}/svct/val_dataset_info.csv",
                f"{base}/ldct/test_dataset_info.csv",
                f"{base}/lact/test_dataset_info.csv",
                f"{base}/svct/test_dataset_info.csv",
            ]
        else:
            return [
                f"{base}/ldct/{s}_dataset_info.csv",
                f"{base}/lact/{s}_dataset_info.csv",
                f"{base}/svct/{s}_dataset_info.csv",
            ]

    raw_csvs = split_paths(split)

    if args.qe_dataset_info_out is None:
        if split == "all":
            out_csv = f"{root}/qe/{dataset}/dataset_info_qe.csv"
        else:
            out_csv = f"{root}/qe/{dataset}/{split}/dataset_info_qe.csv"
    else:
        out_csv = args.qe_dataset_info_out

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    build_qe_dataset_info(
        raw_csvs=raw_csvs,
        out_csv=out_csv,
        dedup_clean=args.dedup_clean,
        max_clean=args.max_clean,
    )

if __name__ == "__main__":
    main()
