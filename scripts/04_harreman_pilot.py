"""Run a small Harreman transporter communication pilot on paired Visium slides."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from harreman_utils import patch_harreman_readonly_diagonal, run_harreman_for_sample


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "harreman_pilot"
DEFAULT_SAMPLES = ["V10S15-054_B", "V10S15-054_A"]


def plot_top_metabolites(df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    plot_df = (
        df.sort_values(["sample", "z_score"], ascending=[True, False])
        .groupby("sample", observed=True)
        .head(top_n)
        .copy()
    )
    plt.figure(figsize=(11, 8))
    sns.barplot(data=plot_df, x="z_score", y="metabolite", hue="sample")
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title(f"Top {top_n} Harreman transporter metabolite signals per slide")
    plt.xlabel("Harreman parametric z score")
    plt.ylabel("metabolite")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_pair_differences(diff: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    plot_df = pd.concat(
        [
            diff.nlargest(top_n, "fibrotic_minus_adjacent_z"),
            diff.nsmallest(top_n, "fibrotic_minus_adjacent_z"),
        ],
        ignore_index=True,
    ).drop_duplicates("metabolite")
    plot_df = plot_df.sort_values("fibrotic_minus_adjacent_z")
    colors = np.where(plot_df["fibrotic_minus_adjacent_z"] >= 0, "#C44E52", "#4C72B0")

    plt.figure(figsize=(10, 10))
    plt.barh(plot_df["metabolite"], plot_df["fibrotic_minus_adjacent_z"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Harreman pilot metabolite z-score difference")
    plt.xlabel("fibrotic minus adjacent z score")
    plt.ylabel("metabolite")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--adjacent-sample", default="V10S15-054_B")
    parser.add_argument("--fibrotic-sample", default="V10S15-054_A")
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--expression-threshold", type=float, default=0.01)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="CUDA initialization")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="harreman")
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_theme(style="whitegrid")
    patch_harreman_readonly_diagonal()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["sample"] = metadata["Visium Slide ID"].astype(str)
    selected_metadata = metadata[metadata["sample"].isin(args.samples)].copy()
    selected_metadata.to_csv(output_dir / "pilot_slide_metadata.csv", index=False)

    metabolite_tables = []
    gene_pair_tables = []
    run_rows = []

    for sample in args.samples:
        metabolites, gene_pairs, run_summary_row = run_harreman_for_sample(
            sample=sample,
            n_neighbors=args.n_neighbors,
            expression_threshold=args.expression_threshold,
        )
        metabolite_tables.append(metabolites)
        gene_pair_tables.append(gene_pairs)
        run_rows.append(run_summary_row)

    metabolite_df = pd.concat(metabolite_tables, ignore_index=True)
    gene_pair_df = pd.concat(gene_pair_tables, ignore_index=True)
    run_summary = pd.DataFrame(run_rows)

    metabolite_df = metabolite_df.merge(
        selected_metadata[["sample", "Disease Label", "Patient_ID", "General Categorization"]],
        on="sample",
        how="left",
    )
    gene_pair_df = gene_pair_df.merge(
        selected_metadata[["sample", "Disease Label", "Patient_ID"]],
        on="sample",
        how="left",
    )
    run_summary = run_summary.merge(
        selected_metadata[["sample", "Disease Label", "Patient_ID", "General Categorization"]],
        on="sample",
        how="left",
    )

    metabolite_df.to_csv(output_dir / "harreman_metabolite_results.csv", index=False)
    gene_pair_df.to_csv(output_dir / "harreman_gene_pair_results.csv", index=False)
    run_summary.to_csv(output_dir / "harreman_pilot_run_summary.csv", index=False)

    adjacent = metabolite_df[metabolite_df["sample"].eq(args.adjacent_sample)]
    fibrotic = metabolite_df[metabolite_df["sample"].eq(args.fibrotic_sample)]
    paired = adjacent[["metabolite", "z_score", "fdr", "communication_score"]].merge(
        fibrotic[["metabolite", "z_score", "fdr", "communication_score"]],
        on="metabolite",
        suffixes=("_adjacent", "_fibrotic"),
    )
    paired["fibrotic_minus_adjacent_z"] = paired["z_score_fibrotic"] - paired["z_score_adjacent"]
    paired["fibrotic_minus_adjacent_score"] = (
        paired["communication_score_fibrotic"] - paired["communication_score_adjacent"]
    )
    paired = paired.sort_values("fibrotic_minus_adjacent_z", ascending=False)
    paired.to_csv(output_dir / "harreman_paired_metabolite_differences.csv", index=False)

    plot_top_metabolites(metabolite_df, output_dir / "top_metabolite_z_scores.png")
    plot_pair_differences(paired, output_dir / "paired_metabolite_z_differences.png")

    overview = {
        "samples": args.samples,
        "adjacent_sample": args.adjacent_sample,
        "fibrotic_sample": args.fibrotic_sample,
        "n_neighbors": args.n_neighbors,
        "expression_threshold": args.expression_threshold,
        "n_metabolites_tested_intersection": int(len(paired)),
        "top_fibrotic_minus_adjacent_metabolites": paired.head(10)[
            ["metabolite", "fibrotic_minus_adjacent_z"]
        ].to_dict(orient="records"),
        "top_adjacent_minus_fibrotic_metabolites": paired.tail(10)[
            ["metabolite", "fibrotic_minus_adjacent_z"]
        ].sort_values("fibrotic_minus_adjacent_z").to_dict(orient="records"),
    }
    with (output_dir / "harreman_pilot_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
