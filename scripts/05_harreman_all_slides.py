"""Scale Harreman transporter communication to paired adjacent and fibrotic slides."""

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "harreman_all_slides"


def safe_name(value: str) -> str:
    return value.replace("-", "_")


def paired_slide_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["sample"] = metadata["Visium Slide ID"].astype(str)
    paired_patients = []
    for patient_id, group in metadata.groupby("Patient_ID", observed=True):
        labels = set(group["Disease Label"])
        if {"Adjacent", "Fibrotic"}.issubset(labels):
            paired_patients.append(patient_id)
    selected = metadata[
        metadata["Patient_ID"].isin(paired_patients)
        & metadata["Disease Label"].isin(["Adjacent", "Fibrotic"])
    ].copy()
    return selected.sort_values(["Patient_ID", "Disease Label", "sample"])


def per_sample_paths(output_dir: Path, sample: str) -> dict[str, Path]:
    sample_dir = output_dir / "per_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(sample)
    return {
        "metabolites": sample_dir / f"{prefix}_harreman_metabolites.csv",
        "gene_pairs": sample_dir / f"{prefix}_harreman_gene_pairs.csv",
        "summary": sample_dir / f"{prefix}_harreman_summary.json",
    }


def run_or_load_sample(
    sample: str,
    metadata_row: pd.Series,
    output_dir: Path,
    n_neighbors: int,
    expression_threshold: float,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    paths = per_sample_paths(output_dir, sample)
    if (
        not force
        and paths["metabolites"].exists()
        and paths["gene_pairs"].exists()
        and paths["summary"].exists()
    ):
        metabolites = pd.read_csv(paths["metabolites"])
        gene_pairs = pd.read_csv(paths["gene_pairs"])
        summary = json.loads(paths["summary"].read_text())
        return metabolites, gene_pairs, summary

    metabolites, gene_pairs, summary = run_harreman_for_sample(
        sample=sample,
        n_neighbors=n_neighbors,
        expression_threshold=expression_threshold,
    )
    for df in [metabolites, gene_pairs]:
        df["Disease Label"] = metadata_row["Disease Label"]
        df["Patient_ID"] = metadata_row["Patient_ID"]
        df["General Categorization"] = metadata_row["General Categorization"]

    summary.update(
        {
            "Disease Label": metadata_row["Disease Label"],
            "Patient_ID": int(metadata_row["Patient_ID"]),
            "General Categorization": metadata_row["General Categorization"],
        }
    )

    metabolites.to_csv(paths["metabolites"], index=False)
    gene_pairs.to_csv(paths["gene_pairs"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2))
    return metabolites, gene_pairs, summary


def compute_paired_differences(metabolites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, patient_df in metabolites.groupby("Patient_ID", observed=True):
        adjacent = patient_df[patient_df["Disease Label"].eq("Adjacent")]
        fibrotic = patient_df[patient_df["Disease Label"].eq("Fibrotic")]
        if adjacent.empty or fibrotic.empty:
            continue
        adjacent_mean = (
            adjacent.groupby("metabolite", observed=True)
            .agg(
                adjacent_z=("z_score", "mean"),
                adjacent_score=("communication_score", "mean"),
                n_adjacent_slides=("sample", "nunique"),
            )
            .reset_index()
        )
        fibrotic_mean = (
            fibrotic.groupby("metabolite", observed=True)
            .agg(
                fibrotic_z=("z_score", "mean"),
                fibrotic_score=("communication_score", "mean"),
                n_fibrotic_slides=("sample", "nunique"),
            )
            .reset_index()
        )
        paired = adjacent_mean.merge(fibrotic_mean, on="metabolite", how="inner")
        paired["Patient_ID"] = patient_id
        paired["fibrotic_minus_adjacent_z"] = paired["fibrotic_z"] - paired["adjacent_z"]
        paired["fibrotic_minus_adjacent_score"] = paired["fibrotic_score"] - paired["adjacent_score"]
        rows.append(paired)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_differences(paired: pd.DataFrame) -> pd.DataFrame:
    summary = (
        paired.groupby("metabolite", observed=True)
        .agg(
            mean_fibrotic_minus_adjacent_z=("fibrotic_minus_adjacent_z", "mean"),
            median_fibrotic_minus_adjacent_z=("fibrotic_minus_adjacent_z", "median"),
            std_fibrotic_minus_adjacent_z=("fibrotic_minus_adjacent_z", "std"),
            n_patients=("Patient_ID", "nunique"),
        )
        .reset_index()
    )
    summary["n_positive_patients"] = paired.groupby("metabolite", observed=True)[
        "fibrotic_minus_adjacent_z"
    ].apply(lambda values: int((values > 0).sum())).values
    summary["n_negative_patients"] = paired.groupby("metabolite", observed=True)[
        "fibrotic_minus_adjacent_z"
    ].apply(lambda values: int((values < 0).sum())).values
    summary["positive_patient_fraction"] = summary["n_positive_patients"] / summary["n_patients"]
    return summary.sort_values("mean_fibrotic_minus_adjacent_z", ascending=False)


def plot_mean_differences(summary: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    plot_df = pd.concat(
        [
            summary.nlargest(top_n, "mean_fibrotic_minus_adjacent_z"),
            summary.nsmallest(top_n, "mean_fibrotic_minus_adjacent_z"),
        ],
        ignore_index=True,
    ).drop_duplicates("metabolite")
    plot_df = plot_df.sort_values("mean_fibrotic_minus_adjacent_z")
    colors = np.where(plot_df["mean_fibrotic_minus_adjacent_z"] >= 0, "#C44E52", "#4C72B0")

    plt.figure(figsize=(10, 11))
    plt.barh(plot_df["metabolite"], plot_df["mean_fibrotic_minus_adjacent_z"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("All paired patients: mean Harreman z-score difference")
    plt.xlabel("mean fibrotic minus adjacent z score")
    plt.ylabel("metabolite")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_patient_heatmap(paired: pd.DataFrame, summary: pd.DataFrame, output_path: Path, top_n: int = 30) -> None:
    selected = pd.concat(
        [
            summary.nlargest(top_n // 2, "mean_fibrotic_minus_adjacent_z"),
            summary.nsmallest(top_n // 2, "mean_fibrotic_minus_adjacent_z"),
        ],
        ignore_index=True,
    )["metabolite"]
    heatmap = paired[paired["metabolite"].isin(selected)].pivot_table(
        index="metabolite",
        columns="Patient_ID",
        values="fibrotic_minus_adjacent_z",
        aggfunc="mean",
    )
    heatmap = heatmap.loc[
        summary[summary["metabolite"].isin(heatmap.index)]
        .sort_values("mean_fibrotic_minus_adjacent_z")["metabolite"]
    ]
    vmax = np.nanquantile(np.abs(heatmap.to_numpy()), 0.95)
    plt.figure(figsize=(9, 10))
    sns.heatmap(heatmap, cmap="vlag", center=0, vmin=-vmax, vmax=vmax, linewidths=0.2)
    plt.title("Paired Harreman z-score differences by patient")
    plt.xlabel("patient")
    plt.ylabel("metabolite")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_runtime(run_summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=run_summary,
        x="n_spots",
        y="runtime_seconds",
        hue="Disease Label",
        size="n_gene_pairs",
        sizes=(35, 180),
    )
    plt.title("Harreman runtime by slide")
    plt.xlabel("spots")
    plt.ylabel("runtime, seconds")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--expression-threshold", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="CUDA initialization")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="harreman")
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_theme(style="whitegrid")
    patch_harreman_readonly_diagonal()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = paired_slide_metadata()
    metadata.to_csv(output_dir / "paired_slide_metadata.csv", index=False)

    metabolite_tables = []
    gene_pair_tables = []
    run_rows = []
    for _, metadata_row in metadata.iterrows():
        sample = metadata_row["sample"]
        print(f"Running/loading Harreman for {sample}")
        metabolites, gene_pairs, summary = run_or_load_sample(
            sample=sample,
            metadata_row=metadata_row,
            output_dir=output_dir,
            n_neighbors=args.n_neighbors,
            expression_threshold=args.expression_threshold,
            force=args.force,
        )
        metabolite_tables.append(metabolites)
        gene_pair_tables.append(gene_pairs)
        run_rows.append(summary)

    metabolites = pd.concat(metabolite_tables, ignore_index=True)
    gene_pairs = pd.concat(gene_pair_tables, ignore_index=True)
    run_summary = pd.DataFrame(run_rows)

    metabolites.to_csv(output_dir / "harreman_all_metabolite_results.csv", index=False)
    gene_pairs.to_csv(output_dir / "harreman_all_gene_pair_results.csv", index=False)
    run_summary.to_csv(output_dir / "harreman_all_run_summary.csv", index=False)

    paired = compute_paired_differences(metabolites)
    paired.to_csv(output_dir / "harreman_patient_paired_metabolite_differences.csv", index=False)
    difference_summary = summarize_differences(paired)
    difference_summary.to_csv(output_dir / "harreman_metabolite_difference_summary.csv", index=False)

    plot_mean_differences(difference_summary, output_dir / "mean_paired_metabolite_differences.png")
    plot_patient_heatmap(paired, difference_summary, output_dir / "paired_difference_heatmap.png")
    plot_runtime(run_summary, output_dir / "harreman_runtime_by_slide.png")

    overview = {
        "n_slides": int(run_summary["sample"].nunique()),
        "n_patients": int(run_summary["Patient_ID"].nunique()),
        "n_adjacent_slides": int(run_summary["Disease Label"].eq("Adjacent").sum()),
        "n_fibrotic_slides": int(run_summary["Disease Label"].eq("Fibrotic").sum()),
        "n_metabolites_with_paired_results": int(difference_summary["metabolite"].nunique()),
        "mean_runtime_seconds": float(run_summary["runtime_seconds"].mean()),
        "total_runtime_seconds": float(run_summary["runtime_seconds"].sum()),
        "top_fibrotic_shifted_metabolites": difference_summary.head(15)[
            [
                "metabolite",
                "mean_fibrotic_minus_adjacent_z",
                "n_patients",
                "positive_patient_fraction",
            ]
        ].to_dict(orient="records"),
        "top_adjacent_shifted_metabolites": difference_summary.tail(15)[
            [
                "metabolite",
                "mean_fibrotic_minus_adjacent_z",
                "n_patients",
                "positive_patient_fraction",
            ]
        ].sort_values("mean_fibrotic_minus_adjacent_z").to_dict(orient="records"),
    }
    with (output_dir / "harreman_all_slides_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
