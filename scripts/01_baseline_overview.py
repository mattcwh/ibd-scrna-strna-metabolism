"""Create first-pass scRNA-seq and spatial metadata summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "baseline_overview"


def save_count_table(obs: pd.DataFrame, columns: list[str], path: Path) -> pd.DataFrame:
    counts = obs.groupby(columns, observed=True).size().reset_index(name="n_cells")
    counts.to_csv(path, index=False)
    return counts


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    title: str,
    output_path: Path,
    rotate: bool = False,
) -> None:
    plt.figure(figsize=(11, 6))
    sns.barplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_hist(obs: pd.DataFrame, column: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(obs[column].dropna(), bins=60)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("n_cells")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_scrna(output_dir: Path) -> None:
    obj = ad.read_h5ad(DATA_DIR / "Cleaned_raw_annotated_object_LK.v2.h5ad", backed="r")
    try:
        obs = obj.obs.copy()
    finally:
        obj.file.close()

    scrna_dir = output_dir / "scrna"
    scrna_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "donor_id",
        "biosample_id",
        "status",
        "fraction",
        "tissue",
        "annotation",
        "annotation2v2",
        "sex",
    ]
    available = [column for column in columns if column in obs.columns]
    obs[available].to_csv(scrna_dir / "cell_metadata_selected_columns.csv")

    for column in available:
        counts = obs[column].value_counts(dropna=False).rename_axis(column).reset_index(name="n_cells")
        counts.to_csv(scrna_dir / f"cell_counts_by_{column}.csv", index=False)

    if {"status", "annotation"}.issubset(obs.columns):
        counts = save_count_table(obs, ["status", "annotation"], scrna_dir / "cell_counts_by_status_annotation.csv")
        plot_bar(
            counts,
            x="annotation",
            y="n_cells",
            hue="status",
            title="scRNA-seq cell counts by status and annotation",
            output_path=scrna_dir / "cell_counts_by_status_annotation.png",
            rotate=True,
        )

    if {"status", "fraction"}.issubset(obs.columns):
        counts = save_count_table(obs, ["status", "fraction"], scrna_dir / "cell_counts_by_status_fraction.csv")
        plot_bar(
            counts,
            x="status",
            y="n_cells",
            hue="fraction",
            title="scRNA-seq cell counts by status and fraction",
            output_path=scrna_dir / "cell_counts_by_status_fraction.png",
        )

    for column in ["total_counts", "pct_counts_mt", "doublet_score"]:
        if column in obs.columns:
            plot_hist(obs, column, scrna_dir / f"hist_{column}.png")


def normalize_slide_id(value: str) -> str:
    return value.replace("-", "_")


def summarize_spatial(output_dir: Path) -> None:
    spatial_dir = output_dir / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["normalized_slide_id"] = metadata["Visium Slide ID"].map(normalize_slide_id)
    metadata.to_csv(spatial_dir / "spatial_slide_metadata.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.countplot(data=metadata, x="Disease Label", order=metadata["Disease Label"].value_counts().index)
    plt.title("Spatial slides by disease label")
    plt.xlabel("Disease Label")
    plt.ylabel("n_slides")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(spatial_dir / "slides_by_disease_label.png", dpi=200)
    plt.close()

    plot_bar(
        metadata,
        x="Visium Slide ID",
        y="Spots",
        hue="Disease Label",
        title="Spatial spots by slide",
        output_path=spatial_dir / "spots_by_slide.png",
        rotate=True,
    )

    obj = ad.read_h5ad(DATA_DIR / "anndata.h5ad", backed="r")
    try:
        obs = obj.obs.copy()
    finally:
        obj.file.close()

    obs.to_csv(spatial_dir / "spot_metadata.csv")
    if "sample" in obs.columns:
        spot_counts = obs["sample"].value_counts().rename_axis("sample").reset_index(name="n_spots")
        spot_counts["normalized_slide_id"] = spot_counts["sample"].map(normalize_slide_id)
        spot_counts = spot_counts.merge(
            metadata[["normalized_slide_id", "Disease Label", "Patient_ID", "Shorthand_ID"]],
            on="normalized_slide_id",
            how="left",
        )
        spot_counts.to_csv(spatial_dir / "spot_counts_by_sample.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize_scrna(args.output_dir)
    summarize_spatial(args.output_dir)
    print(f"Wrote baseline overview outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
