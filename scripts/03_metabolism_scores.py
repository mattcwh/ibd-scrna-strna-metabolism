"""Compute first-pass spatial metabolism and fibrosis gene-set scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "metabolism_scores"


GENE_SETS: dict[str, list[str]] = {
    "glycolysis": [
        "HK1",
        "HK2",
        "GPI",
        "PFKP",
        "PFKL",
        "ALDOA",
        "GAPDH",
        "PGK1",
        "PGAM1",
        "ENO1",
        "PKM",
        "LDHA",
        "SLC2A1",
        "SLC16A3",
    ],
    "oxidative_phosphorylation": [
        "NDUFA1",
        "NDUFA2",
        "NDUFB1",
        "NDUFB8",
        "SDHA",
        "SDHB",
        "UQCRC1",
        "UQCRC2",
        "COX4I1",
        "COX5A",
        "ATP5F1A",
        "ATP5F1B",
        "ATP5MC1",
    ],
    "hypoxia": [
        "HIF1A",
        "VEGFA",
        "CA9",
        "ENO1",
        "LDHA",
        "SLC2A1",
        "PGK1",
        "BNIP3",
        "NDRG1",
        "P4HA1",
    ],
    "retinoid_vitamin_a": [
        "ALDH1A1",
        "ALDH1A2",
        "ALDH1A3",
        "RBP1",
        "RBP4",
        "STRA6",
        "CRABP2",
        "CYP26A1",
        "CYP26B1",
        "RXRA",
        "RARA",
    ],
    "ecm_remodeling": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL5A1",
        "COL6A1",
        "FN1",
        "POSTN",
        "THY1",
        "ACTA2",
        "TAGLN",
        "MMP2",
        "TIMP1",
    ],
    "inflammation": [
        "IL1B",
        "TNF",
        "IL6",
        "CXCL8",
        "CCL2",
        "CXCL10",
        "NFKBIA",
        "PTGS2",
        "S100A8",
        "S100A9",
    ],
    "epithelial_repair": [
        "OLFM4",
        "LGR5",
        "MKI67",
        "REG1A",
        "REG3A",
        "REG3G",
        "SPINK4",
        "TFF3",
        "MUC2",
        "KRT8",
        "KRT18",
    ],
    "fatty_acid_metabolism": [
        "CPT1A",
        "ACADM",
        "ACADVL",
        "HADHA",
        "HADHB",
        "ACOX1",
        "PPARA",
        "CD36",
        "FABP1",
        "FABP2",
        "SLC27A4",
    ],
    "bile_acid_transport": [
        "SLC10A2",
        "SLC51A",
        "SLC51B",
        "FABP6",
        "NR1H4",
        "CYP7A1",
        "CYP27A1",
        "ABCB11",
        "ABCC3",
        "ABCG5",
        "ABCG8",
    ],
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def mean_expression(matrix, gene_indices: list[int]) -> np.ndarray:
    subset = matrix[:, gene_indices]
    means = subset.mean(axis=1)
    if sparse.issparse(means):
        means = means.A1
    else:
        means = np.asarray(means).ravel()
    return means.astype(float)


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["sample"] = metadata["Visium Slide ID"].astype(str)
    return metadata


def compute_scores(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    var_names = pd.Index(adata.var_names.astype(str))
    upper_to_gene = {gene.upper(): gene for gene in var_names}

    score_df = pd.DataFrame(index=adata.obs_names)
    membership_rows = []
    for set_name, requested_genes in GENE_SETS.items():
        matched_genes = [upper_to_gene[gene.upper()] for gene in requested_genes if gene.upper() in upper_to_gene]
        missing_genes = [gene for gene in requested_genes if gene.upper() not in upper_to_gene]
        for gene in matched_genes:
            membership_rows.append(
                {
                    "gene_set": set_name,
                    "gene": gene,
                    "status": "matched",
                }
            )
        for gene in missing_genes:
            membership_rows.append(
                {
                    "gene_set": set_name,
                    "gene": gene,
                    "status": "missing",
                }
            )
        if not matched_genes:
            score_df[f"{set_name}_mean"] = np.nan
            score_df[f"{set_name}_z"] = np.nan
            continue

        gene_indices = [var_names.get_loc(gene) for gene in matched_genes]
        raw_score = mean_expression(adata.X, gene_indices)
        score_df[f"{set_name}_mean"] = raw_score
        score_df[f"{set_name}_z"] = zscore(raw_score)

    membership = pd.DataFrame(membership_rows)
    return score_df, membership


def add_spatial_metadata(adata: ad.AnnData, scores: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    coords = pd.DataFrame(
        adata.obsm["spatial"],
        index=adata.obs_names,
        columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
    )
    obs = adata.obs[["sample"]].copy()
    result = obs.join(coords).join(scores)
    result = result.merge(
        metadata[
            [
                "sample",
                "Disease Label",
                "General Categorization",
                "Patient_ID",
                "Shorthand_ID",
                "Spots",
            ]
        ],
        on="sample",
        how="left",
    )
    return result


def summarize_scores(spots: pd.DataFrame, score_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_summary = (
        spots.groupby(
            ["sample", "Disease Label", "General Categorization", "Patient_ID", "Shorthand_ID"],
            observed=True,
        )[score_names]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    sample_summary.columns = [
        "_".join([str(part) for part in column if part]).strip("_")
        if isinstance(column, tuple)
        else column
        for column in sample_summary.columns
    ]

    disease_summary = (
        sample_summary.groupby("Disease Label", observed=True)[
            [f"{score}_mean" for score in score_names]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    disease_summary.columns = [
        "_".join([str(part) for part in column if part]).strip("_")
        if isinstance(column, tuple)
        else column
        for column in disease_summary.columns
    ]
    return sample_summary, disease_summary


def compute_paired_patient_differences(
    sample_summary: pd.DataFrame,
    score_sets: list[str],
) -> pd.DataFrame:
    rows = []
    score_columns = [f"{score}_z_mean" for score in score_sets]
    for patient_id, group in sample_summary.groupby("Patient_ID", observed=True):
        adjacent = group[group["Disease Label"].eq("Adjacent")]
        fibrotic = group[group["Disease Label"].eq("Fibrotic")]
        if adjacent.empty or fibrotic.empty:
            continue

        row: dict[str, object] = {
            "Patient_ID": patient_id,
            "n_adjacent_slides": int(len(adjacent)),
            "n_fibrotic_slides": int(len(fibrotic)),
            "adjacent_samples": ";".join(adjacent["sample"].astype(str)),
            "fibrotic_samples": ";".join(fibrotic["sample"].astype(str)),
        }
        for score, column in zip(score_sets, score_columns):
            adjacent_mean = float(adjacent[column].mean())
            fibrotic_mean = float(fibrotic[column].mean())
            row[f"{score}_adjacent_mean"] = adjacent_mean
            row[f"{score}_fibrotic_mean"] = fibrotic_mean
            row[f"{score}_fibrotic_minus_adjacent"] = fibrotic_mean - adjacent_mean
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sample_heatmap(sample_summary: pd.DataFrame, score_sets: list[str], output_path: Path) -> None:
    mean_cols = [f"{score}_z_mean" for score in score_sets]
    labels = sample_summary["sample"] + " | " + sample_summary["Disease Label"]
    heatmap = sample_summary.set_index(labels)[mean_cols]
    heatmap.columns = score_sets
    heatmap = heatmap.sort_index()

    plt.figure(figsize=(11, 8))
    sns.heatmap(heatmap, cmap="vlag", center=0, linewidths=0.2, linecolor="white")
    plt.title("Mean spot-level z scores by slide")
    plt.xlabel("gene set")
    plt.ylabel("slide")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_disease_boxplots(
    sample_summary: pd.DataFrame,
    score_sets: list[str],
    output_path: Path,
) -> None:
    rows = []
    for score in score_sets:
        column = f"{score}_z_mean"
        subset = sample_summary[["sample", "Disease Label", column]].copy()
        subset = subset.rename(columns={column: "mean_z_score"})
        subset["gene_set"] = score
        rows.append(subset)
    long = pd.concat(rows, ignore_index=True)

    disease_labels = sorted(long["Disease Label"].dropna().unique())
    point_palette = {label: "#222222" for label in disease_labels}

    plt.figure(figsize=(13, 6))
    sns.boxplot(data=long, x="gene_set", y="mean_z_score", hue="Disease Label", fliersize=0)
    sns.stripplot(
        data=long,
        x="gene_set",
        y="mean_z_score",
        hue="Disease Label",
        dodge=True,
        palette=point_palette,
        alpha=0.45,
        size=2.8,
        legend=False,
    )
    plt.axhline(0, color="#666666", linewidth=0.8)
    plt.title("Slide-level mean metabolism scores by disease label")
    plt.xlabel("gene set")
    plt.ylabel("mean spot-level z score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_all_slide_maps(
    spots: pd.DataFrame,
    score: str,
    output_path: Path,
) -> None:
    column = f"{score}_z"
    samples = sorted(spots["sample"].unique())
    n_cols = 5
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    axes = axes.flatten()

    lower, upper = np.nanquantile(spots[column], [0.02, 0.98])
    for ax, sample in zip(axes, samples):
        slide = spots[spots["sample"].eq(sample)]
        im = ax.scatter(
            slide["pxl_col_in_fullres"],
            slide["pxl_row_in_fullres"],
            c=slide[column],
            cmap="viridis",
            s=1.4,
            linewidths=0,
            vmin=lower,
            vmax=upper,
        )
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        label = slide["Disease Label"].iloc[0]
        ax.set_title(f"{sample}\n{label}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.suptitle(f"{score} spot-level z score", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    fig.colorbar(im, cax=cbar_ax, label="z score")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    maps_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    metadata = load_metadata()
    adata = ad.read_h5ad(DATA_DIR / "anndata.h5ad")
    try:
        scores, membership = compute_scores(adata)
        spots = add_spatial_metadata(adata, scores, metadata)
    finally:
        del adata

    score_sets = list(GENE_SETS)
    z_score_columns = [f"{score}_z" for score in score_sets]
    membership.to_csv(output_dir / "gene_set_membership.csv", index=False)
    spots.to_parquet(output_dir / "spot_metabolism_scores.parquet", index=False)
    spots[
        [
            "sample",
            "Disease Label",
            "Patient_ID",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
            *z_score_columns,
        ]
    ].to_csv(output_dir / "spot_metabolism_scores_selected.csv", index=False)

    sample_summary, disease_summary = summarize_scores(spots, z_score_columns)
    paired_differences = compute_paired_patient_differences(sample_summary, score_sets)
    sample_summary.to_csv(output_dir / "sample_metabolism_score_summary.csv", index=False)
    disease_summary.to_csv(output_dir / "disease_label_metabolism_score_summary.csv", index=False)
    paired_differences.to_csv(output_dir / "paired_patient_metabolism_score_differences.csv", index=False)

    plot_sample_heatmap(sample_summary, score_sets, output_dir / "sample_mean_score_heatmap.png")
    plot_disease_boxplots(sample_summary, score_sets, output_dir / "disease_label_score_boxplots.png")
    for score in score_sets:
        plot_all_slide_maps(spots, score, maps_dir / f"{safe_name(score)}_all_slides.png")

    overview = {
        "n_spots": int(len(spots)),
        "n_slides": int(spots["sample"].nunique()),
        "gene_sets": score_sets,
        "matched_genes_by_set": membership[membership["status"].eq("matched")]
        .groupby("gene_set")
        .size()
        .to_dict(),
        "missing_genes_by_set": membership[membership["status"].eq("missing")]
        .groupby("gene_set")
        .size()
        .to_dict(),
        "outputs": {
            "spot_scores": "spot_metabolism_scores.parquet",
            "sample_summary": "sample_metabolism_score_summary.csv",
            "disease_summary": "disease_label_metabolism_score_summary.csv",
            "paired_differences": "paired_patient_metabolism_score_differences.csv",
        },
        "n_paired_patients": int(len(paired_differences)),
    }
    with (output_dir / "metabolism_scores_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
