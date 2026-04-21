"""Deconvolve spatial spots with the provided CIBERSORTx IBD Atlas signature matrix."""

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
from joblib import Parallel, delayed
from scipy.optimize import nnls


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "spatial_deconvolution"
SIGNATURE_PATH = DATA_DIR / "ibd_atlas_signature_matrix.txt"
SOURCE_GEP_PATH = DATA_DIR / "ibd_atlas_cell_type_sourceGEP.txt"


DEFAULT_MAP_CELL_TYPES = [
    "Absorptive",
    "Secretory",
    "Epithelial_stem",
    "Transit_amplifying",
    "Fibroblast",
    "Myofibroblast",
    "Macrophage",
    "Monocyte",
    "Conventional_CD4",
    "Conventional_CD8",
    "Treg",
    "Vascular_endothelia",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_signature() -> pd.DataFrame:
    signature = pd.read_csv(SIGNATURE_PATH, sep="\t")
    signature = signature.rename(columns={"NAME": "gene"})
    if signature["gene"].duplicated().any():
        signature = signature.groupby("gene", as_index=False).mean(numeric_only=True)
    return signature


def select_common_genes(
    signature: pd.DataFrame,
    spatial_genes: pd.Index,
    max_genes: int,
) -> tuple[list[str], pd.DataFrame, int]:
    signature = signature[signature["gene"].isin(spatial_genes)].copy()
    n_common = int(signature["gene"].nunique())
    cell_type_columns = signature.columns.drop("gene")
    signature["signature_variance"] = np.log1p(signature[cell_type_columns]).var(axis=1)
    signature = signature.sort_values("signature_variance", ascending=False)
    if max_genes > 0:
        signature = signature.head(max_genes)
    return signature["gene"].tolist(), signature, n_common


def solve_one_spot(y: np.ndarray, signature_matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    coef, residual = nnls(signature_matrix, y)
    total = coef.sum()
    if total > 0:
        proportions = coef / total
    else:
        proportions = np.zeros_like(coef)
    denom = float(np.linalg.norm(y))
    relative_residual = float(residual / denom) if denom > 0 else np.nan
    return proportions, float(residual), relative_residual


def run_nnls(
    expression_matrix,
    signature_matrix: np.ndarray,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = (expression_matrix[i, :].toarray().ravel() for i in range(expression_matrix.shape[0]))
    results = Parallel(n_jobs=n_jobs, batch_size=128, prefer="processes")(
        delayed(solve_one_spot)(row, signature_matrix) for row in rows
    )
    proportions = np.vstack([item[0] for item in results])
    residuals = np.array([item[1] for item in results])
    relative_residuals = np.array([item[2] for item in results])
    return proportions, residuals, relative_residuals


def load_spatial_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["sample"] = metadata["Visium Slide ID"].astype(str)
    return metadata


def summarize_proportions(spots: pd.DataFrame, cell_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_summary = (
        spots.groupby(["sample", "Disease Label", "Patient_ID", "General Categorization"], observed=True)[
            cell_types
        ]
        .mean()
        .reset_index()
    )
    disease_summary = (
        sample_summary.groupby("Disease Label", observed=True)[cell_types]
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


def compute_paired_differences(sample_summary: pd.DataFrame, cell_types: list[str]) -> pd.DataFrame:
    rows = []
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
        for cell_type in cell_types:
            adjacent_mean = float(adjacent[cell_type].mean())
            fibrotic_mean = float(fibrotic[cell_type].mean())
            row[f"{cell_type}_adjacent_mean"] = adjacent_mean
            row[f"{cell_type}_fibrotic_mean"] = fibrotic_mean
            row[f"{cell_type}_fibrotic_minus_adjacent"] = fibrotic_mean - adjacent_mean
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sample_heatmap(sample_summary: pd.DataFrame, cell_types: list[str], output_path: Path) -> None:
    heatmap = sample_summary.copy()
    heatmap["label"] = heatmap["sample"] + " | " + heatmap["Disease Label"]
    heatmap = heatmap.set_index("label")[cell_types]
    heatmap = heatmap.loc[:, heatmap.mean().sort_values(ascending=False).index]

    plt.figure(figsize=(13, 8))
    sns.heatmap(heatmap, cmap="viridis", linewidths=0.2, linecolor="white")
    plt.title("Mean deconvolved cell-type proportions by slide")
    plt.xlabel("cell type")
    plt.ylabel("slide")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_paired_differences(paired: pd.DataFrame, cell_types: list[str], output_path: Path) -> None:
    rows = []
    for cell_type in cell_types:
        column = f"{cell_type}_fibrotic_minus_adjacent"
        if column in paired:
            rows.append(
                {
                    "cell_type": cell_type,
                    "mean_fibrotic_minus_adjacent": paired[column].mean(),
                    "positive_patient_fraction": (paired[column] > 0).mean(),
                }
            )
    plot_df = pd.DataFrame(rows).sort_values("mean_fibrotic_minus_adjacent")
    colors = np.where(plot_df["mean_fibrotic_minus_adjacent"] >= 0, "#C44E52", "#4C72B0")

    plt.figure(figsize=(9, 11))
    plt.barh(plot_df["cell_type"], plot_df["mean_fibrotic_minus_adjacent"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Mean paired fibrotic-minus-adjacent cell-type proportion")
    plt.xlabel("difference in deconvolved proportion")
    plt.ylabel("cell type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_all_slide_maps(spots: pd.DataFrame, cell_type: str, output_path: Path) -> None:
    samples = sorted(spots["sample"].unique())
    n_cols = 5
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    axes = axes.flatten()
    upper = np.nanquantile(spots[cell_type], 0.98)
    for ax, sample in zip(axes, samples):
        slide = spots[spots["sample"].eq(sample)]
        im = ax.scatter(
            slide["pxl_col_in_fullres"],
            slide["pxl_row_in_fullres"],
            c=slide[cell_type],
            cmap="magma",
            s=1.4,
            linewidths=0,
            vmin=0,
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
    fig.suptitle(f"{cell_type} deconvolved proportion", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    fig.colorbar(im, cax=cbar_ax, label="proportion")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-genes", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    output_dir = args.output_dir
    maps_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    signature = load_signature()
    adata = ad.read_h5ad(DATA_DIR / "anndata.h5ad")
    metadata = load_spatial_metadata()

    try:
        common_genes, selected_signature, n_common_signature_spatial_genes = select_common_genes(
            signature,
            spatial_genes=adata.var_names,
            max_genes=args.max_genes,
        )
        cell_types = selected_signature.columns.drop(["gene", "signature_variance"]).tolist()
        selected_signature.to_csv(output_dir / "selected_signature_genes.csv", index=False)

        gene_indices = [adata.var_names.get_loc(gene) for gene in common_genes]
        expression = adata.X[:, gene_indices].tocsr()
        signature_matrix = np.log1p(selected_signature[cell_types].to_numpy(dtype=float))

        proportions, residuals, relative_residuals = run_nnls(
            expression,
            signature_matrix=signature_matrix,
            n_jobs=args.n_jobs,
        )

        coords = pd.DataFrame(
            adata.obsm["spatial"],
            index=adata.obs_names,
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
        spots = adata.obs[["sample"]].copy()
    finally:
        del adata

    proportions_df = pd.DataFrame(proportions, columns=cell_types, index=spots.index)
    spots = spots.join(coords).join(proportions_df)
    spots["nnls_residual"] = residuals
    spots["nnls_relative_residual"] = relative_residuals
    spots = spots.merge(
        metadata[["sample", "Disease Label", "Patient_ID", "General Categorization"]],
        on="sample",
        how="left",
    )

    spots.to_parquet(output_dir / "spot_cell_type_proportions.parquet", index=False)
    spots[
        [
            "sample",
            "Disease Label",
            "Patient_ID",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
            "nnls_relative_residual",
            *cell_types,
        ]
    ].to_csv(output_dir / "spot_cell_type_proportions_selected.csv", index=False)

    sample_summary, disease_summary = summarize_proportions(spots, cell_types)
    paired = compute_paired_differences(sample_summary, cell_types)
    sample_summary.to_csv(output_dir / "sample_cell_type_proportion_summary.csv", index=False)
    disease_summary.to_csv(output_dir / "disease_label_cell_type_proportion_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_patient_cell_type_differences.csv", index=False)

    fit_summary = pd.DataFrame(
        {
            "metric": [
                "n_spots",
                "n_cell_types",
                "n_signature_genes_total",
                "n_common_signature_spatial_genes",
                "n_selected_genes",
                "median_relative_residual",
                "mean_relative_residual",
            ],
            "value": [
                len(spots),
                len(cell_types),
                int(signature["gene"].nunique()),
                n_common_signature_spatial_genes,
                len(common_genes),
                float(np.nanmedian(relative_residuals)),
                float(np.nanmean(relative_residuals)),
            ],
        }
    )
    fit_summary.to_csv(output_dir / "deconvolution_fit_summary.csv", index=False)

    overlap = pd.DataFrame(
        {
            "source": ["signature_matrix", "source_GEP", "selected_signature", "spatial_common"],
            "n_genes": [
                int(signature["gene"].nunique()),
                int(pd.read_csv(SOURCE_GEP_PATH, sep="\t", usecols=[0]).iloc[:, 0].nunique()),
                int(len(common_genes)),
                n_common_signature_spatial_genes,
            ],
        }
    )
    overlap.to_csv(output_dir / "signature_gene_overlap_summary.csv", index=False)

    plot_sample_heatmap(sample_summary, cell_types, output_dir / "sample_cell_type_heatmap.png")
    plot_paired_differences(paired, cell_types, output_dir / "paired_cell_type_differences.png")
    for cell_type in [ct for ct in DEFAULT_MAP_CELL_TYPES if ct in cell_types]:
        plot_all_slide_maps(spots, cell_type, maps_dir / f"{safe_name(cell_type)}_all_slides.png")

    paired_means = []
    for cell_type in cell_types:
        column = f"{cell_type}_fibrotic_minus_adjacent"
        paired_means.append(
            {
                "cell_type": cell_type,
                "mean_paired_fibrotic_minus_adjacent": float(paired[column].mean()),
                "positive_patient_fraction": float((paired[column] > 0).mean()),
            }
        )
    paired_mean_df = pd.DataFrame(paired_means).sort_values(
        "mean_paired_fibrotic_minus_adjacent",
        ascending=False,
    )
    paired_mean_df.to_csv(output_dir / "paired_cell_type_difference_summary.csv", index=False)

    overview = {
        "n_spots": int(len(spots)),
        "n_cell_types": int(len(cell_types)),
        "n_signature_genes": int(signature["gene"].nunique()),
        "n_common_signature_spatial_genes": n_common_signature_spatial_genes,
        "n_selected_genes": int(len(common_genes)),
        "median_relative_residual": float(np.nanmedian(relative_residuals)),
        "top_fibrotic_enriched_cell_types": paired_mean_df.head(10).to_dict(orient="records"),
        "top_adjacent_enriched_cell_types": paired_mean_df.tail(10)
        .sort_values("mean_paired_fibrotic_minus_adjacent")
        .to_dict(orient="records"),
    }
    with (output_dir / "spatial_deconvolution_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
