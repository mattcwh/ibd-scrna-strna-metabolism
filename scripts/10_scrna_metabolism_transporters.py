"""Summarize scRNA-seq metabolism scores and transporter genes by cell type."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
METABOLISM_DIR = REPO_ROOT / "output" / "metabolism_scores"
TRANSPORTER_DIR = REPO_ROOT / "output" / "transporter_spatial_validation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "scrna_metabolism_transporters"


TRANSPORTER_PANELS = {
    "epithelial_bile_urea_transport": ["SLC5A1", "SLC10A2", "SLC51A", "SLC51B", "ABCB1", "ABCC3"],
    "fibrotic_amino_acid_transport": ["SLC38A2", "SLC15A3", "SLCO2B1", "SLC7A5", "SLC1A5", "SLC1A4"],
}

METADATA_COLUMNS = [
    "annotation",
    "annotation2v2",
    "status",
    "fraction",
    "tissue",
    "disease",
    "donor_id",
    "biosample_id",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_gene_sets(spatial_genes: pd.Index) -> dict[str, list[str]]:
    membership = pd.read_csv(METABOLISM_DIR / "gene_set_membership.csv")
    matched = membership[membership["status"].eq("matched")].copy()
    gene_sets = {}
    for gene_set, group in matched.groupby("gene_set", observed=True):
        genes = [gene for gene in group["gene"].astype(str).unique() if gene in spatial_genes]
        if genes:
            gene_sets[gene_set] = genes
    return gene_sets


def load_transporter_genes(spatial_genes: pd.Index) -> list[str]:
    gene_map = pd.read_csv(TRANSPORTER_DIR / "focus_metabolite_transporter_gene_map.csv")
    mapped = gene_map.loc[gene_map["present_in_spatial"], "gene"].astype(str).unique().tolist()
    panel_genes = [gene for genes in TRANSPORTER_PANELS.values() for gene in genes]
    genes = sorted({gene for gene in mapped + panel_genes if gene in spatial_genes})
    return genes


def read_csr_rows(h5: h5py.File, start: int, end: int, n_vars: int) -> sparse.csr_matrix:
    x_group = h5["X"]
    indptr = x_group["indptr"][start : end + 1].astype(np.int64)
    data_start = int(indptr[0])
    data_end = int(indptr[-1])
    local_indptr = indptr - data_start
    data = x_group["data"][data_start:data_end]
    indices = x_group["indices"][data_start:data_end]
    return sparse.csr_matrix((data, indices, local_indptr), shape=(end - start, n_vars))


def mean_expression(matrix: sparse.csr_matrix, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.full(matrix.shape[0], np.nan, dtype=np.float32)
    values = matrix[:, indices].mean(axis=1)
    return np.asarray(values).ravel().astype(np.float32)


def compute_features(
    h5ad_path: Path,
    gene_sets: dict[str, list[str]],
    transporter_genes: list[str],
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        var_names = pd.Index(backed.var_names.astype(str))
        n_cells, n_vars = backed.shape
    finally:
        backed.file.close()

    gene_to_idx = {gene: var_names.get_loc(gene) for gene in var_names}
    gene_sets = {
        name: [gene for gene in genes if gene in gene_to_idx]
        for name, genes in gene_sets.items()
    }
    gene_sets = {name: genes for name, genes in gene_sets.items() if genes}
    transporter_genes = [gene for gene in transporter_genes if gene in gene_to_idx]

    feature_names = [f"{name}_mean" for name in gene_sets]
    feature_names.extend([f"{name}_mean" for name in TRANSPORTER_PANELS])
    feature_names.extend([f"{gene}_expression" for gene in transporter_genes])
    features = {name: np.zeros(n_cells, dtype=np.float32) for name in feature_names}

    with h5py.File(h5ad_path, "r") as h5:
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            matrix = read_csr_rows(h5, start, end, n_vars)
            for gene_set, genes in gene_sets.items():
                indices = [gene_to_idx[gene] for gene in genes]
                features[f"{gene_set}_mean"][start:end] = mean_expression(matrix, indices)
            for panel, genes in TRANSPORTER_PANELS.items():
                indices = [gene_to_idx[gene] for gene in genes if gene in gene_to_idx]
                features[f"{panel}_mean"][start:end] = mean_expression(matrix, indices)
            for gene in transporter_genes:
                values = matrix[:, gene_to_idx[gene]]
                if sparse.issparse(values):
                    values = values.toarray().ravel()
                else:
                    values = np.asarray(values).ravel()
                features[f"{gene}_expression"][start:end] = values.astype(np.float32)

    feature_df = pd.DataFrame(features)
    return feature_df, gene_sets, transporter_genes


def load_cell_metadata(h5ad_path: Path) -> pd.DataFrame:
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        metadata = backed.obs[METADATA_COLUMNS].copy()
    finally:
        backed.file.close()
    for column in metadata.columns:
        metadata[column] = metadata[column].astype(str)
    return metadata.reset_index(names="cell_id")


def summarize_by_group(cells: pd.DataFrame, group_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    grouped = cells.groupby(group_cols, observed=True)
    summary = grouped[feature_cols].mean().reset_index()
    counts = grouped.size().reset_index(name="n_cells")
    return counts.merge(summary, on=group_cols, how="left")


def transporter_long_summary(cells: pd.DataFrame, transporter_genes: list[str]) -> pd.DataFrame:
    rows = []
    expression_cols = [f"{gene}_expression" for gene in transporter_genes]
    grouped = cells.groupby(["annotation2v2", "status", "fraction"], observed=True)
    for keys, group in grouped:
        annotation2v2, status, fraction = keys
        for gene, column in zip(transporter_genes, expression_cols, strict=True):
            values = group[column]
            rows.append(
                {
                    "annotation2v2": annotation2v2,
                    "status": status,
                    "fraction": fraction,
                    "gene": gene,
                    "n_cells": int(len(group)),
                    "mean_expression": float(values.mean()),
                    "fraction_expressing": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def status_differences(summary: pd.DataFrame, id_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in summary.groupby(id_cols, observed=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(id_cols, key_values, strict=True))
        normal = group[group["status"].eq("N")]
        fibrotic = group[group["status"].eq("F")]
        inflamed = group[group["status"].eq("I")]
        if normal.empty:
            continue
        for feature in feature_cols:
            row = base | {"feature": feature, "normal_mean": float(normal[feature].mean())}
            if not fibrotic.empty:
                row["fibrotic_mean"] = float(fibrotic[feature].mean())
                row["fibrotic_minus_normal"] = row["fibrotic_mean"] - row["normal_mean"]
            if not inflamed.empty:
                row["inflamed_mean"] = float(inflamed[feature].mean())
                row["inflamed_minus_normal"] = row["inflamed_mean"] - row["normal_mean"]
            rows.append(row)
    return pd.DataFrame(rows)


def top_cell_types_for_features(summary: pd.DataFrame, feature_cols: list[str], min_cells: int) -> pd.DataFrame:
    filtered = summary[summary["n_cells"] >= min_cells].copy()
    rows = []
    for feature in feature_cols:
        top = filtered.sort_values(feature, ascending=False).head(15)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append(
                {
                    "feature": feature,
                    "rank": rank,
                    "annotation2v2": row["annotation2v2"],
                    "fraction": row["fraction"],
                    "n_cells": int(row["n_cells"]),
                    "mean_value": float(row[feature]),
                }
            )
    return pd.DataFrame(rows)


def plot_metabolism_heatmap(summary: pd.DataFrame, metabolism_cols: list[str], output_path: Path) -> None:
    plot_df = summary[summary["n_cells"] >= 300].copy()
    plot_df = plot_df.sort_values("n_cells", ascending=False).head(35)
    heatmap = plot_df.set_index("annotation2v2")[metabolism_cols]
    heatmap = heatmap.loc[:, heatmap.mean().sort_values(ascending=False).index]
    standardized = (heatmap - heatmap.mean(axis=0)) / heatmap.std(axis=0, ddof=0).replace(0, np.nan)
    plt.figure(figsize=(11, 10))
    sns.heatmap(standardized.fillna(0), cmap="vlag", center=0, linewidths=0.2, linecolor="white")
    plt.title("scRNA-seq metabolism scores by cell type")
    plt.xlabel("gene set")
    plt.ylabel("annotation2v2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_transporter_panel_heatmap(summary: pd.DataFrame, panel_cols: list[str], output_path: Path) -> None:
    plot_df = summary[summary["n_cells"] >= 300].copy()
    plot_df["panel_max"] = plot_df[panel_cols].max(axis=1)
    plot_df = plot_df.sort_values("panel_max", ascending=False).head(35)
    heatmap = plot_df.set_index("annotation2v2")[panel_cols]
    standardized = (heatmap - heatmap.mean(axis=0)) / heatmap.std(axis=0, ddof=0).replace(0, np.nan)
    plt.figure(figsize=(7, 10))
    sns.heatmap(standardized.fillna(0), cmap="vlag", center=0, linewidths=0.2, linecolor="white")
    plt.title("scRNA-seq transporter panel scores by cell type")
    plt.xlabel("transporter panel")
    plt.ylabel("annotation2v2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_status_differences(differences: pd.DataFrame, output_path: Path) -> None:
    plot_df = differences.dropna(subset=["fibrotic_minus_normal"]).copy()
    keep = [
        "bile_acid_transport_mean",
        "epithelial_repair_mean",
        "ecm_remodeling_mean",
        "inflammation_mean",
        "epithelial_bile_urea_transport_mean",
        "fibrotic_amino_acid_transport_mean",
    ]
    plot_df = plot_df[plot_df["feature"].isin(keep)]
    plot_df["abs_diff"] = plot_df["fibrotic_minus_normal"].abs()
    plot_df = plot_df.sort_values("abs_diff", ascending=False).head(35)
    plot_df["label"] = plot_df["annotation2v2"] + " | " + plot_df["feature"]
    plot_df = plot_df.sort_values("fibrotic_minus_normal")
    colors = np.where(plot_df["fibrotic_minus_normal"] >= 0, "#C44E52", "#4C72B0")
    plt.figure(figsize=(11, 10))
    plt.barh(plot_df["label"], plot_df["fibrotic_minus_normal"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("scRNA-seq fibrotic-minus-normal feature differences")
    plt.xlabel("mean expression score difference")
    plt.ylabel("cell type and feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--min-cells", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    h5ad_path = DATA_DIR / "Cleaned_raw_annotated_object_LK.v2.h5ad"
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        var_names = pd.Index(backed.var_names.astype(str))
        shape = tuple(backed.shape)
    finally:
        backed.file.close()

    gene_sets = load_gene_sets(var_names)
    transporter_genes = load_transporter_genes(var_names)
    features, matched_gene_sets, transporter_genes = compute_features(
        h5ad_path,
        gene_sets,
        transporter_genes,
        args.chunk_size,
    )
    metadata = load_cell_metadata(h5ad_path)
    cells = pd.concat([metadata, features], axis=1)

    metabolism_cols = [f"{name}_mean" for name in matched_gene_sets]
    panel_cols = [f"{name}_mean" for name in TRANSPORTER_PANELS]
    transporter_cols = [f"{gene}_expression" for gene in transporter_genes]
    feature_cols = metabolism_cols + panel_cols

    cell_type_summary = summarize_by_group(cells, ["annotation2v2", "fraction"], feature_cols)
    status_summary = summarize_by_group(cells, ["annotation2v2", "status", "fraction"], feature_cols)
    donor_summary = summarize_by_group(cells, ["donor_id", "status", "annotation2v2", "fraction"], feature_cols)
    transporter_summary = transporter_long_summary(cells, transporter_genes)
    differences = status_differences(status_summary, ["annotation2v2", "fraction"], feature_cols)
    top_features = top_cell_types_for_features(cell_type_summary, feature_cols, args.min_cells)

    gene_set_membership = pd.DataFrame(
        [
            {"feature": f"{name}_mean", "gene": gene}
            for name, genes in matched_gene_sets.items()
            for gene in genes
        ]
        + [
            {"feature": f"{name}_mean", "gene": gene}
            for name, genes in TRANSPORTER_PANELS.items()
            for gene in genes
            if gene in var_names
        ]
    )

    cell_type_summary.to_csv(output_dir / "scrna_cell_type_feature_summary.csv", index=False)
    status_summary.to_csv(output_dir / "scrna_cell_type_status_feature_summary.csv", index=False)
    donor_summary.to_csv(output_dir / "scrna_donor_cell_type_feature_summary.csv", index=False)
    transporter_summary.to_csv(output_dir / "scrna_transporter_expression_by_cell_type_status.csv", index=False)
    differences.to_csv(output_dir / "scrna_status_feature_differences.csv", index=False)
    top_features.to_csv(output_dir / "scrna_top_cell_types_by_feature.csv", index=False)
    gene_set_membership.to_csv(output_dir / "scrna_feature_gene_membership.csv", index=False)

    plot_metabolism_heatmap(cell_type_summary, metabolism_cols, output_dir / "scrna_metabolism_by_cell_type_heatmap.png")
    plot_transporter_panel_heatmap(
        cell_type_summary,
        panel_cols,
        output_dir / "scrna_transporter_panels_by_cell_type_heatmap.png",
    )
    plot_status_differences(differences, output_dir / "scrna_status_feature_differences.png")

    overview = {
        "scrna_shape": list(shape),
        "n_cells": int(shape[0]),
        "n_genes": int(shape[1]),
        "n_gene_sets": len(matched_gene_sets),
        "gene_sets": {name: genes for name, genes in matched_gene_sets.items()},
        "transporter_genes": transporter_genes,
        "top_cell_types_by_feature": top_features.groupby("feature", observed=True)
        .head(5)
        .to_dict(orient="records"),
        "largest_fibrotic_minus_normal_differences": differences.dropna(subset=["fibrotic_minus_normal"])
        .assign(abs_difference=lambda df: df["fibrotic_minus_normal"].abs())
        .sort_values("abs_difference", ascending=False)
        .head(20)
        .drop(columns=["abs_difference"])
        .to_dict(orient="records"),
    }
    with (output_dir / "scrna_metabolism_transporters_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
