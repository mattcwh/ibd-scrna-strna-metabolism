"""Validate focus Harreman metabolites with transporter gene expression and spatial maps."""

from __future__ import annotations

import argparse
import json
import re
from importlib.resources import files
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
HARREMAN_DIR = REPO_ROOT / "output" / "harreman_all_slides"
DECONV_DIR = REPO_ROOT / "output" / "spatial_deconvolution"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "transporter_spatial_validation"


FOCUS_METABOLITES = [
    "Bile acid",
    "Cholic acid",
    "Urea",
    "L-Histidine",
    "L-Serine",
    "Citric acid",
    "L-Cysteine",
]

REPRESENTATIVE_GENES = {
    "Bile acid": ["SLC51A", "SLC51B"],
    "Cholic acid": ["SLC10A2", "ABCC3", "SLC10A1", "ABCB11", "SLCO1B1", "SLCO1B3"],
    "Urea": ["SLC14A1", "SLC14A2", "SLC5A1"],
    "L-Histidine": ["SLC7A5", "SLC38A1", "SLC38A2", "SLC16A10"],
    "L-Serine": ["SLC1A5", "SLC1A4", "SLC7A10", "SLC38A1", "SLC38A2"],
    "Citric acid": ["SLC13A5", "SLCO2B1"],
    "L-Cysteine": ["SLC7A11", "SLC1A5", "SLC7A8", "SLC38A1", "SLC38A2"],
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_harreman_db() -> pd.DataFrame:
    path = files("harreman") / "data" / "HarremanDB" / "HarremanDB_human_extracellular.csv"
    db = pd.read_csv(path, index_col=0)
    db["genes"] = db["Gene"].fillna("").map(
        lambda value: [gene.strip() for gene in str(value).split("/") if gene.strip()]
    )
    return db


def focus_metabolite_gene_map(db: pd.DataFrame, spatial_genes: pd.Index) -> pd.DataFrame:
    rows = []
    for metabolite in FOCUS_METABOLITES:
        matches = db[db["Metabolite"].eq(metabolite)]
        if matches.empty and metabolite == "Cholic acid":
            matches = db[db["Metabolite"].str.contains("Cholic acid", case=False, regex=False, na=False)]
        for _, row in matches.iterrows():
            for gene in row["genes"]:
                rows.append(
                    {
                        "focus_metabolite": metabolite,
                        "database_metabolite": row["Metabolite"],
                        "gene": gene,
                        "present_in_spatial": gene in spatial_genes,
                        "database": row.get("Database", ""),
                        "evidence": row.get("Evidence", ""),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates()


def mean_expression_by_sample(adata: ad.AnnData, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    present = [gene for gene in genes if gene in adata.var_names]
    coords = pd.DataFrame(
        adata.obsm["spatial"],
        index=adata.obs_names,
        columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
    )
    spot = adata.obs[["sample"]].copy().join(coords)

    for gene in present:
        idx = adata.var_names.get_loc(gene)
        values = adata.X[:, idx]
        if sparse.issparse(values):
            values = values.toarray().ravel()
        else:
            values = np.asarray(values).ravel()
        spot[gene] = values.astype(float)

    long = spot.melt(
        id_vars=["sample", "pxl_row_in_fullres", "pxl_col_in_fullres"],
        value_vars=present,
        var_name="gene",
        value_name="expression",
    )
    sample_summary = (
        long.groupby(["sample", "gene"], observed=True)["expression"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_expression", "median": "median_expression", "std": "std_expression"})
    )
    return long, sample_summary


def paired_gene_differences(sample_summary: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"Disease Label", "Patient_ID", "General Categorization"}
    if required.issubset(sample_summary.columns):
        merged = sample_summary.copy()
    else:
        merged = sample_summary.merge(
            metadata[["sample", "Disease Label", "Patient_ID", "General Categorization"]],
            on="sample",
            how="left",
        )
    rows = []
    for (patient_id, gene), group in merged.groupby(["Patient_ID", "gene"], observed=True):
        adjacent = group[group["Disease Label"].eq("Adjacent")]
        fibrotic = group[group["Disease Label"].eq("Fibrotic")]
        if adjacent.empty or fibrotic.empty:
            continue
        rows.append(
            {
                "Patient_ID": patient_id,
                "gene": gene,
                "adjacent_mean_expression": float(adjacent["mean_expression"].mean()),
                "fibrotic_mean_expression": float(fibrotic["mean_expression"].mean()),
                "fibrotic_minus_adjacent_expression": float(
                    fibrotic["mean_expression"].mean() - adjacent["mean_expression"].mean()
                ),
                "n_adjacent_slides": int(adjacent["sample"].nunique()),
                "n_fibrotic_slides": int(fibrotic["sample"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_gene_differences(paired: pd.DataFrame, gene_map: pd.DataFrame) -> pd.DataFrame:
    summary = (
        paired.groupby("gene", observed=True)
        .agg(
            mean_fibrotic_minus_adjacent_expression=("fibrotic_minus_adjacent_expression", "mean"),
            median_fibrotic_minus_adjacent_expression=("fibrotic_minus_adjacent_expression", "median"),
            n_patients=("Patient_ID", "nunique"),
        )
        .reset_index()
    )
    summary["positive_patient_fraction"] = paired.groupby("gene", observed=True)[
        "fibrotic_minus_adjacent_expression"
    ].apply(lambda values: float((values > 0).mean())).values
    metabolite_lookup = (
        gene_map.groupby("gene", observed=True)["focus_metabolite"]
        .apply(lambda values: ";".join(sorted(set(values))))
        .reset_index()
    )
    return summary.merge(metabolite_lookup, on="gene", how="left").sort_values(
        "mean_fibrotic_minus_adjacent_expression",
        ascending=False,
    )


def correlate_genes_with_harreman(
    sample_summary: pd.DataFrame,
    harreman_metabolites: pd.DataFrame,
    gene_map: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, mapping in gene_map[gene_map["present_in_spatial"]].iterrows():
        gene = mapping["gene"]
        metabolite = mapping["focus_metabolite"]
        gene_df = sample_summary[sample_summary["gene"].eq(gene)][["sample", "mean_expression"]]
        h_df = harreman_metabolites[
            harreman_metabolites["metabolite"].eq(metabolite)
        ][["sample", "z_score"]]
        merged = gene_df.merge(h_df, on="sample", how="inner")
        if len(merged) < 5 or merged["mean_expression"].nunique() < 2 or merged["z_score"].nunique() < 2:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = spearmanr(merged["mean_expression"], merged["z_score"])
        rows.append(
            {
                "focus_metabolite": metabolite,
                "database_metabolite": mapping["database_metabolite"],
                "gene": gene,
                "spearman_r": rho,
                "p_value": p_value,
                "n_slides": int(len(merged)),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def plot_gene_difference_summary(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.dropna(subset=["focus_metabolite"]).copy()
    plot_df = pd.concat([plot_df.head(20), plot_df.tail(20)], ignore_index=True).drop_duplicates("gene")
    plot_df = plot_df.sort_values("mean_fibrotic_minus_adjacent_expression")
    colors = np.where(plot_df["mean_fibrotic_minus_adjacent_expression"] >= 0, "#C44E52", "#4C72B0")
    labels = plot_df["gene"] + " (" + plot_df["focus_metabolite"] + ")"

    plt.figure(figsize=(10, 11))
    plt.barh(labels, plot_df["mean_fibrotic_minus_adjacent_expression"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Transporter gene expression differences")
    plt.xlabel("mean paired fibrotic minus adjacent expression")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_gene_harreman_correlation(corr: pd.DataFrame, output_path: Path) -> None:
    plot_df = corr.dropna(subset=["spearman_r"]).copy()
    plot_df = plot_df.assign(abs_r=plot_df["spearman_r"].abs()).sort_values("abs_r", ascending=False).head(30)
    plot_df = plot_df.sort_values("spearman_r")
    colors = np.where(plot_df["spearman_r"] >= 0, "#4C72B0", "#C44E52")
    labels = plot_df["gene"] + " (" + plot_df["focus_metabolite"] + ")"

    plt.figure(figsize=(10, 9))
    plt.barh(labels, plot_df["spearman_r"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Transporter expression versus Harreman metabolite z score")
    plt.xlabel("Spearman correlation across slides")
    plt.ylabel("gene and metabolite")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_all_slide_gene_map(spot_long: pd.DataFrame, gene: str, metadata: pd.DataFrame, output_path: Path) -> None:
    data = spot_long[spot_long["gene"].eq(gene)].merge(
        metadata[["sample", "Disease Label"]],
        on="sample",
        how="left",
    )
    if data.empty:
        return
    samples = sorted(data["sample"].unique())
    n_cols = 5
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    axes = axes.flatten()
    upper = np.nanquantile(data["expression"], 0.98)
    for ax, sample in zip(axes, samples):
        slide = data[data["sample"].eq(sample)]
        im = ax.scatter(
            slide["pxl_col_in_fullres"],
            slide["pxl_row_in_fullres"],
            c=slide["expression"],
            cmap="viridis",
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
    fig.suptitle(f"{gene} spatial expression", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    fig.colorbar(im, cax=cbar_ax, label="expression")
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

    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["sample"] = metadata["Visium Slide ID"].astype(str)
    harreman_metabolites = pd.read_csv(HARREMAN_DIR / "harreman_all_metabolite_results.csv")

    adata = ad.read_h5ad(DATA_DIR / "anndata.h5ad")
    try:
        db = load_harreman_db()
        gene_map = focus_metabolite_gene_map(db, adata.var_names)
        present_genes = sorted(gene_map.loc[gene_map["present_in_spatial"], "gene"].unique())
        representative = [
            gene
            for genes in REPRESENTATIVE_GENES.values()
            for gene in genes
            if gene in present_genes
        ]
        genes_to_extract = sorted(set(present_genes) | set(representative))
        spot_long, sample_summary = mean_expression_by_sample(adata, genes_to_extract)
    finally:
        del adata

    sample_summary = sample_summary.merge(
        metadata[["sample", "Disease Label", "Patient_ID", "General Categorization"]],
        on="sample",
        how="left",
    )
    paired = paired_gene_differences(sample_summary, metadata)
    gene_diff_summary = summarize_gene_differences(paired, gene_map)
    gene_harreman_corr = correlate_genes_with_harreman(sample_summary, harreman_metabolites, gene_map)

    gene_map.to_csv(output_dir / "focus_metabolite_transporter_gene_map.csv", index=False)
    sample_summary.to_csv(output_dir / "transporter_gene_expression_by_sample.csv", index=False)
    paired.to_csv(output_dir / "paired_transporter_gene_expression_differences.csv", index=False)
    gene_diff_summary.to_csv(output_dir / "transporter_gene_difference_summary.csv", index=False)
    gene_harreman_corr.to_csv(output_dir / "transporter_expression_harreman_correlations.csv", index=False)

    plot_gene_difference_summary(gene_diff_summary, output_dir / "transporter_gene_difference_summary.png")
    plot_gene_harreman_correlation(gene_harreman_corr, output_dir / "transporter_expression_harreman_correlations.png")

    for gene in sorted(set(representative)):
        plot_all_slide_gene_map(spot_long, gene, metadata, maps_dir / f"{safe_name(gene)}_all_slides.png")

    overview = {
        "focus_metabolites": FOCUS_METABOLITES,
        "n_focus_transporters": int(gene_map["gene"].nunique()),
        "n_focus_transporters_present_in_spatial": int(gene_map.loc[gene_map["present_in_spatial"], "gene"].nunique()),
        "top_fibrotic_shifted_transporters": gene_diff_summary.head(15).to_dict(orient="records"),
        "top_adjacent_shifted_transporters": gene_diff_summary.tail(15)
        .sort_values("mean_fibrotic_minus_adjacent_expression")
        .to_dict(orient="records"),
        "strongest_transporter_harreman_correlations": gene_harreman_corr.dropna(subset=["spearman_r"])
        .assign(abs_r=lambda df: df["spearman_r"].abs())
        .sort_values("abs_r", ascending=False)
        .head(20)
        .drop(columns=["abs_r"])
        .to_dict(orient="records"),
    }
    with (output_dir / "transporter_spatial_validation_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
