"""Validate transporter signals within spatial regions defined by deconvolution and pathway scores."""

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
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
METABOLISM_DIR = REPO_ROOT / "output" / "metabolism_scores"
DECONV_DIR = REPO_ROOT / "output" / "spatial_deconvolution"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "spatial_region_validation"


GENE_PANELS = {
    "epithelial_bile_urea_transport": ["SLC5A1", "SLC10A2", "SLC51A", "SLC51B", "ABCB1", "ABCC3"],
    "fibrotic_amino_acid_transport": ["SLC38A2", "SLC15A3", "SLCO2B1", "SLC7A5", "SLC1A5", "SLC1A4"],
}

REGION_DEFINITIONS = {
    "epithelial_bile_region": [
        "Absorptive",
        "Microfold",
        "bile_acid_transport_z",
        "epithelial_repair_z",
    ],
    "stromal_fibrotic_region": [
        "Fibroblast",
        "Myofibroblast",
        "Macrophage",
        "ecm_remodeling_z",
        "inflammation_z",
    ],
    "immune_inflammatory_region": [
        "Macrophage",
        "Monocyte",
        "B_plasma",
        "inflammation_z",
    ],
    "epithelial_absorptive_region": [
        "Absorptive",
        "Microfold",
        "epithelial_repair_z",
    ],
}

MAP_REGIONS = [
    "epithelial_bile_region",
    "stromal_fibrotic_region",
    "immune_inflammatory_region",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def within_sample_z(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mean) / std


def merge_spot_tables() -> pd.DataFrame:
    metabolism = pd.read_csv(METABOLISM_DIR / "spot_metabolism_scores_selected.csv")
    deconv = pd.read_csv(DECONV_DIR / "spot_cell_type_proportions_selected.csv")
    key = ["sample", "Disease Label", "Patient_ID", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    deconv_extra = [column for column in deconv.columns if column not in key]
    merged = metabolism.merge(deconv[key + deconv_extra], on=key, how="inner", validate="one_to_one")
    return merged


def add_region_scores(spots: pd.DataFrame, top_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_rows = []
    for region, columns in REGION_DEFINITIONS.items():
        present = [column for column in columns if column in spots.columns]
        if not present:
            spots[f"{region}_score"] = np.nan
            spots[f"{region}_high"] = False
            continue

        z_columns = []
        for column in present:
            z_column = f"_{region}_{safe_name(column)}_within_sample_z"
            spots[z_column] = spots.groupby("sample", observed=True)[column].transform(within_sample_z)
            z_columns.append(z_column)

        spots[f"{region}_score"] = spots[z_columns].mean(axis=1)
        cutoff = spots.groupby("sample", observed=True)[f"{region}_score"].transform(
            lambda values: values.quantile(1 - top_fraction)
        )
        spots[f"{region}_high"] = spots[f"{region}_score"] >= cutoff
        spots = spots.drop(columns=z_columns)

        summary = (
            spots.groupby(["sample", "Disease Label", "Patient_ID"], observed=True)
            .agg(
                n_spots=("sample", "size"),
                high_region_spots=(f"{region}_high", "sum"),
                mean_region_score=(f"{region}_score", "mean"),
                median_region_score=(f"{region}_score", "median"),
            )
            .reset_index()
        )
        summary["region"] = region
        summary["region_columns"] = ";".join(present)
        region_rows.append(summary)

    region_summary = pd.concat(region_rows, ignore_index=True)
    return spots, region_summary


def extract_gene_expression(genes: list[str]) -> pd.DataFrame:
    adata = ad.read_h5ad(DATA_DIR / "anndata.h5ad")
    try:
        present = [gene for gene in genes if gene in adata.var_names]
        coords = pd.DataFrame(
            adata.obsm["spatial"],
            index=adata.obs_names,
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
        expression = adata.obs[["sample"]].copy().join(coords)
        for gene in present:
            values = adata.X[:, adata.var_names.get_loc(gene)]
            if sparse.issparse(values):
                values = values.toarray().ravel()
            else:
                values = np.asarray(values).ravel()
            expression[gene] = values.astype(float)
    finally:
        del adata
    return expression


def add_gene_expression(spots: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    expression = extract_gene_expression(genes)
    key = ["sample", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    present_genes = [gene for gene in genes if gene in expression.columns]
    merged = spots.merge(expression[key + present_genes], on=key, how="inner", validate="one_to_one")
    return merged, present_genes


def summarize_region_gene_enrichment(spots: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows = []
    strata = [("all", spots)]
    for label, group in spots.groupby("Disease Label", observed=True):
        strata.append((str(label), group))

    for stratum, group in strata:
        for region in REGION_DEFINITIONS:
            flag = f"{region}_high"
            if flag not in group:
                continue
            inside = group[group[flag]]
            outside = group[~group[flag]]
            for gene in genes:
                if gene not in group:
                    continue
                inside_mean = float(inside[gene].mean())
                outside_mean = float(outside[gene].mean())
                rows.append(
                    {
                        "stratum": stratum,
                        "region": region,
                        "gene": gene,
                        "mean_expression_in_region": inside_mean,
                        "mean_expression_outside_region": outside_mean,
                        "in_region_minus_outside": inside_mean - outside_mean,
                        "in_region_to_outside_ratio": inside_mean / outside_mean if outside_mean > 0 else np.nan,
                        "n_in_region_spots": int(len(inside)),
                        "n_outside_region_spots": int(len(outside)),
                    }
                )
    return pd.DataFrame(rows)


def correlate_genes_with_region_scores(spots: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows = []
    for region in REGION_DEFINITIONS:
        score = f"{region}_score"
        if score not in spots:
            continue
        for gene in genes:
            if gene not in spots:
                continue
            valid = spots[[score, gene]].dropna()
            if len(valid) < 10 or valid[score].nunique() < 2 or valid[gene].nunique() < 2:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(valid[score], valid[gene])
            rows.append(
                {
                    "region": region,
                    "gene": gene,
                    "spearman_r": rho,
                    "p_value": p_value,
                    "n_spots": int(len(valid)),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def paired_region_gene_differences(spots: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for region in REGION_DEFINITIONS:
        flag = f"{region}_high"
        for (sample, disease_label, patient_id), group in spots.groupby(
            ["sample", "Disease Label", "Patient_ID"],
            observed=True,
        ):
            high = group[group[flag]]
            row = {
                "sample": sample,
                "Disease Label": disease_label,
                "Patient_ID": patient_id,
                "region": region,
                "n_region_spots": int(len(high)),
            }
            for gene in genes:
                row[f"{gene}_mean_expression"] = float(high[gene].mean())
            summary_rows.append(row)

    by_sample = pd.DataFrame(summary_rows)
    paired_rows = []
    for (patient_id, region), group in by_sample.groupby(["Patient_ID", "region"], observed=True):
        adjacent = group[group["Disease Label"].eq("Adjacent")]
        fibrotic = group[group["Disease Label"].eq("Fibrotic")]
        if adjacent.empty or fibrotic.empty:
            continue
        for gene in genes:
            adjacent_mean = float(adjacent[f"{gene}_mean_expression"].mean())
            fibrotic_mean = float(fibrotic[f"{gene}_mean_expression"].mean())
            paired_rows.append(
                {
                    "Patient_ID": patient_id,
                    "region": region,
                    "gene": gene,
                    "adjacent_region_mean_expression": adjacent_mean,
                    "fibrotic_region_mean_expression": fibrotic_mean,
                    "fibrotic_minus_adjacent_region_expression": fibrotic_mean - adjacent_mean,
                    "n_adjacent_slides": int(adjacent["sample"].nunique()),
                    "n_fibrotic_slides": int(fibrotic["sample"].nunique()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    return by_sample, paired


def summarize_paired_region_differences(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["region", "gene"], observed=True)
        .agg(
            mean_fibrotic_minus_adjacent_region_expression=(
                "fibrotic_minus_adjacent_region_expression",
                "mean",
            ),
            median_fibrotic_minus_adjacent_region_expression=(
                "fibrotic_minus_adjacent_region_expression",
                "median",
            ),
            n_patients=("Patient_ID", "nunique"),
            positive_patient_fraction=(
                "fibrotic_minus_adjacent_region_expression",
                lambda values: float((values > 0).mean()),
            ),
        )
        .reset_index()
        .sort_values("mean_fibrotic_minus_adjacent_region_expression", ascending=False)
    )


def plot_region_gene_heatmap(enrichment: pd.DataFrame, output_path: Path) -> None:
    plot_df = enrichment[enrichment["stratum"].eq("all")].pivot(
        index="gene",
        columns="region",
        values="in_region_minus_outside",
    )
    plot_df = plot_df.loc[plot_df.abs().max(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(9, 7))
    sns.heatmap(plot_df, cmap="vlag", center=0, linewidths=0.3, linecolor="white")
    plt.title("Transporter expression enrichment in high-signal spatial regions")
    plt.xlabel("region")
    plt.ylabel("transporter gene")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_paired_region_differences(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.copy()
    plot_df["label"] = plot_df["gene"] + " | " + plot_df["region"]
    plot_df["abs_diff"] = plot_df["mean_fibrotic_minus_adjacent_region_expression"].abs()
    plot_df = plot_df.sort_values("abs_diff", ascending=False).head(32)
    plot_df = plot_df.sort_values("mean_fibrotic_minus_adjacent_region_expression")
    colors = np.where(plot_df["mean_fibrotic_minus_adjacent_region_expression"] >= 0, "#C44E52", "#4C72B0")

    plt.figure(figsize=(11, 10))
    plt.barh(plot_df["label"], plot_df["mean_fibrotic_minus_adjacent_region_expression"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Region-level transporter expression differences")
    plt.xlabel("mean paired fibrotic minus adjacent expression in high-signal regions")
    plt.ylabel("gene and region")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_region_score_maps(spots: pd.DataFrame, region: str, output_path: Path) -> None:
    score = f"{region}_score"
    samples = sorted(spots["sample"].unique())
    n_cols = 5
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    axes = axes.flatten()
    lower = np.nanquantile(spots[score], 0.02)
    upper = np.nanquantile(spots[score], 0.98)
    for ax, sample in zip(axes, samples):
        slide = spots[spots["sample"].eq(sample)]
        im = ax.scatter(
            slide["pxl_col_in_fullres"],
            slide["pxl_row_in_fullres"],
            c=slide[score],
            cmap="magma",
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
    fig.suptitle(region.replace("_", " "), y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    fig.colorbar(im, cax=cbar_ax, label="within-slide composite score")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    args = parser.parse_args()

    output_dir = args.output_dir
    maps_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    genes = sorted({gene for panel in GENE_PANELS.values() for gene in panel})
    spots = merge_spot_tables()
    spots, region_summary = add_region_scores(spots, args.top_fraction)
    spots, present_genes = add_gene_expression(spots, genes)

    selected_columns = [
        "sample",
        "Disease Label",
        "Patient_ID",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    selected_columns.extend([f"{region}_score" for region in REGION_DEFINITIONS])
    selected_columns.extend([f"{region}_high" for region in REGION_DEFINITIONS])
    selected_columns.extend(
        [
            "Absorptive",
            "Microfold",
            "Fibroblast",
            "Myofibroblast",
            "Macrophage",
            "B_plasma",
            "bile_acid_transport_z",
            "epithelial_repair_z",
            "ecm_remodeling_z",
            "inflammation_z",
        ]
    )
    selected_columns.extend(present_genes)
    selected_columns = [column for column in selected_columns if column in spots.columns]

    enrichment = summarize_region_gene_enrichment(spots, present_genes)
    correlations = correlate_genes_with_region_scores(spots, present_genes)
    region_gene_by_sample, paired = paired_region_gene_differences(spots, present_genes)
    paired_summary = summarize_paired_region_differences(paired)

    spots[selected_columns].to_csv(output_dir / "spot_region_scores_selected.csv", index=False)
    region_summary.to_csv(output_dir / "region_score_summary_by_sample.csv", index=False)
    enrichment.to_csv(output_dir / "region_transporter_enrichment.csv", index=False)
    correlations.to_csv(output_dir / "region_transporter_correlations.csv", index=False)
    region_gene_by_sample.to_csv(output_dir / "region_transporter_expression_by_sample.csv", index=False)
    paired.to_csv(output_dir / "paired_region_transporter_differences.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_region_transporter_difference_summary.csv", index=False)

    plot_region_gene_heatmap(enrichment, output_dir / "region_transporter_enrichment_heatmap.png")
    plot_paired_region_differences(paired_summary, output_dir / "paired_region_transporter_differences.png")
    for region in MAP_REGIONS:
        plot_region_score_maps(spots, region, maps_dir / f"{region}_all_slides.png")

    overview = {
        "top_fraction_per_slide_used_for_high_regions": args.top_fraction,
        "n_spots": int(len(spots)),
        "n_samples": int(spots["sample"].nunique()),
        "region_definitions": REGION_DEFINITIONS,
        "gene_panels": GENE_PANELS,
        "present_genes": present_genes,
        "top_overall_region_enrichments": enrichment[enrichment["stratum"].eq("all")]
        .assign(abs_enrichment=lambda df: df["in_region_minus_outside"].abs())
        .sort_values("abs_enrichment", ascending=False)
        .head(20)
        .drop(columns=["abs_enrichment"])
        .to_dict(orient="records"),
        "top_paired_region_differences": paired_summary.assign(
            abs_difference=lambda df: df["mean_fibrotic_minus_adjacent_region_expression"].abs()
        )
        .sort_values("abs_difference", ascending=False)
        .head(20)
        .drop(columns=["abs_difference"])
        .to_dict(orient="records"),
        "strongest_region_gene_correlations": correlations.dropna(subset=["spearman_r"])
        .assign(abs_r=lambda df: df["spearman_r"].abs())
        .sort_values("abs_r", ascending=False)
        .head(20)
        .drop(columns=["abs_r"])
        .to_dict(orient="records"),
    }
    with (output_dir / "spatial_region_validation_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
