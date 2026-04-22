"""Compare healthy-control signals across scRNA-seq and spatial transcriptomics outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "healthy_control_comparisons"


SCRNA_FEATURES = [
    "bile_acid_transport_mean",
    "epithelial_bile_urea_transport_mean",
    "fibrotic_amino_acid_transport_mean",
    "ecm_remodeling_mean",
    "inflammation_mean",
    "epithelial_repair_mean",
    "fatty_acid_metabolism_mean",
]

SCRNA_CELL_TYPES = [
    "Enterocyte-ANPEP-hi",
    "Enterocyte-FABP-hi",
    "Enterocytes-MT-hi",
    "M2-like macrophage",
    "M2-like macrophage-A2M-hi",
    "CD63+CD81+ macrophage",
    "Inflammatory fibroblasts",
    "Myofibroblasts",
    "Tissue fibroblast",
    "Monocyte",
    "Neutrophils",
]

SPATIAL_METABOLISM_FEATURES = [
    "bile_acid_transport_z_mean",
    "epithelial_repair_z_mean",
    "fatty_acid_metabolism_z_mean",
    "ecm_remodeling_z_mean",
    "inflammation_z_mean",
]

SPATIAL_CELL_TYPES = [
    "Absorptive",
    "Microfold",
    "Secretory",
    "B_plasma",
    "Macrophage",
    "Fibroblast",
    "Myofibroblast",
    "Monocyte",
]

SPATIAL_TRANSPORTER_GENES = [
    "SLC5A1",
    "SLC10A2",
    "SLC51A",
    "SLC51B",
    "ABCB1",
    "ABCC3",
    "SLC38A2",
    "SLC15A3",
    "SLCO2B1",
]

SPATIAL_REGION_GENE_PAIRS = [
    ("epithelial_bile_region", "SLC5A1"),
    ("epithelial_bile_region", "SLC10A2"),
    ("epithelial_bile_region", "SLC51A"),
    ("epithelial_bile_region", "SLC51B"),
    ("stromal_fibrotic_region", "SLC38A2"),
    ("stromal_fibrotic_region", "SLC15A3"),
    ("stromal_fibrotic_region", "SLCO2B1"),
    ("immune_inflammatory_region", "SLCO2B1"),
]


def safe_label(value: str) -> str:
    return value.replace("_mean", "").replace("_z", "").replace("_", " ")


def mann_whitney(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2 or a.nunique() < 2 and b.nunique() < 2:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def cohens_d_independent(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((b.mean() - a.mean()) / pooled)


def scrna_healthy_comparisons(donor_summary: pd.DataFrame) -> pd.DataFrame:
    data = donor_summary[
        donor_summary["annotation2v2"].isin(SCRNA_CELL_TYPES)
        & donor_summary["fraction"].eq("imu")
        & (donor_summary["n_cells"] >= 20)
    ].copy()
    rows = []
    for (cell_type, feature), group in data.melt(
        id_vars=["donor_id", "status", "annotation2v2", "fraction", "n_cells"],
        value_vars=SCRNA_FEATURES,
        var_name="feature",
        value_name="value",
    ).groupby(["annotation2v2", "feature"], observed=True):
        healthy = group[group["status"].eq("H")]["value"]
        for status in ["N", "F", "I"]:
            comparator = group[group["status"].eq(status)]["value"]
            if healthy.empty or comparator.empty:
                continue
            rows.append(
                {
                    "annotation2v2": cell_type,
                    "feature": feature,
                    "comparison": f"{status}_minus_H",
                    "healthy_mean": float(healthy.mean()),
                    "comparison_mean": float(comparator.mean()),
                    "difference_vs_healthy": float(comparator.mean() - healthy.mean()),
                    "cohens_d": cohens_d_independent(healthy, comparator),
                    "mannwhitney_p_value": mann_whitney(healthy, comparator),
                    "n_healthy_donors": int(group[group["status"].eq("H")]["donor_id"].nunique()),
                    "n_comparison_donors": int(group[group["status"].eq(status)]["donor_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def summarize_spatial_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    layer: str,
    feature_name_transform=lambda value: value,
) -> pd.DataFrame:
    rows = []
    control = df[df["Disease Label"].eq("Adjacent (Disease Control)")]
    for feature in feature_cols:
        if feature not in df:
            continue
        for label in ["Adjacent", "Fibrotic"]:
            comp = df[df["Disease Label"].eq(label)]
            rows.append(
                {
                    "analysis_layer": layer,
                    "feature": feature_name_transform(feature),
                    "comparison": f"{label}_minus_DiseaseControl",
                    "disease_control_mean": float(control[feature].mean()),
                    "comparison_mean": float(comp[feature].mean()),
                    "difference_vs_disease_control": float(comp[feature].mean() - control[feature].mean()),
                    "n_disease_control_slides": int(control["sample"].nunique()),
                    "n_comparison_slides": int(comp["sample"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def spatial_healthy_comparisons() -> tuple[pd.DataFrame, pd.DataFrame]:
    metabolism = pd.read_csv(OUTPUT_ROOT / "metabolism_scores" / "sample_metabolism_score_summary.csv")
    deconv = pd.read_csv(OUTPUT_ROOT / "spatial_deconvolution" / "sample_cell_type_proportion_summary.csv")
    transporter = pd.read_csv(OUTPUT_ROOT / "transporter_spatial_validation" / "transporter_gene_expression_by_sample.csv")
    region = pd.read_csv(OUTPUT_ROOT / "spatial_region_validation" / "region_transporter_expression_by_sample.csv")

    metabolism_cmp = summarize_spatial_table(
        metabolism,
        SPATIAL_METABOLISM_FEATURES,
        "spatial_metabolism_score",
        lambda value: value.removesuffix("_z_mean"),
    )
    deconv_cmp = summarize_spatial_table(
        deconv,
        SPATIAL_CELL_TYPES,
        "spatial_deconvolution",
    )

    transporter_wide = transporter[transporter["gene"].isin(SPATIAL_TRANSPORTER_GENES)].pivot_table(
        index=["sample", "Disease Label", "Patient_ID", "General Categorization"],
        columns="gene",
        values="mean_expression",
        aggfunc="mean",
    ).reset_index()
    transporter_cmp = summarize_spatial_table(
        transporter_wide,
        SPATIAL_TRANSPORTER_GENES,
        "slide_transporter_expression",
    )

    region_rows = []
    for region_name, gene in SPATIAL_REGION_GENE_PAIRS:
        column = f"{gene}_mean_expression"
        subset = region[region["region"].eq(region_name)].copy()
        if column not in subset:
            continue
        cmp = summarize_spatial_table(
            subset,
            [column],
            "region_transporter_expression",
            lambda value, rn=region_name, g=gene: f"{rn}::{g}",
        )
        region_rows.append(cmp)
    region_cmp = pd.concat(region_rows, ignore_index=True)

    comparisons = pd.concat([metabolism_cmp, deconv_cmp, transporter_cmp, region_cmp], ignore_index=True)
    sample_counts = (
        metabolism.groupby("Disease Label", observed=True)["sample"]
        .nunique()
        .reset_index(name="n_slides")
    )
    return comparisons, sample_counts


def plot_scrna_healthy_heatmap(scrna: pd.DataFrame, output_path: Path) -> None:
    plot_df = scrna[scrna["comparison"].isin(["F_minus_H", "N_minus_H"])].copy()
    plot_df = plot_df[plot_df["feature"].isin(SCRNA_FEATURES[:5])]
    plot_df["label"] = plot_df["annotation2v2"] + " | " + plot_df["comparison"]
    heatmap = plot_df.pivot_table(index="label", columns="feature", values="difference_vs_healthy", aggfunc="mean")
    heatmap = heatmap.loc[heatmap.abs().max(axis=1).sort_values(ascending=False).head(28).index]
    heatmap = heatmap.rename(columns={col: safe_label(col) for col in heatmap.columns})
    plt.figure(figsize=(11, 10))
    sns.heatmap(heatmap, cmap="vlag", center=0, linewidths=0.2, linecolor="white")
    plt.title("scRNA-seq feature differences versus healthy controls")
    plt.xlabel("feature")
    plt.ylabel("cell type and comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_spatial_healthy_bars(spatial: pd.DataFrame, output_path: Path) -> None:
    priority = spatial[
        spatial["feature"].isin(
            [
                "bile_acid_transport",
                "epithelial_repair",
                "ecm_remodeling",
                "inflammation",
                "SLC5A1",
                "SLC10A2",
                "SLC51A",
                "SLC51B",
                "stromal_fibrotic_region::SLC38A2",
                "stromal_fibrotic_region::SLC15A3",
                "stromal_fibrotic_region::SLCO2B1",
            ]
        )
    ].copy()
    priority["label"] = priority["analysis_layer"] + " | " + priority["feature"] + " | " + priority["comparison"]
    priority = priority.sort_values("difference_vs_disease_control")
    colors = np.where(priority["difference_vs_disease_control"] >= 0, "#C44E52", "#4C72B0")
    plt.figure(figsize=(12, 10))
    plt.barh(priority["label"], priority["difference_vs_disease_control"], color=colors)
    plt.axvline(0, color="#555555", linewidth=0.8)
    plt.title("Spatial descriptive differences versus disease-control adjacent slides")
    plt.xlabel("mean comparison minus disease-control")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_spatial_group_heatmap(spatial: pd.DataFrame, output_path: Path) -> None:
    selected = spatial[
        spatial["analysis_layer"].isin(["spatial_metabolism_score", "slide_transporter_expression"])
    ].copy()
    selected = selected[selected["comparison"].eq("Fibrotic_minus_DiseaseControl")]
    selected = selected.sort_values("difference_vs_disease_control", key=lambda s: s.abs(), ascending=False).head(25)
    heatmap = selected.pivot_table(
        index="feature",
        columns="analysis_layer",
        values="difference_vs_disease_control",
        aggfunc="mean",
    ).fillna(0)
    plt.figure(figsize=(7, 9))
    sns.heatmap(heatmap, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.2, linecolor="white")
    plt.title("Fibrotic spatial differences versus disease control")
    plt.xlabel("analysis layer")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    donor_summary = pd.read_csv(OUTPUT_ROOT / "scrna_metabolism_transporters" / "scrna_donor_cell_type_feature_summary.csv")
    scrna = scrna_healthy_comparisons(donor_summary)
    spatial, spatial_counts = spatial_healthy_comparisons()

    scrna.to_csv(output_dir / "scrna_healthy_status_comparisons.csv", index=False)
    spatial.to_csv(output_dir / "spatial_disease_control_comparisons.csv", index=False)
    spatial_counts.to_csv(output_dir / "spatial_disease_control_sample_counts.csv", index=False)

    plot_scrna_healthy_heatmap(scrna, output_dir / "scrna_healthy_status_comparison_heatmap.png")
    plot_spatial_healthy_bars(spatial, output_dir / "spatial_disease_control_difference_bars.png")
    plot_spatial_group_heatmap(spatial, output_dir / "spatial_fibrotic_vs_disease_control_heatmap.png")

    top_scrna = (
        scrna.assign(abs_difference=lambda df: df["difference_vs_healthy"].abs())
        .sort_values("abs_difference", ascending=False)
        .head(20)
        .drop(columns=["abs_difference"])
    )
    top_spatial = (
        spatial.assign(abs_difference=lambda df: df["difference_vs_disease_control"].abs())
        .sort_values("abs_difference", ascending=False)
        .head(25)
        .drop(columns=["abs_difference"])
    )
    overview = {
        "scrna_statuses": sorted(donor_summary["status"].astype(str).unique().tolist()),
        "scrna_healthy_status": "H",
        "scrna_healthy_donors": int(donor_summary.loc[donor_summary["status"].eq("H"), "donor_id"].nunique()),
        "spatial_disease_control_label": "Adjacent (Disease Control)",
        "spatial_sample_counts": spatial_counts.to_dict(orient="records"),
        "top_scrna_differences_vs_healthy": top_scrna.to_dict(orient="records"),
        "top_spatial_differences_vs_disease_control": top_spatial.to_dict(orient="records"),
        "interpretation_notes": [
            "scRNA-seq healthy-control comparisons use H-status donors as the reference.",
            "Spatial disease-control comparisons are descriptive because only two disease-control adjacent slides are available.",
            "Spatial disease-control slides show high epithelial bile/urea transporter expression relative to Crohn's adjacent and fibrotic groups.",
        ],
    }
    with (output_dir / "healthy_control_comparisons_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
