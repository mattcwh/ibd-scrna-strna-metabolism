"""Interpret Harreman metabolite shifts with deconvolution and marker-gene evidence."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import ConstantInputWarning, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "harreman_deconv_interpretation"
HARREMAN_DIR = REPO_ROOT / "output" / "harreman_all_slides"
DECONV_DIR = REPO_ROOT / "output" / "spatial_deconvolution"


MARKER_SETS: dict[str, list[str]] = {
    "B_plasma": ["MZB1", "JCHAIN", "IGHG1", "IGHA1", "SDC1", "XBP1"],
    "B_cells": ["MS4A1", "CD79A", "CD79B", "CD74"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Myofibroblast": ["ACTA2", "TAGLN", "MYL9", "COL1A1", "POSTN"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
    "Absorptive": ["FABP1", "FABP2", "APOA1", "ALPI", "AQP8"],
    "Secretory": ["MUC2", "TFF3", "SPDEF", "AGR2"],
    "Macrophage": ["LYZ", "C1QA", "C1QB", "CD68", "MSR1"],
    "Endothelial": ["VWF", "PECAM1", "KDR", "EMCN"],
    "Pericyte": ["RGS5", "PDGFRB", "CSPG4", "MCAM"],
}


FOCUS_METABOLITES = [
    "L-Histidine",
    "L-Serine",
    "Citric acid",
    "L-Cysteine",
    "L-Asparagine",
    "L-Alanine",
    "L-Leucine",
    "Urea",
    "Propionate",
    "beta-Dhydroxybutyrate",
    "Bile acid",
    "Cholic acid",
    "Cholesterol",
    "Fatty acid",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "harreman_metabolites": pd.read_csv(HARREMAN_DIR / "harreman_all_metabolite_results.csv"),
        "harreman_gene_pairs": pd.read_csv(HARREMAN_DIR / "harreman_all_gene_pair_results.csv"),
        "harreman_differences": pd.read_csv(HARREMAN_DIR / "harreman_metabolite_difference_summary.csv"),
        "harreman_patient_differences": pd.read_csv(
            HARREMAN_DIR / "harreman_patient_paired_metabolite_differences.csv"
        ),
        "deconv_sample": pd.read_csv(DECONV_DIR / "sample_cell_type_proportion_summary.csv"),
        "deconv_paired": pd.read_csv(DECONV_DIR / "paired_patient_cell_type_differences.csv"),
    }


def deconvolution_cell_types(deconv_sample: pd.DataFrame) -> list[str]:
    metadata_cols = {"sample", "Disease Label", "Patient_ID", "General Categorization"}
    return [column for column in deconv_sample.columns if column not in metadata_cols]


def correlate_slide_level_harreman_deconv(
    harreman_metabolites: pd.DataFrame,
    deconv_sample: pd.DataFrame,
    focus_metabolites: list[str],
    cell_types: list[str],
) -> pd.DataFrame:
    metabolite_wide = harreman_metabolites[
        harreman_metabolites["metabolite"].isin(focus_metabolites)
    ].pivot_table(index="sample", columns="metabolite", values="z_score", aggfunc="mean")
    merged = deconv_sample.set_index("sample")[cell_types].join(metabolite_wide, how="inner")

    rows = []
    for metabolite in metabolite_wide.columns:
        for cell_type in cell_types:
            valid = merged[[cell_type, metabolite]].dropna()
            if len(valid) < 5:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(valid[cell_type], valid[metabolite])
            rows.append(
                {
                    "metabolite": metabolite,
                    "cell_type": cell_type,
                    "spearman_r": rho,
                    "p_value": p_value,
                    "n_slides": int(len(valid)),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def correlate_paired_differences(
    harreman_patient_differences: pd.DataFrame,
    deconv_paired: pd.DataFrame,
    focus_metabolites: list[str],
    cell_types: list[str],
) -> pd.DataFrame:
    rows = []
    for metabolite in focus_metabolites:
        h = harreman_patient_differences[
            harreman_patient_differences["metabolite"].eq(metabolite)
        ][["Patient_ID", "fibrotic_minus_adjacent_z"]]
        if h.empty:
            continue
        merged = deconv_paired.merge(h, on="Patient_ID", how="inner")
        for cell_type in cell_types:
            column = f"{cell_type}_fibrotic_minus_adjacent"
            if column not in merged:
                continue
            valid = merged[[column, "fibrotic_minus_adjacent_z"]].dropna()
            if len(valid) < 5:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(valid[column], valid["fibrotic_minus_adjacent_z"])
            rows.append(
                {
                    "metabolite": metabolite,
                    "cell_type": cell_type,
                    "spearman_r": rho,
                    "p_value": p_value,
                    "n_patients": int(len(valid)),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def top_gene_pair_evidence(
    harreman_gene_pairs: pd.DataFrame,
    harreman_metabolites: pd.DataFrame,
    focus_metabolites: list[str],
) -> pd.DataFrame:
    # Harreman gene-pair outputs do not carry metabolite labels directly. Use top slide-level
    # gene pairs as supporting transporter evidence for each focus metabolite by matching genes
    # from per-metabolite top z-score slides through the all-gene-pair ranked table.
    top_slide = (
        harreman_metabolites[harreman_metabolites["metabolite"].isin(focus_metabolites)]
        .sort_values(["metabolite", "z_score"], ascending=[True, False])
        .groupby("metabolite", observed=True)
        .head(3)
    )
    rows = []
    for _, row in top_slide.iterrows():
        sample = row["sample"]
        gp = harreman_gene_pairs[harreman_gene_pairs["sample"].eq(sample)].head(25)
        for _, gp_row in gp.iterrows():
            rows.append(
                {
                    "metabolite": row["metabolite"],
                    "sample": sample,
                    "metabolite_z_score": row["z_score"],
                    "gene_pair": gp_row["gene_pair"],
                    "gene_pair_z_score": gp_row["z_score"],
                    "gene_pair_fdr": gp_row["fdr"],
                    "Disease Label": row["Disease Label"],
                    "Patient_ID": row["Patient_ID"],
                }
            )
    return pd.DataFrame(rows)


def marker_scores_by_slide() -> pd.DataFrame:
    adata = ad.read_h5ad(DATA_DIR / "anndata.h5ad")
    try:
        rows = []
        for marker_set, genes in MARKER_SETS.items():
            present = [gene for gene in genes if gene in adata.var_names]
            if not present:
                continue
            idx = [adata.var_names.get_loc(gene) for gene in present]
            values = adata.X[:, idx].mean(axis=1)
            if sparse.issparse(values):
                values = values.A1
            else:
                values = np.asarray(values).ravel()
            tmp = pd.DataFrame(
                {
                    "sample": adata.obs["sample"].astype(str).to_numpy(),
                    "marker_set": marker_set,
                    "marker_score": values,
                    "n_genes": len(present),
                    "genes": ";".join(present),
                }
            )
            rows.append(tmp)
    finally:
        del adata

    scores = pd.concat(rows, ignore_index=True)
    summary = (
        scores.groupby(["sample", "marker_set", "n_genes", "genes"], observed=True)["marker_score"]
        .mean()
        .reset_index()
    )
    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["sample"] = metadata["Visium Slide ID"].astype(str)
    return summary.merge(
        metadata[["sample", "Disease Label", "Patient_ID", "General Categorization"]],
        on="sample",
        how="left",
    )


def correlate_marker_deconv(marker_summary: pd.DataFrame, deconv_sample: pd.DataFrame) -> pd.DataFrame:
    cell_types = deconvolution_cell_types(deconv_sample)
    marker_wide = marker_summary.pivot_table(
        index="sample",
        columns="marker_set",
        values="marker_score",
        aggfunc="mean",
    )
    marker_wide = marker_wide.rename(columns={column: f"marker__{column}" for column in marker_wide.columns})
    merged = deconv_sample.set_index("sample")[cell_types].join(marker_wide, how="inner")
    rows = []
    for marker_column in marker_wide.columns:
        marker_set = marker_column.removeprefix("marker__")
        for cell_type in cell_types:
            valid = merged[[cell_type, marker_column]].dropna()
            if len(valid) < 5:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(valid[cell_type], valid[marker_column])
            rows.append(
                {
                    "marker_set": marker_set,
                    "cell_type": cell_type,
                    "spearman_r": rho,
                    "p_value": p_value,
                    "n_slides": int(len(valid)),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def plot_correlation_heatmap(corr: pd.DataFrame, value: str, index: str, columns: str, output_path: Path) -> None:
    heatmap = corr.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    plt.figure(figsize=(13, 8))
    sns.heatmap(heatmap, cmap="vlag", center=0, vmin=-1, vmax=1, linewidths=0.2)
    plt.title(value.replace("_", " "))
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_marker_validation(marker_corr: pd.DataFrame, output_path: Path) -> None:
    expected_pairs = {
        "B_plasma": "B_plasma",
        "B_cells": "Mature_B",
        "Fibroblast": "Fibroblast",
        "Myofibroblast": "Myofibroblast",
        "Epithelial": "Absorptive",
        "Absorptive": "Absorptive",
        "Secretory": "Secretory",
        "Macrophage": "Macrophage",
        "Endothelial": "Vascular_endothelia",
        "Pericyte": "Pericyte",
    }
    rows = []
    for marker_set, cell_type in expected_pairs.items():
        match = marker_corr[
            marker_corr["marker_set"].eq(marker_set) & marker_corr["cell_type"].eq(cell_type)
        ]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    plot_df = pd.DataFrame(rows).sort_values("spearman_r")
    colors = np.where(plot_df["spearman_r"] >= 0, "#4C72B0", "#C44E52")
    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["marker_set"] + " vs " + plot_df["cell_type"], plot_df["spearman_r"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Marker validation for deconvolved cell-type proportions")
    plt.xlabel("Spearman correlation across slides")
    plt.ylabel("marker set and deconvolved cell type")
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
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    inputs = load_inputs()
    cell_types = deconvolution_cell_types(inputs["deconv_sample"])
    focus_metabolites = [
        metabolite
        for metabolite in FOCUS_METABOLITES
        if metabolite in set(inputs["harreman_patient_differences"]["metabolite"])
    ]

    slide_corr = correlate_slide_level_harreman_deconv(
        inputs["harreman_metabolites"],
        inputs["deconv_sample"],
        focus_metabolites,
        cell_types,
    )
    paired_corr = correlate_paired_differences(
        inputs["harreman_patient_differences"],
        inputs["deconv_paired"],
        focus_metabolites,
        cell_types,
    )
    gene_pair_evidence = top_gene_pair_evidence(
        inputs["harreman_gene_pairs"],
        inputs["harreman_metabolites"],
        focus_metabolites,
    )
    marker_summary = marker_scores_by_slide()
    marker_corr = correlate_marker_deconv(marker_summary, inputs["deconv_sample"])

    slide_corr.to_csv(output_dir / "slide_level_harreman_deconv_correlations.csv", index=False)
    paired_corr.to_csv(output_dir / "paired_difference_harreman_deconv_correlations.csv", index=False)
    gene_pair_evidence.to_csv(output_dir / "top_metabolite_gene_pair_evidence.csv", index=False)
    marker_summary.to_csv(output_dir / "marker_scores_by_slide.csv", index=False)
    marker_corr.to_csv(output_dir / "marker_deconvolution_correlations.csv", index=False)

    plot_correlation_heatmap(
        slide_corr,
        value="spearman_r",
        index="metabolite",
        columns="cell_type",
        output_path=output_dir / "slide_level_harreman_deconv_correlation_heatmap.png",
    )
    top_cell_types = (
        paired_corr.assign(abs_r=lambda df: df["spearman_r"].abs())
        .sort_values("abs_r", ascending=False)
        .groupby("metabolite", observed=True)
        .head(5)
    )
    plot_correlation_heatmap(
        top_cell_types,
        value="spearman_r",
        index="metabolite",
        columns="cell_type",
        output_path=output_dir / "paired_difference_top_correlation_heatmap.png",
    )
    plot_marker_validation(marker_corr, output_dir / "marker_deconvolution_validation.png")

    marker_expected = marker_corr[
        (
            marker_corr["marker_set"].eq("B_plasma")
            & marker_corr["cell_type"].eq("B_plasma")
        )
        | (
            marker_corr["marker_set"].eq("Absorptive")
            & marker_corr["cell_type"].eq("Absorptive")
        )
        | (
            marker_corr["marker_set"].eq("Fibroblast")
            & marker_corr["cell_type"].eq("Fibroblast")
        )
        | (
            marker_corr["marker_set"].eq("Macrophage")
            & marker_corr["cell_type"].eq("Macrophage")
        )
    ].sort_values("spearman_r", ascending=False)

    slide_corr_valid = slide_corr.dropna(subset=["spearman_r"]).copy()
    paired_corr_valid = paired_corr.dropna(subset=["spearman_r"]).copy()

    overview = {
        "n_focus_metabolites": int(len(focus_metabolites)),
        "focus_metabolites": focus_metabolites,
        "strongest_positive_slide_level_correlations": slide_corr_valid.head(15).to_dict(orient="records"),
        "strongest_negative_slide_level_correlations": slide_corr_valid.tail(15)
        .sort_values("spearman_r")
        .to_dict(orient="records"),
        "strongest_positive_paired_difference_correlations": paired_corr_valid.head(15).to_dict(orient="records"),
        "strongest_negative_paired_difference_correlations": paired_corr_valid.tail(15)
        .sort_values("spearman_r")
        .to_dict(orient="records"),
        "selected_marker_validation": marker_expected.to_dict(orient="records"),
    }
    with (output_dir / "harreman_deconv_interpretation_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
