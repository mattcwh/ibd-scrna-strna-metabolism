"""Create consolidated report-ready figures for the strongest integrated signals."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "consolidated_figures"


METABOLITE_ORDER = [
    "Urea",
    "Bile acid",
    "Cholic acid",
    "Citric acid",
    "L-Histidine",
    "L-Serine",
    "L-Cysteine",
]

REGION_PAIR_LABELS = {
    "epithelial_bile_region::SLC5A1": "Epithelial bile | SLC5A1",
    "epithelial_bile_region::SLC10A2": "Epithelial bile | SLC10A2",
    "epithelial_bile_region::SLC51A": "Epithelial bile | SLC51A",
    "epithelial_bile_region::SLC51B": "Epithelial bile | SLC51B",
    "stromal_fibrotic_region::SLC38A2": "Stromal fibrotic | SLC38A2",
    "stromal_fibrotic_region::SLC15A3": "Stromal fibrotic | SLC15A3",
    "stromal_fibrotic_region::SLCO2B1": "Stromal fibrotic | SLCO2B1",
    "immune_inflammatory_region::SLCO2B1": "Immune inflammatory | SLCO2B1",
}

SCRNA_FEATURES = [
    "bile_acid_transport_mean",
    "epithelial_bile_urea_transport_mean",
    "fibrotic_amino_acid_transport_mean",
    "ecm_remodeling_mean",
    "inflammation_mean",
]

SCRNA_CELL_TYPES = [
    "Enterocytes-MT-hi",
    "Enterocyte-ANPEP-hi",
    "Enterocyte-FABP-hi",
    "M2-like macrophage-A2M-hi",
    "M2-like macrophage",
    "CD63+CD81+ macrophage",
    "Inflammatory fibroblasts",
    "Myofibroblasts",
    "Monocyte",
    "Neutrophils",
]


def color_for_values(values: pd.Series) -> list[str]:
    return ["#C44E52" if value > 0 else "#4C72B0" for value in values]


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "integrated": pd.read_csv(OUTPUT_ROOT / "integrated_evidence" / "integrated_metabolite_evidence.csv"),
        "priority_models": pd.read_csv(
            OUTPUT_ROOT / "patient_paired_models" / "priority_patient_paired_model_results.csv"
        ),
        "region_stability": pd.read_csv(OUTPUT_ROOT / "region_sensitivity" / "priority_region_signal_stability.csv"),
        "deconv_sensitivity": pd.read_csv(
            OUTPUT_ROOT / "deconvolution_sensitivity" / "marker_deconvolution_paired_sensitivity.csv"
        ),
        "scrna_summary": pd.read_csv(
            OUTPUT_ROOT / "scrna_metabolism_transporters" / "scrna_cell_type_feature_summary.csv"
        ),
        "healthy_spatial": pd.read_csv(
            OUTPUT_ROOT / "healthy_control_comparisons" / "spatial_disease_control_comparisons.csv"
        ),
        "healthy_scrna": pd.read_csv(
            OUTPUT_ROOT / "healthy_control_comparisons" / "scrna_healthy_status_comparisons.csv"
        ),
    }


def metabolite_direction_table(integrated: pd.DataFrame) -> pd.DataFrame:
    table = integrated[integrated["metabolite"].isin(METABOLITE_ORDER)].copy()
    table["metabolite"] = pd.Categorical(table["metabolite"], METABOLITE_ORDER, ordered=True)
    return table.sort_values("metabolite")


def region_pair_table(region_stability: pd.DataFrame) -> pd.DataFrame:
    table = region_stability.copy()
    table["key"] = table["region"] + "::" + table["gene"]
    table = table[table["key"].isin(REGION_PAIR_LABELS)].copy()
    table["label"] = table["key"].map(REGION_PAIR_LABELS)
    order = list(REGION_PAIR_LABELS.values())
    table["label"] = pd.Categorical(table["label"], order, ordered=True)
    return table.sort_values("label")


def scrna_matrix(scrna_summary: pd.DataFrame) -> pd.DataFrame:
    summary = scrna_summary.copy()
    summary = summary[summary["annotation2v2"].isin(SCRNA_CELL_TYPES)]
    collapsed = summary.groupby("annotation2v2", observed=True)[SCRNA_FEATURES].mean()
    collapsed = collapsed.reindex(SCRNA_CELL_TYPES)
    standardized = (collapsed - collapsed.mean(axis=0)) / collapsed.std(axis=0, ddof=0).replace(0, np.nan)
    return standardized.fillna(0)


def deconv_table(deconv: pd.DataFrame) -> pd.DataFrame:
    keep = ["B_plasma", "Absorptive", "Secretory", "Macrophage", "Fibroblast", "Myofibroblast"]
    table = deconv[deconv["marker_set"].isin(keep)].copy()
    table["label"] = table["marker_set"] + " / " + table["deconvolved_cell_type"]
    table = table.sort_values("marker_mean_fibrotic_minus_adjacent")
    return table


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )


def make_main_figure(tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[1.05, 1.0], hspace=0.34, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    metabolite = metabolite_direction_table(tables["integrated"])
    y = np.arange(len(metabolite))
    values = metabolite["mean_fibrotic_minus_adjacent_harreman_z"]
    ax_a.barh(y, values, color=color_for_values(values))
    ax_a.axvline(0, color="#555555", linewidth=0.9)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(metabolite["metabolite"])
    ax_a.set_xlabel("mean paired fibrotic minus adjacent Harreman z")
    ax_a.set_title("Priority metabolite directions")
    for yi, row in zip(y, metabolite.itertuples(), strict=True):
        ax_a.text(
            row.mean_fibrotic_minus_adjacent_harreman_z,
            yi,
            f"  n={row.harreman_n_patients}, +={row.harreman_positive_patient_fraction:.2f}",
            va="center",
            ha="left" if row.mean_fibrotic_minus_adjacent_harreman_z >= 0 else "right",
            fontsize=8,
        )
    add_panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    region = region_pair_table(tables["region_stability"])
    region = region.sort_values("mean_difference_across_fractions")
    y = np.arange(len(region))
    values = region["mean_difference_across_fractions"]
    xerr = np.vstack(
        [
            values - region["min_difference"],
            region["max_difference"] - values,
        ]
    )
    ax_b.barh(y, values, xerr=xerr, color=color_for_values(values), alpha=0.88)
    ax_b.axvline(0, color="#555555", linewidth=0.9)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(region["label"])
    ax_b.set_xlabel("mean paired F-A expression across region thresholds")
    ax_b.set_title("Region-threshold-stable transporter signals")
    add_panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    matrix = scrna_matrix(tables["scrna_summary"])
    nice_cols = {
        "bile_acid_transport_mean": "bile acid",
        "epithelial_bile_urea_transport_mean": "epi bile/urea",
        "fibrotic_amino_acid_transport_mean": "fibrotic AA",
        "ecm_remodeling_mean": "ECM",
        "inflammation_mean": "inflammation",
    }
    matrix = matrix.rename(columns=nice_cols)
    sns.heatmap(matrix, ax=ax_c, cmap="vlag", center=0, linewidths=0.2, linecolor="white", cbar_kws={"label": "cell-type z"})
    ax_c.set_title("scRNA-seq support by cell type")
    ax_c.set_xlabel("feature")
    ax_c.set_ylabel("cell type")
    add_panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    deconv = deconv_table(tables["deconv_sensitivity"])
    y = np.arange(len(deconv))
    width = 0.38
    ax_d.barh(y - width / 2, deconv["marker_mean_fibrotic_minus_adjacent"], height=width, label="marker score", color="#55A868")
    ax_d.barh(y + width / 2, deconv["deconv_mean_fibrotic_minus_adjacent"], height=width, label="NNLS", color="#8172B2")
    ax_d.axvline(0, color="#555555", linewidth=0.9)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(deconv["label"])
    ax_d.set_xlabel("mean paired fibrotic minus adjacent")
    ax_d.set_title("Marker versus deconvolution sensitivity")
    ax_d.legend(loc="lower right")
    add_panel_label(ax_d, "D")

    fig.suptitle("Integrated spatial metabolism evidence summary", fontsize=18, y=0.995)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_interpretation_schematic(tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis("off")

    boxes = [
        {
            "xy": (0.04, 0.58),
            "title": "Adjacent-shifted epithelial axis",
            "body": "Bile acid, cholic acid, and urea\nSLC51A/B, SLC10A2, SLC5A1\nEnterocyte and absorptive-region support",
            "color": "#4C72B0",
        },
        {
            "xy": (0.38, 0.58),
            "title": "Fibrotic-shifted transporter axis",
            "body": "L-histidine, L-serine, citric acid, L-cysteine\nSLC38A2, SLC15A3, SLCO2B1\nMacrophage and stromal-region support",
            "color": "#C44E52",
        },
        {
            "xy": (0.72, 0.58),
            "title": "Sensitivity and statistics",
            "body": "Region directions stable across 10%, 20%, 30% thresholds\nNo feature passes FDR < 0.10 in patient-paired models\nHealthy-control spatial comparison is descriptive, n=2",
            "color": "#8172B2",
        },
        {
            "xy": (0.21, 0.13),
            "title": "Interpretation",
            "body": "Most robust current finding: loss of epithelial bile-acid and urea transporter programs in fibrotic slides.\nFibrotic amino-acid transporter signals are plausible but should remain hypothesis-generating.",
            "color": "#55A868",
            "width": 0.58,
        },
    ]

    for box in boxes:
        x, y = box["xy"]
        width = box.get("width", 0.25)
        height = 0.27
        rect = plt.Rectangle((x, y), width, height, facecolor="white", edgecolor=box["color"], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.015, y + height - 0.04, box["title"], fontsize=13, fontweight="bold", color=box["color"], va="top")
        ax.text(x + 0.015, y + height - 0.095, box["body"], fontsize=11, va="top", linespacing=1.35)

    arrowprops = dict(arrowstyle="->", color="#555555", linewidth=1.5)
    ax.annotate("", xy=(0.35, 0.50), xytext=(0.19, 0.58), arrowprops=arrowprops)
    ax.annotate("", xy=(0.50, 0.50), xytext=(0.50, 0.58), arrowprops=arrowprops)
    ax.annotate("", xy=(0.66, 0.50), xytext=(0.84, 0.58), arrowprops=arrowprops)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_healthy_control_panel(tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    spatial = tables["healthy_spatial"].copy()
    scrna = tables["healthy_scrna"].copy()

    spatial_keep = [
        "epithelial_bile_region::SLC5A1",
        "epithelial_bile_region::SLC10A2",
        "bile_acid_transport",
        "inflammation",
        "ecm_remodeling",
        "B_plasma",
        "stromal_fibrotic_region::SLC15A3",
        "stromal_fibrotic_region::SLCO2B1",
    ]
    spatial_plot = spatial[
        spatial["feature"].isin(spatial_keep)
        & spatial["comparison"].isin(["Adjacent_minus_DiseaseControl", "Fibrotic_minus_DiseaseControl"])
    ].copy()
    spatial_plot["label"] = spatial_plot["feature"] + " | " + spatial_plot["comparison"].str.replace("_minus_DiseaseControl", " vs DC")
    spatial_plot = spatial_plot.sort_values("difference_vs_disease_control")

    scrna_keep = [
        ("Inflammatory fibroblasts", "ecm_remodeling_mean", "F_minus_H"),
        ("Myofibroblasts", "ecm_remodeling_mean", "F_minus_H"),
        ("Enterocyte-FABP-hi", "bile_acid_transport_mean", "F_minus_H"),
        ("Enterocyte-ANPEP-hi", "bile_acid_transport_mean", "F_minus_H"),
        ("Inflammatory fibroblasts", "inflammation_mean", "F_minus_H"),
        ("Enterocyte-ANPEP-hi", "epithelial_repair_mean", "F_minus_H"),
    ]
    mask = np.zeros(len(scrna), dtype=bool)
    for cell_type, feature, comparison in scrna_keep:
        mask |= (
            scrna["annotation2v2"].eq(cell_type)
            & scrna["feature"].eq(feature)
            & scrna["comparison"].eq(comparison)
        )
    scrna_plot = scrna[mask].copy()
    scrna_plot["label"] = (
        scrna_plot["annotation2v2"]
        + " | "
        + scrna_plot["feature"].str.replace("_mean", "").str.replace("_", " ")
    )
    scrna_plot = scrna_plot.sort_values("difference_vs_healthy")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.2, 1.0]})

    colors = color_for_values(spatial_plot["difference_vs_disease_control"])
    axes[0].barh(spatial_plot["label"], spatial_plot["difference_vs_disease_control"], color=colors)
    axes[0].axvline(0, color="#555555", linewidth=0.8)
    axes[0].set_title("Spatial disease-control contrasts")
    axes[0].set_xlabel("mean comparison minus disease-control")
    axes[0].set_ylabel("")
    add_panel_label(axes[0], "A")

    colors = color_for_values(scrna_plot["difference_vs_healthy"])
    axes[1].barh(scrna_plot["label"], scrna_plot["difference_vs_healthy"], color=colors)
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("scRNA-seq disease minus healthy contrasts")
    axes[1].set_xlabel("mean F minus H")
    axes[1].set_ylabel("")
    add_panel_label(axes[1], "B")

    fig.suptitle("Healthy-control comparison summary", fontsize=17, y=0.995)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(tables: dict[str, pd.DataFrame], output_path: Path) -> pd.DataFrame:
    integrated = metabolite_direction_table(tables["integrated"])
    rows = []
    for row in integrated.itertuples():
        evidence = []
        if row.harreman_direction == "adjacent-shifted":
            evidence.append("epithelial/absorptive transporter support")
        else:
            evidence.append("macrophage/stromal transporter support")
        if "SLC" in str(row.top_transporter_harreman_correlations) or "ABC" in str(row.top_transporter_harreman_correlations):
            evidence.append("transporter-Harreman correlation")
        rows.append(
            {
                "metabolite": row.metabolite,
                "direction": row.harreman_direction,
                "mean_fibrotic_minus_adjacent_harreman_z": row.mean_fibrotic_minus_adjacent_harreman_z,
                "n_patients": row.harreman_n_patients,
                "interpretation": row.scrna_supporting_cell_types.split(";")[0] if isinstance(row.scrna_supporting_cell_types, str) else "",
                "evidence_summary": "; ".join(evidence),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables()

    make_main_figure(tables, output_dir / "integrated_evidence_summary_panel.png")
    make_interpretation_schematic(tables, output_dir / "integrated_interpretation_schematic.png")
    make_healthy_control_panel(tables, output_dir / "healthy_control_summary_panel.png")
    summary = write_summary_table(tables, output_dir / "consolidated_priority_signal_summary.csv")

    overview = {
        "figures": [
            "integrated_evidence_summary_panel.png",
            "integrated_interpretation_schematic.png",
            "healthy_control_summary_panel.png",
        ],
        "n_priority_metabolites": int(len(summary)),
        "priority_metabolites": summary["metabolite"].tolist(),
        "interpretation_notes": [
            "Adjacent-shifted bile-acid, cholic-acid, and urea signals have the strongest cross-layer epithelial support.",
            "Fibrotic-shifted amino-acid and organic-anion transporter signals are plausible but should remain hypothesis-generating.",
            "Patient-paired models did not identify FDR-supported features at 0.10.",
            "Region-threshold sensitivity supports the region-level transporter directions.",
        ],
    }
    with (output_dir / "consolidated_figures_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
