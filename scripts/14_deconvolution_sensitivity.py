"""Compare NNLS deconvolution signals with marker-gene score sensitivity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
DECONV_DIR = OUTPUT_ROOT / "spatial_deconvolution"
INTERPRETATION_DIR = OUTPUT_ROOT / "harreman_deconv_interpretation"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "deconvolution_sensitivity"


EXPECTED_MARKER_TO_DECONV = {
    "B_plasma": "B_plasma",
    "B_cells": "Mature_B",
    "Absorptive": "Absorptive",
    "Secretory": "Secretory",
    "Fibroblast": "Fibroblast",
    "Myofibroblast": "Myofibroblast",
    "Macrophage": "Macrophage",
    "Endothelial": "Vascular_endothelia",
    "Pericyte": "Pericyte",
}


def paired_marker_differences(marker_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (patient_id, marker_set), group in marker_scores.groupby(["Patient_ID", "marker_set"], observed=True):
        adjacent = group[group["Disease Label"].eq("Adjacent")]
        fibrotic = group[group["Disease Label"].eq("Fibrotic")]
        if adjacent.empty or fibrotic.empty:
            continue
        rows.append(
            {
                "Patient_ID": patient_id,
                "marker_set": marker_set,
                "adjacent_marker_score": float(adjacent["marker_score"].mean()),
                "fibrotic_marker_score": float(fibrotic["marker_score"].mean()),
                "marker_fibrotic_minus_adjacent": float(
                    fibrotic["marker_score"].mean() - adjacent["marker_score"].mean()
                ),
                "n_adjacent_slides": int(adjacent["sample"].nunique()),
                "n_fibrotic_slides": int(fibrotic["sample"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_comparison(marker_paired: pd.DataFrame, deconv_paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for marker_set, cell_type in EXPECTED_MARKER_TO_DECONV.items():
        marker = marker_paired[marker_paired["marker_set"].eq(marker_set)][
            ["Patient_ID", "marker_fibrotic_minus_adjacent"]
        ]
        column = f"{cell_type}_fibrotic_minus_adjacent"
        if column not in deconv_paired:
            continue
        deconv = deconv_paired[["Patient_ID", column]].rename(columns={column: "deconv_fibrotic_minus_adjacent"})
        merged = marker.merge(deconv, on="Patient_ID", how="inner")
        if len(merged) >= 5 and merged["marker_fibrotic_minus_adjacent"].nunique() > 1:
            rho, p_value = spearmanr(merged["marker_fibrotic_minus_adjacent"], merged["deconv_fibrotic_minus_adjacent"])
        else:
            rho, p_value = np.nan, np.nan
        marker_mean = float(merged["marker_fibrotic_minus_adjacent"].mean())
        deconv_mean = float(merged["deconv_fibrotic_minus_adjacent"].mean())
        rows.append(
            {
                "marker_set": marker_set,
                "deconvolved_cell_type": cell_type,
                "n_patients": int(len(merged)),
                "marker_mean_fibrotic_minus_adjacent": marker_mean,
                "deconv_mean_fibrotic_minus_adjacent": deconv_mean,
                "same_mean_direction": bool(np.sign(marker_mean) == np.sign(deconv_mean)),
                "marker_positive_patient_fraction": float((merged["marker_fibrotic_minus_adjacent"] > 0).mean()),
                "deconv_positive_patient_fraction": float((merged["deconv_fibrotic_minus_adjacent"] > 0).mean()),
                "paired_difference_spearman_r": rho,
                "paired_difference_spearman_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def long_patient_comparison(marker_paired: pd.DataFrame, deconv_paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for marker_set, cell_type in EXPECTED_MARKER_TO_DECONV.items():
        marker = marker_paired[marker_paired["marker_set"].eq(marker_set)][
            ["Patient_ID", "marker_fibrotic_minus_adjacent"]
        ]
        column = f"{cell_type}_fibrotic_minus_adjacent"
        if column not in deconv_paired:
            continue
        deconv = deconv_paired[["Patient_ID", column]].rename(columns={column: "deconv_fibrotic_minus_adjacent"})
        merged = marker.merge(deconv, on="Patient_ID", how="inner")
        merged["marker_set"] = marker_set
        merged["deconvolved_cell_type"] = cell_type
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def plot_comparison_summary(comparison: pd.DataFrame, output_path: Path) -> None:
    plot_df = comparison.copy()
    plot_df["label"] = plot_df["marker_set"] + " vs " + plot_df["deconvolved_cell_type"]
    plot_df = plot_df.sort_values("marker_mean_fibrotic_minus_adjacent")
    x = np.arange(len(plot_df))
    width = 0.38
    plt.figure(figsize=(11, 7))
    plt.barh(x - width / 2, plot_df["marker_mean_fibrotic_minus_adjacent"], height=width, label="marker score")
    plt.barh(x + width / 2, plot_df["deconv_mean_fibrotic_minus_adjacent"], height=width, label="NNLS deconvolution")
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.yticks(x, plot_df["label"])
    plt.title("Paired marker-score and deconvolution direction sensitivity")
    plt.xlabel("mean fibrotic minus adjacent")
    plt.ylabel("marker and deconvolved cell type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_patient_scatter(long_df: pd.DataFrame, output_path: Path) -> None:
    grid = sns.FacetGrid(long_df, col="marker_set", col_wrap=3, sharex=False, sharey=False, height=3)
    grid.map_dataframe(
        sns.scatterplot,
        x="marker_fibrotic_minus_adjacent",
        y="deconv_fibrotic_minus_adjacent",
    )
    for ax in grid.axes.flatten():
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.axvline(0, color="#777777", linewidth=0.7)
    grid.set_axis_labels("marker score F-A", "NNLS proportion F-A")
    grid.fig.suptitle("Patient-paired marker versus NNLS differences", y=1.02)
    grid.tight_layout()
    grid.savefig(output_path, dpi=220)
    plt.close(grid.fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    marker_scores = pd.read_csv(INTERPRETATION_DIR / "marker_scores_by_slide.csv")
    marker_slide_corr = pd.read_csv(INTERPRETATION_DIR / "marker_deconvolution_correlations.csv")
    deconv_paired = pd.read_csv(DECONV_DIR / "paired_patient_cell_type_differences.csv")

    marker_paired = paired_marker_differences(marker_scores)
    comparison = build_comparison(marker_paired, deconv_paired)
    patient_long = long_patient_comparison(marker_paired, deconv_paired)
    expected_slide_corr = marker_slide_corr[
        marker_slide_corr.apply(
            lambda row: EXPECTED_MARKER_TO_DECONV.get(row["marker_set"]) == row["cell_type"],
            axis=1,
        )
    ].copy()

    marker_paired.to_csv(output_dir / "paired_marker_score_differences.csv", index=False)
    comparison.to_csv(output_dir / "marker_deconvolution_paired_sensitivity.csv", index=False)
    patient_long.to_csv(output_dir / "marker_deconvolution_patient_long.csv", index=False)
    expected_slide_corr.to_csv(output_dir / "expected_marker_deconvolution_slide_correlations.csv", index=False)

    plot_comparison_summary(comparison, output_dir / "marker_deconvolution_paired_direction.png")
    plot_patient_scatter(patient_long, output_dir / "marker_deconvolution_patient_scatter.png")

    overview = {
        "expected_marker_to_deconvolution_pairs": EXPECTED_MARKER_TO_DECONV,
        "n_expected_pairs": len(EXPECTED_MARKER_TO_DECONV),
        "n_same_mean_direction": int(comparison["same_mean_direction"].sum()),
        "paired_sensitivity": comparison.to_dict(orient="records"),
        "expected_slide_level_correlations": expected_slide_corr.to_dict(orient="records"),
    }
    with (output_dir / "deconvolution_sensitivity_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
