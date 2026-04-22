"""Run sensitivity checks for spatial high-region definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
REGION_DIR = OUTPUT_ROOT / "spatial_region_validation"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "region_sensitivity"


REGIONS = [
    "epithelial_bile_region",
    "epithelial_absorptive_region",
    "stromal_fibrotic_region",
    "immune_inflammatory_region",
]

GENES = [
    "ABCB1",
    "ABCC3",
    "SLC10A2",
    "SLC15A3",
    "SLC1A4",
    "SLC1A5",
    "SLC38A2",
    "SLC51A",
    "SLC51B",
    "SLC5A1",
    "SLC7A5",
    "SLCO2B1",
]

PRIORITY_PAIRS = {
    ("epithelial_bile_region", "SLC5A1"),
    ("epithelial_bile_region", "SLC10A2"),
    ("epithelial_bile_region", "SLC51A"),
    ("epithelial_bile_region", "SLC51B"),
    ("epithelial_bile_region", "ABCB1"),
    ("epithelial_bile_region", "ABCC3"),
    ("epithelial_absorptive_region", "SLC5A1"),
    ("epithelial_absorptive_region", "SLC10A2"),
    ("epithelial_absorptive_region", "SLC51A"),
    ("epithelial_absorptive_region", "SLC51B"),
    ("stromal_fibrotic_region", "SLC38A2"),
    ("stromal_fibrotic_region", "SLC15A3"),
    ("stromal_fibrotic_region", "SLCO2B1"),
    ("immune_inflammatory_region", "SLCO2B1"),
}


def summarize_one_fraction(spots: pd.DataFrame, top_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    enrichment_rows = []
    sample_rows = []
    for region in REGIONS:
        score = f"{region}_score"
        cutoff = spots.groupby("sample", observed=True)[score].transform(
            lambda values: values.quantile(1 - top_fraction)
        )
        high = spots[score] >= cutoff
        for gene in GENES:
            inside = spots.loc[high, gene]
            outside = spots.loc[~high, gene]
            enrichment_rows.append(
                {
                    "top_fraction": top_fraction,
                    "region": region,
                    "gene": gene,
                    "mean_expression_in_region": float(inside.mean()),
                    "mean_expression_outside_region": float(outside.mean()),
                    "in_region_minus_outside": float(inside.mean() - outside.mean()),
                    "in_region_to_outside_ratio": float(inside.mean() / outside.mean()) if outside.mean() > 0 else np.nan,
                    "n_in_region_spots": int(high.sum()),
                    "n_outside_region_spots": int((~high).sum()),
                    "priority": (region, gene) in PRIORITY_PAIRS,
                }
            )

        grouped = spots.assign(_high_region=high).groupby(
            ["sample", "Disease Label", "Patient_ID", "_high_region"],
            observed=True,
        )
        means = grouped[GENES].mean().reset_index()
        means = means[means["_high_region"]].drop(columns="_high_region")
        means["region"] = region
        means["top_fraction"] = top_fraction
        sample_rows.append(means)

    return pd.DataFrame(enrichment_rows), pd.concat(sample_rows, ignore_index=True)


def paired_differences(sample_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (top_fraction, region, patient_id), group in sample_summary.groupby(
        ["top_fraction", "region", "Patient_ID"],
        observed=True,
    ):
        adjacent = group[group["Disease Label"].eq("Adjacent")]
        fibrotic = group[group["Disease Label"].eq("Fibrotic")]
        if adjacent.empty or fibrotic.empty:
            continue
        for gene in GENES:
            adjacent_mean = float(adjacent[gene].mean())
            fibrotic_mean = float(fibrotic[gene].mean())
            rows.append(
                {
                    "top_fraction": top_fraction,
                    "region": region,
                    "Patient_ID": patient_id,
                    "gene": gene,
                    "adjacent_region_mean_expression": adjacent_mean,
                    "fibrotic_region_mean_expression": fibrotic_mean,
                    "fibrotic_minus_adjacent_region_expression": fibrotic_mean - adjacent_mean,
                    "priority": (region, gene) in PRIORITY_PAIRS,
                }
            )
    return pd.DataFrame(rows)


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["top_fraction", "region", "gene", "priority"], observed=True)
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
    )


def stability_summary(summary: pd.DataFrame) -> pd.DataFrame:
    priority = summary[summary["priority"]].copy()
    rows = []
    for (region, gene), group in priority.groupby(["region", "gene"], observed=True):
        diffs = group.sort_values("top_fraction")["mean_fibrotic_minus_adjacent_region_expression"].to_numpy()
        signs = np.sign(diffs)
        nonzero = signs[signs != 0]
        rows.append(
            {
                "region": region,
                "gene": gene,
                "n_fractions": int(len(group)),
                "mean_difference_across_fractions": float(np.mean(diffs)),
                "min_difference": float(np.min(diffs)),
                "max_difference": float(np.max(diffs)),
                "same_sign_all_fractions": bool(len(nonzero) > 0 and len(set(nonzero)) == 1),
                "mean_positive_patient_fraction": float(group["positive_patient_fraction"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_difference_across_fractions")


def plot_priority_sensitivity(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary[summary["priority"]].copy()
    plot_df["label"] = plot_df["region"] + "::" + plot_df["gene"]
    order = (
        plot_df.groupby("label", observed=True)["mean_fibrotic_minus_adjacent_region_expression"]
        .mean()
        .sort_values()
        .index
    )
    plt.figure(figsize=(11, 9))
    sns.lineplot(
        data=plot_df,
        x="top_fraction",
        y="mean_fibrotic_minus_adjacent_region_expression",
        hue="label",
        hue_order=order,
        marker="o",
    )
    plt.axhline(0, color="#666666", linewidth=0.8)
    plt.title("Region-definition sensitivity for priority transporter signals")
    plt.xlabel("top fraction of spots per slide used as high-signal region")
    plt.ylabel("mean paired fibrotic minus adjacent expression")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_stability_heatmap(stability: pd.DataFrame, output_path: Path) -> None:
    plot_df = stability.copy()
    plot_df["label"] = plot_df["region"] + "::" + plot_df["gene"]
    heatmap = plot_df.set_index("label")[
        ["mean_difference_across_fractions", "mean_positive_patient_fraction"]
    ].sort_values("mean_difference_across_fractions")
    plt.figure(figsize=(7, 7))
    sns.heatmap(heatmap, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.3, linecolor="white")
    plt.title("Priority signal stability across region thresholds")
    plt.xlabel("stability metric")
    plt.ylabel("region and gene")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-fractions", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    spots = pd.read_csv(REGION_DIR / "spot_region_scores_selected.csv")
    all_enrichment = []
    all_samples = []
    for top_fraction in args.top_fractions:
        enrichment, sample_summary = summarize_one_fraction(spots, top_fraction)
        all_enrichment.append(enrichment)
        all_samples.append(sample_summary)

    enrichment = pd.concat(all_enrichment, ignore_index=True)
    sample_summary = pd.concat(all_samples, ignore_index=True)
    paired = paired_differences(sample_summary)
    summary = paired_summary(paired)
    stability = stability_summary(summary)

    enrichment.to_csv(output_dir / "region_threshold_enrichment_sensitivity.csv", index=False)
    sample_summary.to_csv(output_dir / "region_threshold_expression_by_sample.csv", index=False)
    paired.to_csv(output_dir / "region_threshold_paired_differences.csv", index=False)
    summary.to_csv(output_dir / "region_threshold_paired_difference_summary.csv", index=False)
    stability.to_csv(output_dir / "priority_region_signal_stability.csv", index=False)

    plot_priority_sensitivity(summary, output_dir / "priority_region_threshold_sensitivity.png")
    plot_stability_heatmap(stability, output_dir / "priority_region_signal_stability.png")

    overview = {
        "top_fractions": args.top_fractions,
        "n_priority_pairs": len(PRIORITY_PAIRS),
        "n_stable_same_sign_priority_pairs": int(stability["same_sign_all_fractions"].sum()),
        "stable_priority_pairs": stability[stability["same_sign_all_fractions"]].to_dict(orient="records"),
        "least_stable_priority_pairs": stability[~stability["same_sign_all_fractions"]].to_dict(orient="records"),
    }
    with (output_dir / "region_sensitivity_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
