"""Run patient-paired models for prioritized spatial metabolism signals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "patient_paired_models"


FOCUS_METABOLITES = {
    "Bile acid",
    "Cholic acid",
    "Urea",
    "L-Histidine",
    "L-Serine",
    "Citric acid",
    "L-Cysteine",
}

FOCUS_GENES = {
    "SLC5A1",
    "SLC10A2",
    "SLC51A",
    "SLC51B",
    "ABCB1",
    "ABCC3",
    "SLC38A2",
    "SLC15A3",
    "SLCO2B1",
    "SLC7A5",
    "SLC1A5",
    "SLC1A4",
}

FOCUS_REGION_GENE_PAIRS = {
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

FOCUS_METABOLISM = {
    "bile_acid_transport",
    "epithelial_repair",
    "ecm_remodeling",
    "inflammation",
    "fatty_acid_metabolism",
}

FOCUS_CELL_TYPES = {
    "Absorptive",
    "Microfold",
    "Fibroblast",
    "Myofibroblast",
    "Macrophage",
    "Monocyte",
    "B_plasma",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    valid = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if valid.sum() == 0:
        return pd.Series(q, index=p_values.index)
    valid_idx = np.where(valid)[0]
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q[valid_idx[order]] = adjusted
    return pd.Series(q, index=p_values.index)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def one_sample_paired_stats(values: pd.Series, rng: np.random.Generator, n_bootstrap: int) -> dict[str, float]:
    clean = values.dropna().astype(float).to_numpy()
    n = len(clean)
    if n == 0:
        return {
            "n_patients": 0,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "std_difference": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "cohens_dz": np.nan,
            "positive_patient_fraction": np.nan,
            "paired_t_p_value": np.nan,
            "wilcoxon_p_value": np.nan,
        }

    mean = float(np.mean(clean))
    median = float(np.median(clean))
    std = float(np.std(clean, ddof=1)) if n > 1 else np.nan
    ci_low, ci_high = bootstrap_ci(clean, rng, n_bootstrap)
    cohens_dz = mean / std if np.isfinite(std) and std > 0 else np.nan
    positive_fraction = float((clean > 0).mean())

    if n > 1 and np.isfinite(std) and std > 0:
        paired_t_p = float(stats.ttest_1samp(clean, popmean=0).pvalue)
    else:
        paired_t_p = np.nan

    nonzero = clean[clean != 0]
    if len(nonzero) > 0:
        try:
            wilcoxon_p = float(stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            wilcoxon_p = np.nan
    else:
        wilcoxon_p = np.nan

    return {
        "n_patients": int(n),
        "mean_difference": mean,
        "median_difference": median,
        "std_difference": std,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "cohens_dz": cohens_dz,
        "positive_patient_fraction": positive_fraction,
        "paired_t_p_value": paired_t_p,
        "wilcoxon_p_value": wilcoxon_p,
    }


def long_from_harreman() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_ROOT / "harreman_all_slides" / "harreman_patient_paired_metabolite_differences.csv")
    return df.rename(columns={"metabolite": "feature", "fibrotic_minus_adjacent_z": "difference"})[
        ["Patient_ID", "feature", "difference"]
    ].assign(analysis_layer="harreman_metabolite_z", feature_class="metabolite")


def long_from_slide_transporters() -> pd.DataFrame:
    df = pd.read_csv(
        OUTPUT_ROOT / "transporter_spatial_validation" / "paired_transporter_gene_expression_differences.csv"
    )
    return df.rename(columns={"gene": "feature", "fibrotic_minus_adjacent_expression": "difference"})[
        ["Patient_ID", "feature", "difference"]
    ].assign(analysis_layer="slide_transporter_expression", feature_class="transporter_gene")


def long_from_region_transporters() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_ROOT / "spatial_region_validation" / "paired_region_transporter_differences.csv")
    df["feature"] = df["region"] + "::" + df["gene"]
    return df.rename(columns={"fibrotic_minus_adjacent_region_expression": "difference"})[
        ["Patient_ID", "feature", "region", "gene", "difference"]
    ].assign(analysis_layer="region_transporter_expression", feature_class="region_transporter_gene")


def long_from_wide(path: Path, analysis_layer: str, feature_class: str, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    diff_cols = [column for column in df.columns if column.endswith(suffix)]
    for column in diff_cols:
        feature = column.removesuffix(suffix)
        tmp = df[["Patient_ID", column]].rename(columns={column: "difference"})
        tmp["feature"] = feature
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True).assign(
        analysis_layer=analysis_layer,
        feature_class=feature_class,
    )


def is_priority(row: pd.Series) -> bool:
    layer = row["analysis_layer"]
    feature = row["feature"]
    if layer == "harreman_metabolite_z":
        return feature in FOCUS_METABOLITES
    if layer == "slide_transporter_expression":
        return feature in FOCUS_GENES
    if layer == "region_transporter_expression":
        region = row.get("region", "")
        gene = row.get("gene", "")
        return (region, gene) in FOCUS_REGION_GENE_PAIRS
    if layer == "spatial_metabolism_score":
        return feature in FOCUS_METABOLISM
    if layer == "spatial_deconvolution_proportion":
        return feature in FOCUS_CELL_TYPES
    return False


def build_model_results(long: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    group_cols = ["analysis_layer", "feature_class", "feature"]
    if "region" in long.columns:
        long["region"] = long["region"].fillna("")
    if "gene" in long.columns:
        long["gene"] = long["gene"].fillna("")

    for keys, group in long.groupby(group_cols, observed=True):
        analysis_layer, feature_class, feature = keys
        stats_row = one_sample_paired_stats(group["difference"], rng, n_bootstrap)
        row = {
            "analysis_layer": analysis_layer,
            "feature_class": feature_class,
            "feature": feature,
            "region": group["region"].iloc[0] if "region" in group else "",
            "gene": group["gene"].iloc[0] if "gene" in group else "",
        } | stats_row
        rows.append(row)

    results = pd.DataFrame(rows)
    results["paired_t_fdr"] = results.groupby("analysis_layer", observed=True)["paired_t_p_value"].transform(
        benjamini_hochberg
    )
    results["wilcoxon_fdr"] = results.groupby("analysis_layer", observed=True)["wilcoxon_p_value"].transform(
        benjamini_hochberg
    )
    results["direction"] = np.where(
        results["mean_difference"] > 0,
        "fibrotic_higher",
        np.where(results["mean_difference"] < 0, "adjacent_higher", "no_mean_shift"),
    )
    results["priority"] = results.apply(is_priority, axis=1)
    return results.sort_values(["priority", "analysis_layer", "paired_t_p_value"], ascending=[False, True, True])


def plot_priority_effects(priority: pd.DataFrame, output_path: Path) -> None:
    plot_df = priority.copy()
    plot_df = plot_df[plot_df["n_patients"] >= 5].copy()
    plot_df["abs_dz"] = plot_df["cohens_dz"].abs()
    plot_df = plot_df.sort_values("abs_dz", ascending=False).head(45)
    plot_df["label"] = plot_df["analysis_layer"] + " | " + plot_df["feature"]
    plot_df = plot_df.sort_values("cohens_dz")
    colors = np.where(plot_df["cohens_dz"] >= 0, "#C44E52", "#4C72B0")

    plt.figure(figsize=(12, 12))
    plt.barh(plot_df["label"], plot_df["cohens_dz"], color=colors)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Priority patient-paired effect sizes")
    plt.xlabel("paired effect size, Cohen dz")
    plt.ylabel("analysis layer and feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_layer_volcano(results: pd.DataFrame, output_path: Path) -> None:
    plot_df = results.copy()
    plot_df = plot_df[plot_df["n_patients"] >= 5].copy()
    plot_df["minus_log10_p"] = -np.log10(plot_df["paired_t_p_value"].clip(lower=1e-300))
    plot_df["priority_label"] = np.where(plot_df["priority"], "priority", "other")
    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=plot_df,
        x="cohens_dz",
        y="minus_log10_p",
        hue="analysis_layer",
        style="priority_label",
        alpha=0.75,
    )
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.title("Patient-paired model effect sizes and p-values")
    plt.xlabel("paired effect size, Cohen dz")
    plt.ylabel("-log10 paired t-test p-value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260422)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    long_tables = [
        long_from_harreman(),
        long_from_slide_transporters(),
        long_from_region_transporters(),
        long_from_wide(
            OUTPUT_ROOT / "metabolism_scores" / "paired_patient_metabolism_score_differences.csv",
            "spatial_metabolism_score",
            "metabolism_gene_set",
            "_fibrotic_minus_adjacent",
        ),
        long_from_wide(
            OUTPUT_ROOT / "spatial_deconvolution" / "paired_patient_cell_type_differences.csv",
            "spatial_deconvolution_proportion",
            "deconvolved_cell_type",
            "_fibrotic_minus_adjacent",
        ),
    ]
    long = pd.concat(long_tables, ignore_index=True, sort=False)
    long["Patient_ID"] = long["Patient_ID"].astype(str)
    long.to_csv(output_dir / "paired_model_input_long.csv", index=False)

    results = build_model_results(long, args.n_bootstrap, args.seed)
    priority = results[results["priority"]].copy()
    significant = results[
        (results["n_patients"] >= 5)
        & (results["paired_t_fdr"] < 0.10)
        & (results["wilcoxon_fdr"] < 0.10)
    ].copy()

    results.to_csv(output_dir / "patient_paired_model_results.csv", index=False)
    priority.to_csv(output_dir / "priority_patient_paired_model_results.csv", index=False)
    significant.to_csv(output_dir / "fdr_supported_patient_paired_model_results.csv", index=False)

    plot_priority_effects(priority, output_dir / "priority_patient_paired_effect_sizes.png")
    plot_layer_volcano(results, output_dir / "patient_paired_model_volcano.png")

    overview = {
        "n_input_patient_feature_rows": int(len(long)),
        "n_modeled_features": int(len(results)),
        "n_priority_features": int(len(priority)),
        "n_fdr_supported_features": int(len(significant)),
        "fdr_threshold": 0.10,
        "top_priority_by_effect_size": priority.assign(abs_dz=lambda df: df["cohens_dz"].abs())
        .sort_values("abs_dz", ascending=False)
        .head(25)
        .drop(columns=["abs_dz"])
        .to_dict(orient="records"),
        "fdr_supported_features": significant.sort_values("paired_t_fdr")
        .head(25)
        .to_dict(orient="records"),
    }
    with (output_dir / "patient_paired_models_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
