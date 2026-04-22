"""Create an integrated evidence table for priority Harreman metabolites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "integrated_evidence"


FOCUS_METABOLITES = [
    "Bile acid",
    "Cholic acid",
    "Urea",
    "L-Histidine",
    "L-Serine",
    "Citric acid",
    "L-Cysteine",
]

SCRNA_FEATURE_BY_METABOLITE = {
    "Bile acid": ["bile_acid_transport_mean", "epithelial_bile_urea_transport_mean"],
    "Cholic acid": ["bile_acid_transport_mean", "epithelial_bile_urea_transport_mean"],
    "Urea": ["epithelial_bile_urea_transport_mean"],
    "L-Histidine": ["fibrotic_amino_acid_transport_mean"],
    "L-Serine": ["fibrotic_amino_acid_transport_mean"],
    "Citric acid": ["fibrotic_amino_acid_transport_mean"],
    "L-Cysteine": ["fibrotic_amino_acid_transport_mean"],
}


def join_records(records: list[str], max_records: int = 4) -> str:
    cleaned = [record for record in records if isinstance(record, str) and record]
    return "; ".join(cleaned[:max_records])


def direction(value: float) -> str:
    if not np.isfinite(value):
        return "not available"
    if value > 0:
        return "fibrotic-shifted"
    if value < 0:
        return "adjacent-shifted"
    return "no mean shift"


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def clean(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    with path.open("w") as handle:
        handle.write("| " + " | ".join(df.columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for _, row in df.iterrows():
            handle.write("| " + " | ".join(clean(row[column]) for column in df.columns) + " |\n")


def top_deconv(metabolite: str, correlations: pd.DataFrame) -> str:
    data = correlations[correlations["metabolite"].eq(metabolite)].copy()
    if data.empty:
        return ""
    data["abs_r"] = data["spearman_r"].abs()
    data = data.sort_values("abs_r", ascending=False).head(4)
    return join_records([f"{row.cell_type} rho {fmt(row.spearman_r)}" for row in data.itertuples()])


def top_transporter_correlations(metabolite: str, correlations: pd.DataFrame) -> str:
    data = correlations[correlations["focus_metabolite"].eq(metabolite)].dropna(subset=["spearman_r"]).copy()
    if data.empty:
        return ""
    data["abs_r"] = data["spearman_r"].abs()
    data = data.sort_values("abs_r", ascending=False).head(4)
    return join_records([f"{row.gene} rho {fmt(row.spearman_r)}" for row in data.itertuples()])


def transporter_direction_summary(metabolite: str, gene_map: pd.DataFrame, diff: pd.DataFrame) -> str:
    genes = gene_map.loc[gene_map["focus_metabolite"].eq(metabolite), "gene"].unique()
    data = diff[diff["gene"].isin(genes)].copy()
    if data.empty:
        return ""
    data["abs_diff"] = data["mean_fibrotic_minus_adjacent_expression"].abs()
    data = data.sort_values("abs_diff", ascending=False).head(4)
    return join_records(
        [
            f"{row.gene} F-A {fmt(row.mean_fibrotic_minus_adjacent_expression)}"
            for row in data.itertuples()
        ]
    )


def region_summary(metabolite: str, gene_map: pd.DataFrame, enrichment: pd.DataFrame, paired: pd.DataFrame) -> str:
    genes = gene_map.loc[gene_map["focus_metabolite"].eq(metabolite), "gene"].unique()
    enr = enrichment[enrichment["stratum"].eq("all") & enrichment["gene"].isin(genes)].copy()
    pair = paired[paired["gene"].isin(genes)].copy()
    records = []
    if not enr.empty:
        enr["abs_enrichment"] = enr["in_region_minus_outside"].abs()
        enr = enr.sort_values("abs_enrichment", ascending=False).head(3)
        records.extend(
            [
                f"{row.region} {row.gene} region-out {fmt(row.in_region_minus_outside)}"
                for row in enr.itertuples()
            ]
        )
    if not pair.empty:
        pair["abs_difference"] = pair["mean_fibrotic_minus_adjacent_region_expression"].abs()
        pair = pair.sort_values("abs_difference", ascending=False).head(3)
        records.extend(
            [
                f"{row.region} {row.gene} paired F-A {fmt(row.mean_fibrotic_minus_adjacent_region_expression)}"
                for row in pair.itertuples()
            ]
        )
    return join_records(records, max_records=6)


def scrna_support(metabolite: str, top_features: pd.DataFrame) -> str:
    features = SCRNA_FEATURE_BY_METABOLITE.get(metabolite, [])
    records = []
    for feature in features:
        data = top_features[top_features["feature"].eq(feature)].sort_values("rank").head(4)
        for row in data.itertuples():
            records.append(f"{row.annotation2v2} ({feature}, {fmt(row.mean_value)})")
    return join_records(records, max_records=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    harreman = pd.read_csv(OUTPUT_ROOT / "harreman_all_slides" / "harreman_metabolite_difference_summary.csv")
    deconv_corr = pd.read_csv(
        OUTPUT_ROOT / "harreman_deconv_interpretation" / "slide_level_harreman_deconv_correlations.csv"
    )
    gene_map = pd.read_csv(OUTPUT_ROOT / "transporter_spatial_validation" / "focus_metabolite_transporter_gene_map.csv")
    transporter_corr = pd.read_csv(
        OUTPUT_ROOT / "transporter_spatial_validation" / "transporter_expression_harreman_correlations.csv"
    )
    transporter_diff = pd.read_csv(
        OUTPUT_ROOT / "transporter_spatial_validation" / "transporter_gene_difference_summary.csv"
    )
    region_enrichment = pd.read_csv(OUTPUT_ROOT / "spatial_region_validation" / "region_transporter_enrichment.csv")
    region_paired = pd.read_csv(
        OUTPUT_ROOT / "spatial_region_validation" / "paired_region_transporter_difference_summary.csv"
    )
    scrna_top = pd.read_csv(OUTPUT_ROOT / "scrna_metabolism_transporters" / "scrna_top_cell_types_by_feature.csv")

    rows = []
    for metabolite in FOCUS_METABOLITES:
        h = harreman[harreman["metabolite"].eq(metabolite)]
        if h.empty:
            mean_diff = np.nan
            positive_fraction = np.nan
            n_patients = 0
        else:
            hrow = h.iloc[0]
            mean_diff = float(hrow["mean_fibrotic_minus_adjacent_z"])
            positive_fraction = float(hrow["positive_patient_fraction"])
            n_patients = int(hrow["n_patients"])

        rows.append(
            {
                "metabolite": metabolite,
                "harreman_direction": direction(mean_diff),
                "mean_fibrotic_minus_adjacent_harreman_z": mean_diff,
                "harreman_positive_patient_fraction": positive_fraction,
                "harreman_n_patients": n_patients,
                "top_deconvolved_cell_type_correlations": top_deconv(metabolite, deconv_corr),
                "top_transporter_harreman_correlations": top_transporter_correlations(metabolite, transporter_corr),
                "paired_transporter_expression_direction": transporter_direction_summary(
                    metabolite,
                    gene_map,
                    transporter_diff,
                ),
                "spatial_region_evidence": region_summary(metabolite, gene_map, region_enrichment, region_paired),
                "scrna_supporting_cell_types": scrna_support(metabolite, scrna_top),
            }
        )

    evidence = pd.DataFrame(rows)
    evidence.to_csv(output_dir / "integrated_metabolite_evidence.csv", index=False)
    write_markdown_table(evidence, output_dir / "integrated_metabolite_evidence.md")

    overview = {
        "focus_metabolites": FOCUS_METABOLITES,
        "n_focus_metabolites": len(FOCUS_METABOLITES),
        "evidence_rows": evidence.to_dict(orient="records"),
    }
    with (output_dir / "integrated_evidence_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
