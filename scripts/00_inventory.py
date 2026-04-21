"""Inventory data files and AnnData metadata for the IBD scRNA/spatial project."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import anndata as ad
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "inventory"


def normalize_slide_id(value: str) -> str:
    """Normalize slide IDs across metadata and archive filenames."""
    return value.replace("-", "_").removesuffix(".tar.gz")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def list_data_files() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "file": path.name,
                    "suffix": "".join(path.suffixes),
                    "size_mb": round(file_size_mb(path), 3),
                }
            )
    return pd.DataFrame(rows)


def inspect_tar_archives() -> tuple[pd.DataFrame, pd.DataFrame]:
    archive_rows = []
    member_rows = []
    for archive in sorted(DATA_DIR.glob("V*.tar.gz")):
        with tarfile.open(archive, "r:gz") as tf:
            members = [member for member in tf.getmembers() if member.isfile()]

        names = [Path(member.name).name for member in members]
        archive_rows.append(
            {
                "archive": archive.name,
                "archive_slide_id": archive.name.removesuffix(".tar.gz"),
                "normalized_slide_id": normalize_slide_id(archive.name),
                "n_files": len(names),
                "has_matrix": any(name.endswith(".matrix.mtx") for name in names),
                "has_barcodes": any(name.endswith(".barcodes.csv") for name in names),
                "has_genes": any(name.endswith(".genes.csv") for name in names),
                "has_positions": any(
                    name.endswith(".tissue_positions_list.csv") for name in names
                ),
                "has_scalefactors": any(
                    name.endswith(".scalefactors_json.json") for name in names
                ),
                "has_hires_image": any(name.endswith(".tissue_hires_image.png") for name in names),
                "has_lowres_image": any(name.endswith(".tissue_lowres_image.png") for name in names),
            }
        )
        for name in names:
            member_rows.append({"archive": archive.name, "member": name})

    return pd.DataFrame(archive_rows), pd.DataFrame(member_rows)


def inspect_h5ad(path: Path) -> dict[str, object]:
    obj = ad.read_h5ad(path, backed="r")
    try:
        obs = obj.obs
        var = obj.var
        summary: dict[str, object] = {
            "file": path.name,
            "n_obs": int(obj.n_obs),
            "n_vars": int(obj.n_vars),
            "obs_columns": list(obs.columns),
            "var_columns": list(var.columns),
            "obsm_keys": list(obj.obsm.keys()),
            "layers": list(obj.layers.keys()),
            "uns_keys": list(obj.uns.keys()),
            "categorical_obs": {},
        }
        categorical_obs: dict[str, list[str]] = {}
        for column in obs.columns:
            if isinstance(obs[column].dtype, pd.CategoricalDtype):
                values = [str(item) for item in obs[column].cat.categories]
                categorical_obs[column] = values
        summary["categorical_obs"] = categorical_obs
        return summary
    finally:
        obj.file.close()


def write_h5ad_column_tables(summaries: list[dict[str, object]], output_dir: Path) -> None:
    obs_rows = []
    var_rows = []
    obsm_rows = []
    layer_rows = []
    category_rows = []

    for summary in summaries:
        file_name = str(summary["file"])
        for column in summary["obs_columns"]:
            obs_rows.append({"file": file_name, "obs_column": column})
        for column in summary["var_columns"]:
            var_rows.append({"file": file_name, "var_column": column})
        for key in summary["obsm_keys"]:
            obsm_rows.append({"file": file_name, "obsm_key": key})
        for layer in summary["layers"]:
            layer_rows.append({"file": file_name, "layer": layer})
        for column, values in dict(summary["categorical_obs"]).items():
            for value in values:
                category_rows.append({"file": file_name, "obs_column": column, "category": value})

    pd.DataFrame(obs_rows).to_csv(output_dir / "h5ad_obs_columns.csv", index=False)
    pd.DataFrame(var_rows).to_csv(output_dir / "h5ad_var_columns.csv", index=False)
    pd.DataFrame(obsm_rows).to_csv(output_dir / "h5ad_obsm_keys.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(output_dir / "h5ad_layers.csv", index=False)
    pd.DataFrame(category_rows).to_csv(output_dir / "h5ad_obs_categories.csv", index=False)


def inspect_spatial_metadata() -> pd.DataFrame:
    path = DATA_DIR / "spatial_transcriptome_slide_metadata.csv"
    df = pd.read_csv(path)
    df["metadata_slide_id"] = df["Visium Slide ID"]
    df["normalized_slide_id"] = df["metadata_slide_id"].map(normalize_slide_id)
    return df


def inspect_sample_metadata() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_metadata.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = list_data_files()
    archive_summary, archive_members = inspect_tar_archives()
    spatial_metadata = inspect_spatial_metadata()
    sample_metadata = inspect_sample_metadata()

    archive_metadata_join = spatial_metadata.merge(
        archive_summary[["archive", "normalized_slide_id"]],
        on="normalized_slide_id",
        how="outer",
        indicator=True,
    )

    h5ad_summaries = [
        inspect_h5ad(DATA_DIR / "anndata.h5ad"),
        inspect_h5ad(DATA_DIR / "Cleaned_raw_annotated_object_LK.v2.h5ad"),
    ]

    data_files.to_csv(output_dir / "data_files.csv", index=False)
    archive_summary.to_csv(output_dir / "spatial_archive_summary.csv", index=False)
    archive_members.to_csv(output_dir / "spatial_archive_members.csv", index=False)
    spatial_metadata.to_csv(output_dir / "spatial_metadata_normalized.csv", index=False)
    sample_metadata.to_csv(output_dir / "sample_metadata.csv", index=False)
    archive_metadata_join.to_csv(output_dir / "spatial_archive_metadata_join.csv", index=False)
    write_h5ad_column_tables(h5ad_summaries, output_dir)

    with (output_dir / "h5ad_summary.json").open("w") as handle:
        json.dump(h5ad_summaries, handle, indent=2)

    overview = {
        "n_data_files": int(len(data_files)),
        "n_spatial_archives": int(len(archive_summary)),
        "n_spatial_metadata_rows": int(len(spatial_metadata)),
        "n_sample_metadata_rows": int(len(sample_metadata)),
        "spatial_archive_metadata_join_counts": archive_metadata_join["_merge"]
        .value_counts()
        .to_dict(),
        "h5ad_shapes": {
            item["file"]: {"n_obs": item["n_obs"], "n_vars": item["n_vars"]}
            for item in h5ad_summaries
        },
    }
    with (output_dir / "inventory_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
