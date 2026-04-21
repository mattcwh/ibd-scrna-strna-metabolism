"""Generate per-slide spatial QC plots and Visium image overlays."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "spatial_qc"


def normalize_slide_id(value: str) -> str:
    return value.replace("-", "_").removesuffix(".tar.gz")


def metadata_slide_from_archive_stem(stem: str) -> str:
    if stem.startswith("V10A14_143"):
        return stem.replace("V10A14_143", "V10A14-143")
    return stem


def find_archive_for_sample(sample: str) -> Path:
    candidates = [sample, normalize_slide_id(sample)]
    matches = []
    for candidate in dict.fromkeys(candidates):
        matches.extend(sorted(DATA_DIR.glob(f"{candidate}.tar.gz")))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one archive for {sample}, found {len(matches)}")
    return matches[0]


def archive_prefix(archive: Path) -> str:
    return archive.name.removesuffix(".tar.gz")


def read_positions(archive: Path) -> pd.DataFrame:
    prefix = archive_prefix(archive)
    member = f"{prefix}.tissue_positions_list.csv"
    with tarfile.open(archive, "r:gz") as tf:
        handle = tf.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member)
        return pd.read_csv(
            handle,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )


def read_scalefactors(archive: Path) -> dict[str, float]:
    prefix = archive_prefix(archive)
    member = f"{prefix}.scalefactors_json.json"
    with tarfile.open(archive, "r:gz") as tf:
        handle = tf.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member)
        return json.loads(handle.read().decode("utf-8"))


def read_image(archive: Path, image_kind: str = "lowres") -> Image.Image:
    prefix = archive_prefix(archive)
    member = f"{prefix}.tissue_{image_kind}_image.png"
    with tarfile.open(archive, "r:gz") as tf:
        handle = tf.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member)
        return Image.open(io.BytesIO(handle.read())).convert("RGB")


def load_spatial_coordinates() -> pd.DataFrame:
    obj = ad.read_h5ad(DATA_DIR / "anndata.h5ad", backed="r")
    try:
        obs = obj.obs.copy()
        coords = pd.DataFrame(
            obj.obsm["spatial"],
            index=obj.obs_names,
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
    finally:
        obj.file.close()

    coords["sample"] = obs["sample"].astype(str).to_numpy()
    coords["barcode"] = [
        index.removeprefix(f"{sample}_") for index, sample in zip(coords.index, coords["sample"])
    ]
    return coords


def plot_coordinate_scatter(slide: pd.DataFrame, metadata: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(
        slide["pxl_col_in_fullres"],
        slide["pxl_row_in_fullres"],
        s=3,
        linewidths=0,
        alpha=0.85,
    )
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    title = (
        f"{metadata['Visium Slide ID']} | {metadata['Disease Label']} | "
        f"patient {metadata['Patient_ID']}"
    )
    plt.title(title)
    plt.xlabel("full-resolution pixel column")
    plt.ylabel("full-resolution pixel row")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_image_overlay(
    slide: pd.DataFrame,
    image: Image.Image,
    scalefactors: dict[str, float],
    metadata: pd.Series,
    output_path: Path,
) -> None:
    scale = scalefactors["tissue_lowres_scalef"]
    spot_diameter = scalefactors["spot_diameter_fullres"] * scale
    x = slide["pxl_col_in_fullres"] * scale
    y = slide["pxl_row_in_fullres"] * scale

    plt.figure(figsize=(7, 7))
    plt.imshow(image)
    plt.scatter(
        x,
        y,
        s=max(1.0, spot_diameter * 0.14),
        c="#1E7A2F",
        alpha=0.45,
        linewidths=0,
    )
    plt.axis("off")
    title = f"{metadata['Visium Slide ID']} spot overlay"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_all_slides(coords: pd.DataFrame, metadata: pd.DataFrame, output_path: Path) -> None:
    merged = coords.merge(
        metadata[["sample", "Disease Label", "Patient_ID"]],
        on="sample",
        how="left",
    )
    samples = sorted(merged["sample"].unique())
    n_cols = 5
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    axes = axes.flatten()
    palette = {
        "Fibrotic": "#C44E52",
        "Adjacent": "#4C72B0",
        "Adjacent (Disease Control)": "#55A868",
    }
    for ax, sample in zip(axes, samples):
        slide = merged[merged["sample"].eq(sample)]
        label = str(slide["Disease Label"].iloc[0])
        ax.scatter(
            slide["pxl_col_in_fullres"],
            slide["pxl_row_in_fullres"],
            s=1.2,
            linewidths=0,
            alpha=0.85,
            color=palette.get(label, "#4C4C4C"),
        )
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{sample}\n{label}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(samples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def summarize_slide(
    sample: str,
    slide: pd.DataFrame,
    archive_positions: pd.DataFrame,
    metadata: pd.Series,
    scalefactors: dict[str, float],
    image: Image.Image,
) -> dict[str, object]:
    position_barcodes = set(archive_positions["barcode"])
    observed_barcodes = set(slide["barcode"])
    in_tissue_positions = archive_positions[archive_positions["in_tissue"].eq(1)]

    return {
        "sample": sample,
        "patient_id": metadata["Patient_ID"],
        "disease_label": metadata["Disease Label"],
        "general_categorization": metadata["General Categorization"],
        "n_spots_anndata": int(len(slide)),
        "n_positions_in_archive": int(len(archive_positions)),
        "n_in_tissue_positions_archive": int(len(in_tissue_positions)),
        "n_off_tissue_positions_archive": int(len(archive_positions) - len(in_tissue_positions)),
        "n_barcode_overlap": int(len(observed_barcodes & position_barcodes)),
        "n_anndata_barcodes_missing_from_archive": int(len(observed_barcodes - position_barcodes)),
        "n_archive_barcodes_missing_from_anndata": int(len(position_barcodes - observed_barcodes)),
        "min_fullres_row": float(slide["pxl_row_in_fullres"].min()),
        "max_fullres_row": float(slide["pxl_row_in_fullres"].max()),
        "min_fullres_col": float(slide["pxl_col_in_fullres"].min()),
        "max_fullres_col": float(slide["pxl_col_in_fullres"].max()),
        "lowres_width": int(image.width),
        "lowres_height": int(image.height),
        "tissue_lowres_scalef": float(scalefactors["tissue_lowres_scalef"]),
        "spot_diameter_fullres": float(scalefactors["spot_diameter_fullres"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")
    metadata = pd.read_csv(DATA_DIR / "spatial_transcriptome_slide_metadata.csv")
    metadata["sample"] = metadata["Visium Slide ID"].astype(str)
    coords = load_spatial_coordinates()

    plot_all_slides(coords, metadata, output_dir / "all_slides_coordinates.png")

    rows = []
    for sample in sorted(coords["sample"].unique()):
        slide = coords[coords["sample"].eq(sample)].copy()
        meta = metadata.loc[metadata["sample"].eq(sample)]
        if len(meta) != 1:
            raise ValueError(f"Expected one metadata row for {sample}, found {len(meta)}")
        metadata_row = meta.iloc[0]

        archive = find_archive_for_sample(sample)
        positions = read_positions(archive)
        scalefactors = read_scalefactors(archive)
        image = read_image(archive, "lowres")

        rows.append(summarize_slide(sample, slide, positions, metadata_row, scalefactors, image))
        plot_coordinate_scatter(
            slide,
            metadata_row,
            plots_dir / f"{normalize_slide_id(sample)}_coordinates.png",
        )
        plot_image_overlay(
            slide,
            image,
            scalefactors,
            metadata_row,
            overlays_dir / f"{normalize_slide_id(sample)}_lowres_overlay.png",
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "spatial_qc_summary.csv", index=False)

    overview = {
        "n_slides": int(summary["sample"].nunique()),
        "total_spots": int(summary["n_spots_anndata"].sum()),
        "total_anndata_barcodes_missing_from_archive": int(
            summary["n_anndata_barcodes_missing_from_archive"].sum()
        ),
        "total_archive_barcodes_missing_from_anndata": int(
            summary["n_archive_barcodes_missing_from_anndata"].sum()
        ),
        "slides_by_disease_label": summary["disease_label"].value_counts().to_dict(),
    }
    with (output_dir / "spatial_qc_overview.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
