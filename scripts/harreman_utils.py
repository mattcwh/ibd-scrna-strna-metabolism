"""Shared helpers for Harreman transporter communication analyses."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import anndata as ad
import harreman as hm
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def patch_harreman_readonly_diagonal() -> None:
    """Work around a Harreman/Pandas compatibility issue in lc_zs construction."""
    original_fill_diagonal = np.fill_diagonal

    def safe_fill_diagonal(a, val, wrap=False):
        try:
            return original_fill_diagonal(a, val, wrap=wrap)
        except ValueError as error:
            if "read-only" in str(error):
                return None
            raise

    np.fill_diagonal = safe_fill_diagonal


def load_spatial_subset(samples: list[str]) -> ad.AnnData:
    adata = ad.read_h5ad(DATA_DIR / "anndata.h5ad")
    adata = adata[adata.obs["sample"].isin(samples)].copy()
    adata.obs_names = adata.obs_names.astype(str).astype(object)
    adata.var_names = adata.var_names.astype(str).astype(object)
    adata.obs["sample"] = adata.obs["sample"].astype(str)
    return adata


def run_harreman_on_adata(
    adata: ad.AnnData,
    n_neighbors: int,
    expression_threshold: float,
) -> ad.AnnData:
    hm.pp.extract_interaction_db(
        adata,
        species="human",
        database="transporter",
        extracellular_only=True,
        verbose=False,
    )
    hm.tl.compute_knn_graph(
        adata,
        compute_neighbors_on_key="spatial",
        n_neighbors=n_neighbors,
        sample_key="sample",
        weighted_graph=True,
        verbose=False,
    )
    hm.tl.apply_gene_filtering(
        adata,
        layer_key=None,
        model="none",
        feature_elimination=True,
        threshold=expression_threshold,
        autocorrelation_filt=False,
        expression_filt=False,
        de_filt=False,
        verbose=False,
    )
    hm.tl.compute_gene_pairs(
        adata,
        layer_key=None,
        ct_specific=False,
        verbose=False,
    )
    hm.tl.compute_cell_communication(
        adata,
        layer_key_p_test=None,
        model="none",
        test="parametric",
        device="cpu",
        verbose=False,
    )
    return adata


def metabolite_results(adata: ad.AnnData, sample: str) -> pd.DataFrame:
    result = adata.uns["ccc_results"]["p"]["m"]
    metabolites = list(adata.uns["gene_pair_dict"].keys())
    df = pd.DataFrame(
        {
            "sample": sample,
            "metabolite": metabolites,
            "communication_score": result["cs"],
            "z_score": result["Z"],
            "p_value": result["Z_pval"],
            "fdr": result["Z_FDR"],
        }
    )
    df["n_gene_pairs"] = df["metabolite"].map(
        {metabolite: len(indices) for metabolite, indices in adata.uns["gene_pair_dict"].items()}
    )
    return df.sort_values("z_score", ascending=False)


def gene_pair_results(adata: ad.AnnData, sample: str) -> pd.DataFrame:
    result = adata.uns["ccc_results"]["p"]["gp"]
    gene_pairs = [
        " - ".join(
            [
                " + ".join(part) if isinstance(part, list) else str(part)
                for part in pair
            ]
        )
        for pair in adata.uns["gene_pairs"]
    ]
    return pd.DataFrame(
        {
            "sample": sample,
            "gene_pair": gene_pairs,
            "communication_score": result["cs"],
            "z_score": result["Z"],
            "p_value": result["Z_pval"],
            "fdr": result["Z_FDR"],
        }
    ).sort_values("z_score", ascending=False)


def summarize_slide(adata: ad.AnnData, sample: str, runtime_seconds: float) -> dict[str, object]:
    database = adata.varm["database"]
    return {
        "sample": sample,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_transport_metabolites_after_database": int((database != 0).any(axis=0).sum()),
        "n_transporter_genes_after_filtering": int((database != 0).any(axis=1).sum()),
        "n_gene_pairs": int(len(adata.uns["gene_pairs"])),
        "n_metabolites_tested": int(len(adata.uns["gene_pair_dict"])),
        "n_spatial_weight_edges": int(adata.obsp["weights"].nnz),
        "runtime_seconds": round(runtime_seconds, 3),
    }


def run_harreman_for_sample(
    sample: str,
    n_neighbors: int,
    expression_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    start = time.time()
    adata = load_spatial_subset([sample])
    adata = run_harreman_on_adata(
        adata,
        n_neighbors=n_neighbors,
        expression_threshold=expression_threshold,
    )
    runtime = time.time() - start
    metabolites = metabolite_results(adata, sample)
    gene_pairs = gene_pair_results(adata, sample)
    summary = summarize_slide(adata, sample, runtime)
    return metabolites, gene_pairs, summary
