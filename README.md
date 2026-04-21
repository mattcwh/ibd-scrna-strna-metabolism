# Single-cell and spatial transcriptomics of stricturing Crohn's disease

## Data

Dataset includes:
- anndata.h5ad: anndata object (scanpy)
- V*.tar.gz: raw spatial transcriptomics files

Data was downloaded by running:
```
mkdir data
curl -L https://zenodo.org/api/records/14509802/files-archive -o data/download.zip
cd data/
unzip download.zip
```
## Samples

- To understand fibrosis in CD, we obtained 61 biopsy and resection samples from the colon or ileum of 21 CD and 10 non-IBD patients.
- From the latter group, we collected biopsies from screening procedures, representing healthy subjects, and resected colonic tissue obtained in the absence of chronic inflammation from patients who underwent a partial resection for diverticulitis.
- Non-IBD small intestinal tissue was obtained from biopsies, as we were unable to obtain this tissue from resections.
- We processed epithelial and non-epithelial fractions independently by stripping the epithelium with EDTA, digesting the underlying tissue enzymatically, and loading both separately.
- Samples were assigned a status based on both gross pathological examination and endoscopic score:
    - non-IBD (tissues from non-IBD subjects, used as controls)
    - non-stricture (non-inflamed tissues from CD patients with a normal pathology report)
    - inflamed (inflamed tissues without evidence of fibrosis)
    - stricture (inflamed tissues with fibrotic areas)
- These samples were used to generate single-cell transcriptomes. In the original publication, after filtering and quality control, 347,017 cells were obtained.
- To directly interrogate the interactions underpinning local cellular networks, we used Visium spatial transcriptomics to profile 20 resections from 10 patients, partially overlapping with our single-cell cohort: 10 tissue sections with stricturing (from 8 CD patients) and 10 without stricturing (from 8 patients, including 2 non-IBD controls). For 7 patients, we obtained adjacent strictured and non-strictured regions.