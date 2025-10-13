import json
import gzip
import pickle
from io import BytesIO
import requests
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_openml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import joblib
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, List, Sequence
import scipy
from sklearn.datasets import make_swiss_roll
import os, io, zipfile, tarfile, tempfile, json, warnings

def download_and_load_dataset(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)
    return data


def get_macosko():
    # --- Download Macosko data ---
    url_macosko = "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz"
    data_macosko = download_and_load_dataset(url_macosko)
    x_macosko = data_macosko["pca_50"].astype("float32")
    y_macosko = data_macosko["CellType1"].astype(str)
    y_macosko_encoded = LabelEncoder().fit_transform(y_macosko)
    return x_macosko, y_macosko_encoded, "macosko"

def get_zheng():
    # --- Download 10x mouse Zheng data ---
    url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
    data_10x = download_and_load_dataset(url_10x)
    x_10x = data_10x["pca_50"].astype("float32")
    y_10x = data_10x["CellType1"].astype("str")
    y_10x_encoded = LabelEncoder().fit_transform(y_10x)
    return x_10x, y_10x_encoded, "zheng"

def get_fashion_mnist():
    if not os.path.isfile("fashion-mnist.csv"):
        csv_data = requests.get("https://www.openml.org/data/get_csv/18238735/phpnBqZGZ")
        with open("fashion-mnist.csv", "w") as f:
            f.write(csv_data.text)
    source_df = pd.read_csv("fashion-mnist.csv")

    data = source_df.iloc[:, :784].values.astype(np.float32)
    target = source_df["class"].values
    return data, target, "fashion_mnist"

def get_mouse_msi():
    X = np.load("data/msi.npy")
    X = X.reshape(-1, X.shape[-1])
    X = X[X.max(axis=-1) > 0]
    X = X / (1e-6 + X.sum(axis=-1)[:, None])
    return X, None, "mouse_msi"

def get_mnist():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return X, y, "mnist"


def get_swiss_roll():
    X, y = make_swiss_roll(n_samples=10000, noise=0.1)
    return X, None, "swiss_roll"

def get_mammoth():
    X = np.float32(json.load(open("data/mammoth_3d.json")))
    return X, None, "mammoth"

def get_cifar():
    data = np.load("data/CIFAR10-resnet50.npy", allow_pickle=True)[()]
    X = data["X_train"]
    y = data["y_train"]
    return X, y, "cifar"

def get_cifar100():
    data = np.load("data/CIFAR100-resnet50.npy", allow_pickle=True)[()]
    X = data["X_train"]
    y = data["y_train"]
    return X, y, "cifar100"


# def get_imagenetmini():
#     data = np.load("DatasetImageNetMini1000-resnet50.npy", allow_pickle=True)[()]
#     X = data["X_train"]
#     y = data["y_train"]
#     return X, y, "imagenetmini"

# def get_20newsgroups():
#     dataset = fetch_20newsgroups(subset='all',
#                                 shuffle=True, random_state=42)
#     vectorizer = CountVectorizer(min_df=5, stop_words='english')
#     word_doc_matrix = vectorizer.fit_transform(dataset.data)
#     y = dataset.target_names
#     y = LabelEncoder().fit_transform(y)
#     return word_doc_matrix, y, "20newsgroups"







def load_paul15():
    """
    Returns (X, y, "PAUL15"): mouse myeloid progenitors (trajectory benchmark).
    """
    import scanpy as sc
    adata = sc.datasets.paul15()
    X = _X_dense(adata, use="X")
    y = _labels_from_adata(adata)  # usually 'paul15_clusters'
    return X, None, "PAUL15"

def load_pbmc3k():
    """
    Returns (X, y, "PBMC_3K") using scanpy's processed pbmc3k if available.
    """
    import scanpy as sc
    try:
        adata = sc.datasets.pbmc3k_processed()
    except Exception:
        adata = sc.datasets.pbmc3k()
        # minimal processing for labels if needed
        if "louvain" not in adata.obs:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.pca(adata, n_comps=min(50, adata.n_vars))
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.n_vars))
            sc.tl.leiden(adata, key_added="louvain")
    X = _X_dense(adata, use="X")
    y = _labels_from_adata(adata)
    return X, y, "PBMC_3K"

def load_pbmc68k(mode: str = "pca"):
    """
    Returns (X, y, "PBMC_68K") using scanpy.datasets.pbmc68k_reduced().
    mode: "pca" (50 PCs, recommended) or "X".
    """
    import scanpy as sc
    adata = sc.datasets.pbmc68k_reduced()
    X = _X_dense(adata, use="pca" if mode == "pca" else "X")
    y = _labels_from_adata(adata)
    return X, y, "PBMC_68K"


_PREFERRED_LABEL_KEYS = [
    "cell_type", "cell_type_ontology_term_id", "celltype", "celltypes",
    "major_cell_type", "free_annotation", "louvain", "leiden", "clusters"
]


def _labels_from_adata(adata):
    for k in _PREFERRED_LABEL_KEYS:
        if k in adata.obs.columns:
            return LabelEncoder().fit_transform(adata.obs[k].astype(str).to_numpy())
    return LabelEncoder().fit_transform(np.array([str(x) for x in adata.obs.index]))

def _X_dense(adata, use="X", pca_key="X_pca"):
    if use == "pca" and pca_key in adata.obsm_keys():
        return np.asarray(adata.obsm[pca_key])
    X = adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)
    return X

def _format_id_list_for_filter(ids):
    quoted = ",".join([f"'{i}'" for i in ids])
    return f"[{quoted}]"

# ---------- census dataset-id discovery ----------

def _find_dataset_ids(organism: str, title_substrings=(), collection_substrings=(), citation_substrings=()):
    import cellxgene_census as cgc
    import pandas as pd

    with cgc.open_soma() as census:
        df = census["census_info"]["datasets"].read().concat().to_pandas()

    mask = pd.Series(False, index=df.index)
    if title_substrings:
        t = df["dataset_title"].fillna("")
        for s in title_substrings:
            mask |= t.str.contains(s, case=False, regex=False)
    if collection_substrings:
        c = df["collection_name"].fillna("")
        for s in collection_substrings:
            mask |= c.str.contains(s, case=False, regex=False)
    if citation_substrings and "publication_citation" in df.columns:
        pc = df["publication_citation"].fillna("")
        for s in citation_substrings:
            mask |= pc.str.contains(s, case=False, regex=False)
    if "organism" in df.columns:
        mask &= (df["organism"] == organism)

    ids = df.loc[mask, "dataset_id"].dropna().astype(str).unique().tolist()
    return ids

# ---------- main census loader (no var_value_filter) ----------

def _load_by_dataset_ids(
    name: str,
    organism: str,
    dataset_ids: list[str],
    tissue_general: str | None = None,
    max_cells: int | None = None,
    use_pca: bool = True,
    pca_n: int = 50,
):
    import scanpy as sc
    import cellxgene_census as cgc

    id_list = _format_id_list_for_filter(dataset_ids)
    clauses = [f"dataset_id in {id_list}", "is_primary_data == True"]
    if tissue_general is not None:
        clauses.append(f"tissue_general == '{tissue_general}'")
    obs_value_filter = " and ".join(clauses)

    with cgc.open_soma() as census:
        adata = cgc.get_anndata(
            census=census,
            organism=organism,
            obs_value_filter=obs_value_filter,
            # NOTE: no var_value_filter here; schema does not expose feature_is_filtered
            X_name="raw",  # "normalized" exists only in newer schema; "raw" is always present
        )

    # Optional subsample for speed
    if max_cells is not None and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0)

    # Optional PCA (typical 50 PCs for DR benchmarks)
    if use_pca:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars))
        adata = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.pca(adata, n_comps=min(pca_n, adata.n_vars))

    X = _X_dense(adata, use="pca" if use_pca else "X")
    y = _labels_from_adata(adata)
    y = LabelEncoder().fit_transform(y)
    return X, y, name
# --------- requested pancreas loaders (no 'contains' in obs_value_filter) ---------

def load_pancreas_muraro(max_cells: int | None = 10000, use_pca: bool = True, pca_n: int = 50):
    """
    Muraro et al. (human pancreas). Finds dataset_id(s) via datasets metadata,
    then filters obs with 'dataset_id in [...] and tissue_general == "pancreas"'.
    """
    ids = _find_dataset_ids(
        organism="Homo sapiens",
        title_substrings=("Muraro",),
        collection_substrings=("pancreas",),  # lenient extra hook
        citation_substrings=("Muraro",),
    )
    if not ids:
        # Fallback: broad pancreas filter (no study restriction)
        import scanpy as sc, cellxgene_census as cgc
        with cgc.open_soma() as census:
            adata = cgc.get_anndata(
                census=census,
                organism="Homo sapiens",
                obs_value_filter="tissue_general == 'pancreas' and is_primary_data == True",
                var_value_filter="feature_is_filtered == False",
                X_name="raw",
            )
        if use_pca:
            sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars))
            adata = adata[:, adata.var["highly_variable"]].copy()
            sc.pp.pca(adata, n_comps=min(pca_n, adata.n_vars))
        X = _X_dense(adata, use="pca" if use_pca else "X")
        y = _labels_from_adata(adata)
        return X, y, "PANCREAS_MURARO_FALLBACK"
    return _load_by_dataset_ids(
        name="PANCREAS_MURARO",
        organism="Homo sapiens",
        dataset_ids=ids,
        tissue_general="pancreas",
        max_cells=max_cells,
        use_pca=use_pca,
        pca_n=pca_n,
    )

def load_pancreas_baron(max_cells: int | None = 20000, use_pca: bool = True, pca_n: int = 50):
    ids = _find_dataset_ids(
        organism="Homo sapiens",
        title_substrings=("Baron",),
        collection_substrings=("pancreas",),
        citation_substrings=("Baron",),
    )
    if not ids:
        raise RuntimeError("No Baron pancreas datasets found in this Census version.")
    return _load_by_dataset_ids(
        name="PANCREAS_BARON",
        organism="Homo sapiens",
        dataset_ids=ids,
        tissue_general="pancreas",
        max_cells=max_cells,
        use_pca=use_pca,
        pca_n=pca_n,
    )

def load_pancreas_segerstolpe(max_cells: int | None = 10000, use_pca: bool = True, pca_n: int = 50):
    ids = _find_dataset_ids(
        organism="Homo sapiens",
        title_substrings=("Segerstolpe",),
        collection_substrings=("pancreas",),
        citation_substrings=("Segerstolpe",),
    )
    if not ids:
        raise RuntimeError("No Segerstolpe pancreas datasets found in this Census version.")
    return _load_by_dataset_ids(
        name="PANCREAS_SEGERSTOLPE",
        organism="Homo sapiens",
        dataset_ids=ids,
        tissue_general="pancreas",
        max_cells=max_cells,
        use_pca=use_pca,
        pca_n=pca_n,
    )

def _prep_X(adata, use_pca=True, pca_n=50):
    import scanpy as sc
    if use_pca:
        if "X_pca" not in adata.obsm_keys():
            # light, standard pipeline
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), flavor="seurat_v3")
            ad = adata[:, adata.var["highly_variable"]].copy()
            sc.pp.pca(ad, n_comps=min(pca_n, ad.n_vars))
            return np.asarray(ad.obsm["X_pca"])
        else:
            return np.asarray(adata.obsm["X_pca"])
    try:
        return np.float32(np.asarray(adata.X.todense()))
    except:
        return np.float32(np.asarray(adata.X))

def _maybe_subsample(adata, max_cells):
    import scanpy as sc
    if max_cells is not None and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0)

def _labels_from_adata_or_cluster(adata, force_cluster=False):
    import scanpy as sc
    # pick an existing label if available
    if not force_cluster:
        for k in _PREFERRED_LABEL_KEYS:
            if k in adata.obs.columns:
                return adata.obs[k].astype(str).to_numpy(), k

    # otherwise create quick graph clusters to serve as labels
    if "X_pca" not in adata.obsm_keys():
        # minimal preprocessing
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=min(50, adata.n_vars))
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
    sc.tl.leiden(adata, key_added="leiden")
    return adata.obs["leiden"].astype(str).to_numpy(), "leiden"


def load_visium(sample_id="V1_Human_Lymph_Node", use_pca=False, pca_n=50, max_cells=None, force_cluster=False):
    """
    Visium Spatial (scanpy built-in downloader via 10x SGE).
    Common sample_ids: 'V1_Human_Lymph_Node', 'V1_Adult_Mouse_Brain',
                       'V1_Breast_Cancer_Block_A_Section_1'
    """
    import scanpy as sc
    ad = sc.datasets.visium_sge(sample_id=sample_id)
    _maybe_subsample(ad, max_cells)
    X = _prep_X(ad, use_pca=use_pca, pca_n=pca_n)
    y, ykey = _labels_from_adata_or_cluster(ad, force_cluster=force_cluster)
    return X, y, f"VISIUM_{sample_id}"

def load_krumsiek11(use_pca=False, pca_n=11, max_cells=None):
    """Krumsiek11 synthetic gene-regulatory network (biological toy)."""
    import scanpy as sc
    ad = sc.datasets.krumsiek11()
    _maybe_subsample(ad, max_cells)
    X = _prep_X(ad, use_pca=use_pca, pca_n=pca_n)
    y, ykey = _labels_from_adata_or_cluster(ad)
    return X, y, "KRUMSIEK11"

# ------------------------- scvelo datasets (optional: pip install scvelo) -------------------------

def load_scvelo_pancreas(use_pca=False, pca_n=50, max_cells=None):
    """Pancreas (scvelo example dataset)."""
    try:
        import scvelo as scv
    except ImportError as e:
        raise ImportError("Please install scvelo: pip install scvelo") from e
    ad = scv.datasets.pancreas()
    _maybe_subsample(ad, max_cells)
    X = _prep_X(ad, use_pca=use_pca, pca_n=pca_n)
    y, ykey = _labels_from_adata_or_cluster(ad)
    return X, y, "PANCREAS_SCV"

def load_scvelo_dentategyrus(use_pca=False, pca_n=50, max_cells=None):
    """Dentate gyrus neurogenesis (trajectory)."""
    try:
        import scvelo as scv
    except ImportError as e:
        raise ImportError("Please install scvelo: pip install scvelo") from e
    ad = scv.datasets.dentategyrus()
    _maybe_subsample(ad, max_cells)
    X = _prep_X(ad, use_pca=use_pca, pca_n=pca_n)
    y, ykey = _labels_from_adata_or_cluster(ad)
    return X, y, "DENTATE_GYRUS_SCV"

# ------------------------- squidpy spatial datasets (optional: pip install squidpy) -------------------------

def load_squidpy_visium_hne(use_pca=False, pca_n=50, max_cells=None):
    """Squidpy demo H&E Visium dataset (human lymph node)."""
    try:
        import squidpy as sq
    except ImportError as e:
        raise ImportError("Please install squidpy: pip install squidpy") from e
    ad = sq.datasets.visium_hne_adata()
    _maybe_subsample(ad, max_cells)
    X = _prep_X(ad, use_pca=use_pca, pca_n=pca_n)
    y, ykey = _labels_from_adata_or_cluster(ad)
    return X, y, "VISIUM_HNE_SQ"


from itertools import combinations
def make_chain(n_per=200, D=100, sigma=100, f=lambda k: k, seed=0):
    rng = np.random.default_rng(seed)
    # centers with cumulative spacing = f
    gaps = np.array([f(i) - f(i-1) for i in range(1,10)])
    u = rng.normal(size=(D,)); u /= np.linalg.norm(u)
    centers = [np.zeros(D)]
    for g in gaps:
        centers.append(centers[-1] + g * u)
    centers = np.stack(centers)
    # random rotation
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    centers = centers @ Q
    X, y = [], []
    for i, c in enumerate(centers):
        Xi = c + sigma * rng.normal(size=(n_per, D))
        X.append(Xi); y += [i]*n_per
    return np.vstack(X), np.array(y), "CHAIN"







def openml_open(key, version="active", zscore=True):
    """
    key: str dataset name (uses version='active' by default) OR int data_id
    returns: X (float32 array), y (str array), name (str)
    """
    if isinstance(key, int):
        ds = fetch_openml(data_id=key, as_frame=True)  # no version arg for IDs
        name = f"OpenML_{key}"
    else:
        ds = fetch_openml(name=key, version=version, as_frame=True)  # version='active'
        name = f"OpenML_{ds.details['name']}"

    # Data & target as pandas objects
    X_df = ds.data.copy()
    y_ser = ds.target.copy()

    # Drop unlabeled rows if any
    mask = ~pd.isna(y_ser)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    X_df = X_df.loc[mask]
    y_ser = y_ser.loc[mask].astype(str)

    # One-hot encode any non-numeric columns (e.g., 'splice')
    non_num = X_df.select_dtypes(exclude=[np.number]).columns
    if len(non_num) > 0:
        X_df = pd.get_dummies(X_df, columns=list(non_num), dummy_na=False)

    # Clean & to numpy
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X_df.to_numpy(dtype=np.float32)
    y = y_ser.to_numpy()

    if zscore and X.size > 0:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        X = ((X - mu) / sd).astype(np.float32)

    return X, y, name

def load_mice_protein():
    """MiceProtein dataset - proteomics data."""
    X, y, name = openml_open("MiceProtein")
    return X, y, "MICE_PROTEIN"

def load_arcene():
    """Arcene dataset - mass spectrometry proteomics data."""
    X, y, name = openml_open("arcene") 
    return X, y, "ARCENE"

def load_breast_cancer():
    """Breast Cancer Wisconsin (Diagnostic) dataset."""
    X, y, name = openml_open(1510)
    return X, y, "BREAST_CANCER"

# def load_blood_transfusion():
#     """Blood Transfusion Service Center dataset."""
#     X, y, name = openml_open(1464)
#     return X, y, "BLOOD_TRANSFUSION" 

def load_qsar_biodeg():
    """QSAR biodegradation dataset."""
    X, y, name = openml_open("qsar-biodeg")
    return X, y, "QSAR_BIODEG"

def load_parkinsons():
    """Parkinson's disease voice recording dataset."""
    X, y, name = openml_open("parkinsons")
    return X, y, "PARKINSONS"


def load_birthweight():
    """Low birthweight dataset."""
    X, y, name = openml_open(203)
    return X, y, "BIRTHWEIGHT"












# Get all functions that start with get_ or load_
DATASET_FUNCS = {name[4:]: func for name, func in globals().items() 
                if name.startswith('get_') and callable(func)}
DATASET_FUNCS.update({name[5:]: func for name, func in globals().items() 
                    if name.startswith('load_') and callable(func)})
#DATASET_FUNCS["chain"] = make_chain

def get_available_datasets() -> List[str]:
    """Returns list of available dataset names."""
    return sorted(list(DATASET_FUNCS.keys()))

class DatasetManager:
    """Manages dataset loading with caching and sampling capabilities."""
    
    def __init__(self, cache_dir: str = ".cache/datasets"):
        """
        Initialize DatasetManager.
        
        Args:
            cache_dir: Directory to store cached datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, dataset_name: str) -> Path:
        """Get path for cached dataset file."""
        return self.cache_dir / f"{dataset_name}.joblib"
        
    def load_dataset(self, dataset_name: str, 
                    sample_ratio: Optional[float] = None,
                    random_state: int = 42) -> Tuple:
        """
        Load a dataset by name, with optional sampling.
        
        Args:
            dataset_name: Name of dataset to load
            sample_ratio: If not None, fraction of data to randomly sample
            random_state: Random seed for sampling
            
        Returns:
            Tuple of (X, y, name) as returned by dataset loading functions
        """
        if dataset_name.lower() not in DATASET_FUNCS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        cache_path = self._get_cache_path(dataset_name.lower())
        
        # Try loading from cache first
        if cache_path.exists():
            X, y, name = joblib.load(cache_path)
        else:
            # Load dataset and cache it
            X, y, name = DATASET_FUNCS[dataset_name.lower()]()
            joblib.dump((X, y, name), cache_path)
        
        # Sample if requested
        if sample_ratio is not None:
            if not 0 < sample_ratio <= 1:
                raise ValueError("sample_ratio must be between 0 and 1")
                
            rng = np.random.RandomState(random_state)
            n_samples = int(X.shape[0] * sample_ratio)
            indices = rng.choice(X.shape[0], size=n_samples, replace=False)
            
            X = X[indices]
            if y is not None:
                y = y[indices]
                
        if sample_ratio is not None:
            name = name + f"_sampled_{sample_ratio}_{random_state}"
        
        if name == "swiss_roll":
            y = np.int32(y)
        
        if name == "zheng":
            X = X[::20, :]
            y = y[::20]
        
        if isinstance(X, scipy.sparse.csr_matrix):
            X = X.toarray()
        
        return np.float32(X), y, name


