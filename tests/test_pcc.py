import pytest
import numpy as np
import torch
from sklearn.datasets import make_blobs
from pcc import PCC, PCUMAP

@pytest.fixture
def synthetic_data():
    X, clusters = make_blobs(n_samples=100, n_features=50, centers=5, random_state=42)
    return X, [clusters] # PCC expects list of cluster assignments

@pytest.fixture 
def blob_data():
    X, _ = make_blobs(n_samples=100, n_features=50, centers=5, random_state=42)
    return X

def test_pcc_basic(synthetic_data):
    X, clusters = synthetic_data
    pcc = PCC(n_components=2, num_epochs=10)
    embedding = pcc.fit_transform(X, clusters)
    assert embedding.shape == (100, 2)

def test_pcc_parameters(synthetic_data):
    X, clusters = synthetic_data
    pcc = PCC(n_components=3,
              num_epochs=3, 
              num_points=10,
              pearson=True,
              spearman=False,
              beta=10)
    embedding = pcc.fit_transform(X, clusters)
    assert embedding.shape == (100, 3)

def test_pcc_no_clusters(synthetic_data):
    X, _ = synthetic_data
    pcc = PCC(n_components=2,
              num_epochs=10,
              cluster=False)
    embedding = pcc.fit_transform(X, None)
    assert embedding.shape == (100, 2)

def test_pcumap_basic(blob_data):
    # backend=None: torchdr >= 0.4 defaults to the faiss backend, which is not a dependency
    pcumap = PCUMAP(n_components=2, backend=None)
    embedding = pcumap.fit_transform(blob_data)
    assert embedding.shape == (100, 2)


def test_reference_point_sampling(blob_data):
    pcumap = PCUMAP(num_points=50, device='cpu')
    indices = pcumap.get_reference_points(blob_data, 50)
    assert len(indices) == 50
    assert all(i < len(blob_data) for i in indices)


def test_pcumap_applies_correlation_loss(blob_data, monkeypatch):
    # torchdr >= 0.3 computes UMAP gradients in closed form and never calls
    # _loss, so the correlation term has to be hooked in elsewhere.
    calls = []
    original = PCUMAP.correlation_loss

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PCUMAP, "correlation_loss", spy)
    pcumap = PCUMAP(n_components=2, num_points=50, max_iter=30,
                    epoch_to_start_correlation_loss=10, device="cpu", backend=None)
    embedding = pcumap.fit_transform(blob_data)
    assert embedding.shape == (100, 2)
    assert len(calls) > 0
