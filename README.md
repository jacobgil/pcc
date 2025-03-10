# PCC - Dimensionality reduction with very high global structure preservation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Build Status](https://github.com/jacobgil/pcc/workflows/Tests/badge.svg)

![PCC](logo.png)


`pip install pccdr`


⭐ This is a python package for dimensionality reduction (DR) with high global structure preservation.

⭐ That means that unlike in popular DR methods like UMAP, the distances between transformed points - will actually mean something.

⭐ Use PCUMAP for simply enhancing the widely used UMAP method with global structure preservation

⭐ Or use it with our own PCC objective that resutls with extremely high global structure preservation, and competitive local structure


*(For spearman correlation support, install [torchsort](https://github.com/teddykoker/torchsort) (`pip install torchsort`))*

PCC is built on the idea of sampling reference points, meausring distances of all data points from the reference points, and maximizing the correlations of these distances in the high dimensional data, and the transformed low dimensional data.


# Usage examples
See examples/macosko.ipynb for more detailed explanation and usage examples.


There are two modes:

## Plugging into UMAP, for getting a meaningful transformation where distances between points mean something

Here we use the excellent recent [TorchDR](https://github.com/TorchDR/TorchDR) library, and add plug in our objective into UMAP.

```python
from pcc import PCUMAP
pcumap_embedding = PCUMAP(device='cuda').fit_transform(X)
```


## PCC as a standalone DR method with a multi task objective

This optimizes a local structure preservation multi task objective that tries to predict which clusters points belong to,
as well as global structure preservation loss that maximizes corerlations between distances of all points to sampled reference points.

First, lets cluster the points with different clustering models:

```python

np.random.seed(0)

clusters = []
n_clusters_list = [4, 8, 16, 32, 64]
for n_clusters in n_clusters_list:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    cluster_labels = kmeans.fit_predict(X)
    clusters.append(cluster_labels)
```


Then we can call PCC:

```python
pcc_reducer = PCC(num_components=2, num_epochs=2000, num_points=1000, pearson=True, 
                  spearman=False, beta=5, k_epoch=2)
pcc_embedding = pcc_reducer.fit_transform(X, clusters)
```
