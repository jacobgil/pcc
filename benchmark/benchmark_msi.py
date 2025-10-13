import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from zadu import zadu
import torch
from pcc import PCC, PCUMAP
from pcc.metrics import Metrics
import pandas as pd
from msi_atlas.annotations import get_annotations
from pcc.metrics import Metrics
from sklearn.cluster import KMeans
import torch
import gc
from torchdr import UMAP
from msi_visual.utils import normalize
import tqdm
from importlib import reload
from PIL import Image
from pcc import PCC
import cv2
import torch
import os

class MSIDRWrapper:
    def __init__(self, dr_method):
        self.dr_method = dr_method
    
    def __call__(self, X):
        h, w = X.shape[:2]
        mask = X.sum(axis=-1) > 0
        X_1d = np.reshape(X, (-1, X.shape[-1])).copy()
        
        reduced = self.dr_method.fit_transform(X_1d)
        result = np.reshape(reduced, (h, w, -1))
        result[mask == 0] = 0
        result = np.uint8(255 * normalize(result))
        result[mask == 0] = 0
        return result



methods = ["dtne", "PCA", "PCUMAP", "PCC", "localmap", "pacmap", "UMAP", "TSNE", "trimap", "phate"]
def get_reducer(name, clusters, random_state=42):
    if name == "PCUMAP":
        return PCUMAP(n_components=3, device='cuda', epoch_to_start_correlation_loss=10, num_points=1000, random_state=random_state, lmc_batch_size=4096*2*2)
    elif name == "PCC":
        reducer = PCC(n_components=3, num_epochs=1000, num_points=1000, beta=20, k_epoch=50, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        reducer.fit_transform = lambda x: PCC.fit_transform(reducer, x, clusters)
        return reducer
    elif name == "PCA":
        return PCA(n_components=3, random_state=random_state)
    elif name == "UMAP":
        return UMAP(n_components=3, random_state=random_state)
    elif name == "TSNE":
        from sklearn.manifold import TSNE
        return TSNE(n_components=3, random_state=random_state)
    elif name == "dtne":
        from dtne import DTNE
        return DTNE(n_components=3, k_neighbors=10, l=2, random_state=random_state)
    elif name == "localmap":
        from pacmap import LocalMAP
        localmap = LocalMAP(n_components=3, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=random_state)
        localmap.fit_transform = lambda x: localmap.fit_transform(x)
        return localmap
    elif name == "pacmap":
        from pacmap import PaCMAP
        return PaCMAP(n_components=3, random_state=random_state)
    elif name == "trimap":
        import trimap
        return trimap.TRIMAP(n_dims=3, random_state=random_state)
    
    elif name == "phate":
        import phate
        return phate.PHATE(n_components=3, random_state=random_state)


def equalize_hist(viz):
    return cv2.merge([cv2.equalizeHist(c) for c in cv2.split(viz)])

def get_msi_metrics(X, viz, mouse_index, category_keys=["main_cat1", "Allen_name", "sub_cat2"]):
    polygons, categories = get_annotations("/home/jacob/Desktop/NRL4485-s2_reannotation_23-12-24_PAHJ.json", mouse_index, category_keys=category_keys)

    # Draw polygons over the visualization
    viz_with_annotations = viz.copy()
    mask = np.zeros(viz_with_annotations.shape[:2], dtype=np.uint8)

    # Create label encoder and fit to categories
    label_encoder = LabelEncoder()
    categories = label_encoder.fit_transform(categories)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(categories))))

    for polygon, category in zip(polygons, categories):
        category = int(category)
        # Convert polygon points to integer coordinates
        points = polygon.squeeze().astype(np.int32) // 2
        

        color = np.uint8(255 * colors[category])[:3]
        color = [int(c) for c in color]
        cv2.drawContours(viz_with_annotations, [points], -1, color=color, thickness=1, lineType=cv2.LINE_AA)
        cv2.drawContours(mask, [points], -1, color=category + 1, thickness=-1, lineType=cv2.LINE_AA)

    X_from_annotations = X[mask > 0]
    embeddings = viz[mask > 0]
    cluster_assignments = mask[mask > 0]


    metrics = Metrics(X_from_annotations, embeddings, cluster_assignments, sub_sample=10, n_clusters=[32, 64, 128], cluster_assignments={10: cluster_assignments}, label_local=True).compute_all_metrics()
    return metrics







from collections import defaultdict
results = defaultdict(list)

for random_state in tqdm.tqdm(range(10)):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    for mouse_index in tqdm.tqdm([0, 1, 2, 3]):
        X = np.load(f"/home/jacob/Desktop/msi_data/{mouse_index}.npy")
        X = X / (1e-6 + X.sum(axis=-1, keepdims=True))
        X = np.float32(X)
        X = X.transpose().transpose(1, 2, 0)[::-1, :, :].copy()
        X = X[::2, ::2, :].copy()


        from umap import UMAP
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        umap_reducer = UMAP(n_components=10, random_state=random_state)
        X_1d = np.reshape(X, (-1, X.shape[-1])).copy()
        X_umap = umap_reducer.fit_transform(X_1d)
        clusters = []
        n_clusters_list = [4, 8, 16, 32]
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto") 
            cluster_labels = kmeans.fit_predict(X_umap)
            clusters.append(cluster_labels)

        
        for name in methods:
            try:
                method = get_reducer(name, clusters, random_state)



                X = np.load(f"/home/jacob/Desktop/msi_data/{mouse_index}.npy")
                X = X / (1e-6 + X.sum(axis=-1, keepdims=True))
                X = np.float32(X)
                X = X.transpose().transpose(1, 2, 0)[::-1, :, :].copy()
                X = X[::2, ::2, :].copy()


                viz = MSIDRWrapper(method)(X)


                dst = "images"
                os.makedirs(dst, exist_ok=True)
                Image.fromarray(viz).save(f"{dst}/{name}_{mouse_index}_{random_state}.png")
                
                metrics = get_msi_metrics(X, viz, mouse_index)
                print(name, random_state, mouse_index, metrics)
                for key, val in metrics.items():
                    results[key].append(val)
                results["seed"].append(random_state)
                results["mouse_index"].append(mouse_index)
                results["method"].append(name)
            except Exception as e:
                print(f"Error running {name} on {mouse_index} with seed {random_state}: {str(e)}")
            print(results)
            print(name, random_state, mouse_index)
            pd.DataFrame.from_dict(results).to_csv(f"msi_benchmark.csv", index=False)
