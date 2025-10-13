from pacmap import LocalMAP
from pcc import PCUMAP, PCC
import numpy as np

AVAILABLE_METHODS = [
    "trimap",
    "PCUMAP",
    "PCC2",
    "PCC2_No_Cluster", 
    "PCC2_No_Corr",
    "PCC2linear_cluster_weight08",
    "PCC2linear_cluster_weight05",
    "PCC2linear_cluster_weight0",

    "PCC",
    "PCC3",
    "PCC4",
    "PCC5",
    "PCC6",



    "dtne",
    "localmap",
    "umap",
    "tsne",
    #"isomap", 
    "pacmac",
    "phate",
    "pca"
]

#AVAILABLE_METHODS = ["PCC5_tmp1"]

#AVAILABLE_METHODS = ["MiCS+LMC", "MiCS+LMC_no_corr", "MiCS+LMC_no_cluster", "MiCS+LMC_beta1", "MiCS+LMC_beta5", "MiCS+LMC_beta10", "MiCS+LMC_k_epoch1", "MiCS+LMC_k_temperate_1", "UMAP+LMC", "UMAP+LMC_correlation_loss_weight100", "UMAP+LMC_correlation_loss_weight1000", "UMAP+LMC_correlation_loss_weight10000", "UMAP+LMC_correlation_loss_weight100000", "UMAP+LMC_10points", "UMAP+LMC_100points", "UMAP+LMC_500points", "UMAP+LMC_2000points", "UMAP+LMC_epoch_to_start_correlation_loss0", "UMAP+LMC_epoch_to_start_correlation_loss1", "UMAP+LMC_epoch_to_start_correlation_loss2", "UMAP+LMC_epoch_to_start_correlation_loss3", "UMAP+LMC_epoch_to_start_correlation_loss4", "UMAP+LMC_epoch_to_start_correlation_loss5", "UMAP+LMC_epoch_to_start_correlation_loss20", "UMAP+LMC_epoch_to_start_correlation_loss30"]
AVAILABLE_METHODS = ["MiCS+LMC_beta5", "MiCS+LMC_k_epoch100", "MiCS+LMC", "UMAP+LMC", "MiCS+LMC_beta10", "MiCS+LMC_k_epoch1", "UMAP+LMC_epoch_to_start_correlation_loss20", "MiCS+LMC_k_temperate_1", "UMAP+LMC_correlation_loss_weight10000"]
AVAILABLE_METHODS = AVAILABLE_METHODS + ["trimap", "dtne", "localmap", "umap", "tsne", "pacmac", "phate", "pca"]


def tranasform_data(X, method, clusters, random_state=42):
    X = np.float32(X)

    if method == "MiCS+LMC":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=20, k_epoch=50, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "MiCS+LMC_no_corr":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=0, k_epoch=50, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "MiCS+LMC_no_cluster":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=20, k_epoch=1, temperature=20, batch_size=4096*2*2, cluster=False, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "MiCS+LMC_beta1":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=1, k_epoch=50, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "MiCS+LMC_beta5":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=50, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "MiCS+LMC_beta10":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=10, k_epoch=50, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "MiCS+LMC_k_epoch1":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=20, k_epoch=1, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "MiCS+LMC_k_epoch100":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=20, k_epoch=100, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)


    elif method == "MiCS+LMC_k_temperate_1":
        pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=20, k_epoch=50, temperature=1, batch_size=4096*2*2, cluster=True, random_state=random_state)
        return pcc_reducer.fit_transform(X, clusters)

    elif method == "UMAP+LMC":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=1000, random_state=random_state, min_dist=0.2)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_correlation_loss_weight100":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=1000, random_state=random_state, min_dist=0.1, correlation_loss_weight=100)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_correlation_loss_weight1000":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=1000, random_state=random_state, min_dist=0.1, correlation_loss_weight=1000)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_correlation_loss_weight10000":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=1000, random_state=random_state, min_dist=0.1, correlation_loss_weight=10000)
        return pcumap_reducer.fit_transform(X)


    elif method == "UMAP+LMC_correlation_loss_weight100000":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=1000, random_state=random_state, min_dist=0.1, correlation_loss_weight=100000)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_10points":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=10, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_100points":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=100, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_500points":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=500, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)


    elif method == "UMAP+LMC_2000points":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=2000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)


    elif method == "UMAP+LMC_epoch_to_start_correlation_loss0":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=0, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_epoch_to_start_correlation_loss1":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=1, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_epoch_to_start_correlation_loss2":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=2, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_epoch_to_start_correlation_loss3":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=3, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_epoch_to_start_correlation_loss4":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=4, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_epoch_to_start_correlation_loss5":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=5, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_epoch_to_start_correlation_loss20":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=20, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)

    elif method == "UMAP+LMC_epoch_to_start_correlation_loss30":
        pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=30, num_points=1000, random_state=random_state, min_dist=0.1)
        return pcumap_reducer.fit_transform(X)











    # if method == "PCUMAP":
    #     pcumap_reducer = PCUMAP(device='cuda', epoch_to_start_correlation_loss=10, num_points=1000, random_state=random_state, min_dist=0.2)
    #     return pcumap_reducer.fit_transform(X)

    # if method == "PCC2":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=1, temperature=50, batch_size=4096*2*2, linear_cluster_weight=1.0, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)
    # elif method == "PCC2_No_Corr":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=0, k_epoch=1, temperature=50, batch_size=4096*2*2, linear_cluster_weight=1.0, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)
    # elif method == "PCC2_No_Cluster":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=1, temperature=50, batch_size=4096*2*2, linear_cluster_weight=1.0, cluster=False, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)
    # elif method == "PCC2linear_cluster_weight08":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=1, temperature=50, batch_size=4096*2*2, linear_cluster_weight=0.8, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)
    # elif method == "PCC2linear_cluster_weight05":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=1, temperature=50, batch_size=4096*2*2, linear_cluster_weight=0.5, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)
    # elif method == "PCC2linear_cluster_weight0":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=1, temperature=50, batch_size=4096*2*2, linear_cluster_weight=0.8, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)


    # elif method == "PCC":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=100, temperature=50, batch_size=4096*2*2, linear_cluster_weight=1.0, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)


    # elif method == "PCC3":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=10, k_epoch=100, temperature=20, batch_size=4096*2*2, linear_cluster_weight=1.0, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)

    # elif method == "PCC4":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=10, k_epoch=50, temperature=20, batch_size=4096*2*2, linear_cluster_weight=1.0, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)

    # elif method == "PCC5":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=20, k_epoch=50, temperature=20, batch_size=4096*2*2, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)


    # elif method == "PCC5_no_cluster":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=20, k_epoch=1, temperature=20, batch_size=4096*2*2, cluster=False, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)





    # elif method == "PCC6":
    #     pcc_reducer = PCC(n_components=2, num_epochs=1000, num_points=1000, pearson=True, beta=5, k_epoch=100, temperature=1, batch_size=4096*2*2, linear_cluster_weight=1.0, cluster=True, random_state=random_state)
    #     return pcc_reducer.fit_transform(X, clusters)

    elif method == "dtne":
        from dtne import DTNE
        dtne_operator = DTNE(k_neighbors=10, l=2, random_state=random_state)
        return dtne_operator.fit_transform(X)
    elif method == "localmap":
        from pacmap import LocalMAP
        reducer = LocalMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=random_state)
        return reducer.fit_transform(X, init="pca")

    if method == "umap":
        from umap import UMAP
        reducer = UMAP(random_state=random_state)
        return reducer.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(random_state=random_state)
        return reducer.fit_transform(X)
    elif method == "isomap":
        from sklearn.manifold import Isomap
        reducer = Isomap()
        return reducer.fit_transform(X)
    elif method == "pacmac":
        from pacmap import PaCMAP
        reducer = PaCMAP(random_state=random_state)
        return reducer.fit_transform(X)
    elif method == "trimap":
        import trimap
        reducer = trimap.TRIMAP(random_state=random_state)
        return reducer.fit_transform(X)
    elif method == "phate":
        import phate
        reducer = phate.PHATE(random_state=random_state)
        return reducer.fit_transform(X)
    elif method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=random_state).fit_transform(X)
