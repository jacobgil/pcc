from zadu import zadu


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.svm import SVC


import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def knn_label_metrics(X_emb_full, y_full, ks=(1,5,15)):
    """Return neighborhood hit (NH@k) and kNN majority-vote classification (Acc/F1) per k.
       Uses leave-one-out style by dropping the self neighbor."""
    X_emb = X_emb_full[::1, :]
    y = y_full[::1]
    n = len(y)
    nn = NearestNeighbors(n_neighbors=max(ks)+1, metric='euclidean').fit(X_emb)
    dists, idxs = nn.kneighbors(X_emb)  # first neighbor is self
    out = {}
    for k in ks:
        neigh = idxs[:, 1:k+1]
        neigh_labels = y[neigh]
        # Neighborhood Hit: fraction of neighbors with same label
        nh = (neigh_labels == y[:, None]).mean()
        # kNN majority vote prediction
        # (break ties by smallest label to keep deterministic)
        preds = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=int(y.max())+1).argmax(),
            1, neigh_labels
        )
        acc = accuracy_score(y, preds)
        f1m = f1_score(y, preds, average='macro')
        #out[k] = dict(NH=nh, acc=acc, f1_macro=f1m)
        out[k] = acc
    return np.mean(list(out.values()))


def linear_probe_cv(X_emb_full, y_full, C=1.0, folds=5, seed=0):
    """Stratified k-fold logistic regression; reports Acc/F1-macro."""
    X_emb = X_emb_full[::1, :]
    y = y_full[::1]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X_emb, y):
        clf = SVC(C=C, kernel='linear', random_state=0)
        clf.fit(X_emb[tr], y[tr])
        p = clf.predict(X_emb[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average='macro'))
    return np.mean(accs)


class Metrics:
    def __init__(self, X, embedding, y=None, cluster_assignments=None, sub_sample=32, n_clusters=[4, 8, 16, 32], label_local=False):
        self.X = np.float32(X)
        self.embedding = embedding
        self.y = y

        

        self.cluster_assignments = cluster_assignments
        self.sub_sample = sub_sample
        self.n_clusters = n_clusters
        self.label_local = label_local


    

    def get_zadu_metrics(self):
        """Get standard ZADU metrics"""
        spec = [{"id": "tnc"},
                {"id": "mrre"},
                #{"id": "stress"},
                #{"id": "kl_div"},
                {"id": "pr"},
                {"id": "srho"}]
        
        if self.label_local:
            spec.append({"id": "ca_tnc"})

        if self.label_local:
            scores = zadu.ZADU(spec, self.X[::self.sub_sample, :]).measure(self.embedding[::self.sub_sample, :], self.y[::self.sub_sample])
        else:
            scores = zadu.ZADU(spec, self.X[::self.sub_sample, :]).measure(self.embedding[::self.sub_sample, :])        
        result = {}
        for item in scores:
            for k, v in item.items():
                result[k] = float(v)
        return result
        
    def get_clustering_metrics(self):
        """Get clustering quality metrics if labels provided"""
            
        
        ami_scores = {}
        
        if self.y is not None:
            for k in self.n_clusters:
                kmeans = KMeans(n_clusters=k, random_state=42)
                pred = kmeans.fit_predict(self.embedding)
                ami_scores[f'ami_{k}'] = adjusted_mutual_info_score(self.y, pred)
                
            ami_scores = {'ami_avg':  float(np.mean(list(ami_scores.values())))}
        else:
            ami_scores = {'ami_avg':  np.nan}
        return ami_scores
        
    def get_cluster_preservation(self):
        """Get preservation metrics between and within clusters"""
        if self.cluster_assignments is None:
            return {}
            
        preservation_scores = {}
        within_scores = []
        between_scores = []

        for k, assignments in self.cluster_assignments.items():
            labels = np.asarray(assignments)
            clusters = np.unique(labels)
            K = k

            # 1) Cluster centers in original space / embedding (same cluster order)
            centers_orig = np.vstack([self.X[labels == c].mean(axis=0) for c in clusters])
            centers_emb  = np.vstack([self.embedding[labels == c].mean(axis=0) for c in clusters])

            # 2) Within-cluster preservation: weighted mean over clusters
            spec = [{"id": "pr"}]
            scores, weights = [], []
            for c in clusters:
                mask = (labels == c)
                n_c = int(mask.sum())
                if n_c >= 3:  # avoid degenerate/ill-defined cases
                    # Calculate number of samples to use (10% of cluster size)
                    n_samples = max(3, int(0.1 * n_c))
                    # Get indices of samples in this cluster
                    cluster_indices = np.where(mask)[0]
                    # Randomly sample 10% of indices
                    sampled_indices = np.random.choice(cluster_indices, size=n_samples, replace=False)
                    # Create mask for sampled indices
                    sample_mask = np.zeros_like(mask)
                    sample_mask[sampled_indices] = True
                    
                    zadu_results = zadu.ZADU(spec, self.X[sample_mask]).measure(self.embedding[sample_mask])
                    s = {}
                    for item in zadu_results:
                        for k, v in item.items():
                            s[k] = float(v)
            
                    s = s['pearson_r']
                    scores.append(s)
                    weights.append(n_c)

            within_score = (np.average(scores, weights=weights) if scores else np.nan)
            within_scores.append(within_score)

            # 3) Between-centers preservation (global cluster layout)
            # Note: meaningful only if K >= 3
            zadu_results = zadu.ZADU(spec, centers_orig).measure(centers_emb)
            s = {}
            for item in zadu_results:
                for k, v in item.items():
                    s[k] = float(v)
            
            between_score = (s['pearson_r']
                            if K >= 3 else np.nan)
            between_scores.append(between_score)
            
            
        preservation_scores = {'within_cluster_pr_avg': float(np.mean(within_scores)), 'between_centers_pr_avg': float(np.mean(between_scores))}

    


        return preservation_scores
        
    def get_classification_metrics(self):
        """Get classification metrics"""
        result = {}
        if self.y is not None:
            try:
                result['knn_label_metrics'] = knn_label_metrics(self.embedding, self.y)
            except Exception as e:
                result['knn_label_metrics'] = np.nan
            try:
                result['linear_probe_cv'] = linear_probe_cv(self.embedding, self.y)
            except Exception as e:
                result['linear_probe_cv'] = np.nan
        else:
            result['knn_label_metrics'] = np.nan
            result['linear_probe_cv'] = np.nan
        return result
            

    def compute_all_metrics(self):
        """Compute all available metrics"""
        metrics = {}
        metrics.update(self.get_zadu_metrics())
        #metrics.update(self.get_clustering_metrics())
        metrics.update(self.get_cluster_preservation())
        metrics.update(self.get_classification_metrics())
        return metrics

