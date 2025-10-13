import sys
sys.path.append("benchmark")
from methods import tranasform_data, AVAILABLE_METHODS
from datasets import get_available_datasets, DatasetManager
import time
from pcc.metrics import Metrics
from sklearn.preprocessing import LabelEncoder
import tqdm
import numpy as np
from sklearn.cluster import KMeans
import torch
import gc
import random

def get_cluster_labels(X, n_clusters_list, random_state=42):
    from umap import UMAP

    try:
        umap_reducer = UMAP(n_components=10, random_state=random_state)
        X_umap = umap_reducer.fit_transform(X)
    except Exception as e:
        print(f"Error fitting UMAP: {e}")
        umap_reducer = UMAP(n_components=10, random_state=random_state, min_dist=0.2)
        X_umap = umap_reducer.fit_transform(X)
    clusters = []
    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto") 
        cluster_labels = kmeans.fit_predict(X_umap)
        clusters.append(cluster_labels)
    return clusters




import os
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
class BenchmarkManager:
    def __init__(self, output_dir="results", n_seeds=1, n_processes=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_seeds = n_seeds
        self.n_processes = n_processes
        self.n_clusters_list = [4, 8, 16, 32]
        
    def _get_result_path(self, dataset_name, method, seed):
        return self.output_dir / f"{dataset_name}_{method}_{seed}_result.csv"
        
    def _run_single_experiment(self, args):
        print(f"Running experiment {args}")
        dataset_name, seed = args
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Load data
        manager = DatasetManager()
        gc.collect()
        
        X, y, name = manager.load_dataset(dataset_name)
        if y is not None:
            y = LabelEncoder().fit_transform(y)

        # Get clusters
        clusters = get_cluster_labels(X, self.n_clusters_list, random_state=seed)
        cluster_assignments = dict(zip(self.n_clusters_list, clusters))
        
        results = []
        for method in tqdm.tqdm(AVAILABLE_METHODS):
            result_path = self._get_result_path(dataset_name, method, seed)
            
            # Skip if result already exists
            if result_path.exists():
                print(f"Result already exists for {dataset_name}, {method}, seed {seed}")
                results.extend(pd.read_csv(result_path).to_dict('records'))
                continue
                
            try:
                t0 = time.time()
                embeddings_2d = tranasform_data(X, method, clusters, random_state=seed)
                t1 = time.time()
                metrics = Metrics(X, embeddings_2d, y, sub_sample=20, 
                                n_clusters=self.n_clusters_list,
                                cluster_assignments=cluster_assignments).compute_all_metrics()
                t2 = time.time()
                
                result = {
                    'dataset': dataset_name,
                    'method': method,
                    'seed': seed,
                    #'embedding_time': t1 - t0,
                    #'metrics_time': t2 - t1,
                    **metrics
                }
                print(result)
                results.append(result)
                
                # Save individual result
                pd.DataFrame([result]).to_csv(result_path, index=False)
                
            except Exception as e:
                print(f"Error running {method} on {dataset_name} with seed {seed}: {str(e)}")
                
        return results

    def run_benchmark(self, dataset_name):
        # Determine seeds to run
        seeds_to_run = []
        for seed in range(self.n_seeds):
            # Check if any method is missing for this seed
            missing_methods = []
            for method in AVAILABLE_METHODS:
                result_path = self._get_result_path(dataset_name, method, seed)
                if not result_path.exists():
                    missing_methods.append(method)
            if missing_methods:
                seeds_to_run.append(seed)
                print(f"Seed {seed} missing methods: {missing_methods}")
        
        if not seeds_to_run:
            print(f"All seeds and methods already completed for {dataset_name}")
            return
            
        # Run experiments
        experiment_args = [(dataset_name, seed) for seed in seeds_to_run]
        
        if self.n_processes and self.n_processes > 1:
            with Pool(self.n_processes) as pool:
                all_results = pool.map(self._run_single_experiment, experiment_args)
        else:
            all_results = [self._run_single_experiment(args) for args in tqdm.tqdm(experiment_args)]
            
        # Combine all results into a single summary file
        all_results_path = self.output_dir / f"{dataset_name}_results.csv"
        all_results_df = []
        
        # Load all individual results
        for seed in range(self.n_seeds):
            for method in AVAILABLE_METHODS:
                result_path = self._get_result_path(dataset_name, method, seed)
                if result_path.exists():
                    result_df = pd.read_csv(result_path)
                    all_results_df.append(result_df)
        
        if all_results_df:
            pd.concat(all_results_df).to_csv(all_results_path, index=False)
            print(f"Combined results saved to {all_results_path}")

t0 = time.time()
# Initialize and run benchmark
datasets = get_available_datasets()[::-1]

datasets = ['zheng', 'visium', 'squidpy_visium_hne', 'scvelo_pancreas', 'scvelo_dentategyrus', 'qsar_biodeg', 'pbmc68k', 'pbmc3k', 'paul15', 'parkinsons', 'pancreas_segerstolpe', 'pancreas_muraro', 'pancreas_baron', 'mouse_msi', 'mice_protein', 'macosko', 'krumsiek11', 'breast_cancer', 'birthweight', 'arcene']
datasets = datasets[::-1]


benchmark = BenchmarkManager(n_processes=None)
for dataset in datasets:
    benchmark.run_benchmark(dataset)

print(f"Time taken: {time.time() - t0} seconds")