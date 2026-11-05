import torch
from torchdr import UMAP
import numpy as np
import tqdm
from typing import List, Optional
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans, kmeans_plusplus


def correlation(pred: torch.Tensor, target: torch.Tensor,
                dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute correlation between two tensors.

    Args:
        pred: Prediction tensor
        target: Target tensor
        dim: Dimension along which to compute correlation. If None, treats tensors as 1D.

    Returns:
        Correlation coefficient as a tensor
    """
    if dim is None:
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return (pred * target).sum()
    else:
        pred = pred - pred.mean(dim=dim)[:, None]
        pred = pred / pred.norm(dim=dim)[:, None]
        target = target - target.mean(dim=dim)[:, None]
        target = target / target.norm(dim=dim)[:, None]
        return (pred * target).sum(dim=dim).mean()


class PCC:
    """
    Principal Component Correlation class for dimensionality reduction with correlation preservation.
    """

    def __init__(
            self,
            num_points: int = 1000,
            regularization_strength: float = 0.005,
            sampling: str = "random",
            num_epochs: int = 500,
            n_components: int = 2,
            beta: float = 5.0,
            spearman: bool = False,
            pearson: bool = True,
            k_epoch: int = 1,
            cluster: bool = True):
        """
        Initialize PCC.

        Args:
            num_points: Number of reference points to sample
            regularization_strength: Strength of soft ranking regularization
            sampling: Method for sampling reference points ("random", "kmeans++", "kmeans", "coreset")
            num_epochs: Number of optimization epochs
            n_components: Number of output dimensions
            beta: Weight of correlation loss
            spearman: Whether to use Spearman correlation
            pearson: Whether to use Pearson correlation
            k_epoch: Frequency of correlation loss computation
            cluster: Whether to use clustering
        """
        self.num_epochs = num_epochs
        self.regularization_strength = regularization_strength
        self.sampling = sampling
        self.num_points = num_points
        self.n_components = n_components
        self.clusters = None
        self.beta = beta
        self.spearman = spearman
        self.pearson = pearson
        self.k_epoch = k_epoch
        self.cluster = cluster

    def __repr__(self) -> str:
        return f"Saliency PCC: num_epochs: {self.num_epochs} regularization_strength: {self.regularization_strength} \
            sampling: {self.sampling} num_points:{self.num_points} pearson: {self.pearson} spearman: {self.spearman}"

    def get_reference_points(self, data: np.ndarray, Np: int) -> np.ndarray:
        """
        Reduces (NxD) data matrix from N to Np data points.

        Args:
            data: Data matrix of shape [N, D]
            Np: Number of reference points

        Returns:
            Indices of selected reference points
        """
        N = data.shape[0]
        D = data.shape[1]
        method = self.sampling

        if method == "random":
            return np.random.choice(list(range(N)), Np)

        elif method == "kmeans++":
            _, indices = kmeans_plusplus(data, n_clusters=Np, random_state=0)
            return indices

        elif method == "kmeans":
            kmeans = KMeans(
                n_clusters=Np,
                random_state=0,
                n_init="auto").fit(data)
            return pairwise_distances(
                data, kmeans.cluster_centers_).argmin(
                axis=0)

        elif method == "coreset":
            u = np.mean(data, axis=0)
            q = np.linalg.norm(data - u, axis=1)**2
            sum = np.sum(q)
            d = q / sum
            q = 0.5 * (d + 1.0 / N)
            return np.random.choice(N, Np, p=q)

    def initialize_embeddings(self, X: np.ndarray,
                              y: List[np.ndarray]) -> None:
        """
        Initialize embeddings and optimization parameters.

        Args:
            X: Input data matrix
            y: List of cluster labels for each layer
        """
        self.clusters = []
        self.visualiation_to_cluster = []
        if self.cluster:
            for labels in y:
                label_tensor = torch.tensor(labels)
                if torch.cuda.is_available():
                    label_tensor = label_tensor.cuda()
                self.clusters.append(label_tensor)
                num_clusters = labels.max() + 1

                layer = torch.nn.Sequential(
                    torch.nn.Linear(
                        self.n_components,
                        num_clusters))

                if torch.cuda.is_available():
                    layer = layer.cuda()
                self.visualiation_to_cluster.append(layer)

        self.data = X
        self.resample(number_of_points=self.num_points)
        self.visualization = 10 * \
            torch.randn(len(self.data), self.n_components)
        if torch.cuda.is_available():
            self.visualization = self.visualization.cuda()

        self.visualization.requires_grad = True
        self.visualization = torch.nn.Parameter(self.visualization)
        params = [{'params': self.visualization, 'weight_decay': 0}]
        if self.cluster:
            for l in self.visualiation_to_cluster:
                params.append({'params': l.parameters(), 'weight_decay': 0})
        self.optim = torch.optim.Adam(params, lr=1)

    def resample(self, number_of_points: int) -> None:
        """
        Sample new reference points and compute distances.

        Args:
            number_of_points: Number of reference points to sample
        """
        self.indices = self.get_reference_points(
            self.data, number_of_points)
        reference_points = self.data[self.indices, :]
        euclidean = pairwise_distances(
            self.data,
            reference_points,
            metric='euclidean')
        if self.spearman:
            self.euclidean_ranks = torch.from_numpy(
                euclidean.argsort().argsort()).float()
            if torch.cuda.is_available():
                self.euclidean_ranks = self.euclidean_ranks.cuda()
        if self.pearson:
            self.euclidean = torch.from_numpy(euclidean)
            if torch.cuda.is_available():
                self.euclidean = self.euclidean.cuda()

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit model and transform data.

        Args:
            X: Input data matrix
            y: List of cluster labels

        Returns:
            Transformed data
        """
        self.initialize_embeddings(X, y)

        for epoch in tqdm.tqdm(range(self.num_epochs)):
            output = self.compute_epoch(epoch + 1)
        return output

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.predict(data)

    def compute_epoch(self, epoch: int) -> np.ndarray:
        """
        Compute one optimization epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Updated embeddings
        """
        outputs = self.visualization
        loss = 0
        reference_points = outputs[self.indices]
        output_distances = torch.cdist(outputs, reference_points)

        if self.cluster:
            for layer, clusters in zip(
                    self.visualiation_to_cluster, self.clusters):
                o = layer(outputs)
                cluster_loss = torch.nn.CrossEntropyLoss()(o, clusters.long())
                loss = loss + cluster_loss
            loss = loss / len(self.clusters)

        if (epoch % self.k_epoch == self.k_epoch - 1):
            correlation_loss = 0
            if self.spearman:
                import torchsort
                output_ranks = torchsort.soft_rank(
                    output_distances,
                    regularization_strength=self.regularization_strength)
                correlation_loss = correlation_loss - \
                    correlation(output_ranks, self.euclidean_ranks, dim=-1).mean()
            if self.pearson:
                correlation_loss = correlation_loss - \
                    correlation(output_distances, self.euclidean).mean()

            if self.pearson and self.spearman:
                correlation_loss = correlation_loss / 2

            alpha = abs(correlation_loss.detach().cpu().numpy())
            if self.cluster:
                loss = loss + correlation_loss * self.beta / alpha
            else:
                loss = correlation_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return outputs.detach().cpu().numpy()


class PCUMAP(UMAP):
    """
    Principal Component UMAP - combines UMAP with correlation preservation.
    """

    def __init__(
            self,
            num_points: int = 500,
            regularization_strength: float = 0.005,
            sampling: str = "random",
            n_components: int = 2,
            beta: float = 10.0,
            spearman: bool = False,
            pearson: bool = True,
            epoch_to_start_correlation_loss: int = 10,
            correlation_loss_weight: float = 90000,
            **kwargs):
        """
        Initialize PCUMAP.

        Args:
            num_points: Number of reference points to sample
            regularization_strength: Strength of soft ranking regularization
            sampling: Method for sampling reference points ("random", "kmeans++", "kmeans", "coreset")
            n_components: Number of output dimensions
            beta: Weight of correlation loss
            spearman: Whether to use Spearman correlation
            pearson: Whether to use Pearson correlation
            epoch_to_start_correlation_loss: Epoch to start computing correlation loss
            correlation_loss_weight: Weight of correlation loss
            **kwargs: Additional arguments passed to UMAP
        """

        super().__init__(**kwargs)
        self.epoch_for_comp = 0
        self.regularization_strength = regularization_strength
        self.sampling = sampling
        self.num_points = num_points
        self.n_components = n_components
        self.clusters = None
        self.beta = beta
        self.spearman = spearman
        self.pearson = pearson
        self.epoch_to_start_correlation_loss = epoch_to_start_correlation_loss
        self.correlation_loss_weight = correlation_loss_weight

    def get_reference_points(self, data: np.ndarray, Np: int) -> np.ndarray:
        """
        Reduces (NxD) data matrix from N to Np data points.

        Args:
            data: Data matrix of shape [N, D]
            Np: Number of reference points

        Returns:
            Indices of selected reference points
        """
        N = data.shape[0]
        method = self.sampling

        if method == "random":
            return np.random.choice(list(range(N)), self.num_points)

        elif method == "kmeans++":
            _, indices = kmeans_plusplus(data, n_clusters=Np, random_state=0)
            return indices

        elif method == "kmeans":
            kmeans = KMeans(
                n_clusters=Np,
                random_state=0,
                n_init="auto").fit(data)
            return pairwise_distances(
                data, kmeans.cluster_centers_).argmin(
                axis=0)

        elif method == "coreset":
            u = np.mean(data, axis=0)
            q = np.linalg.norm(data - u, axis=1)**2
            sum = np.sum(q)
            d = q / sum
            q = 0.5 * (d + 1.0 / N)
            return np.random.choice(N, Np, p=q)

    def fit(self, X: np.ndarray, **kwargs) -> 'PCUMAP':
        self.initialize_embeddings(X)
        return super().fit(X, **kwargs)

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.initialize_embeddings(X)
        print("got embeddings")
        return super().fit_transform(X, **kwargs)

    def initialize_embeddings(self, data: np.ndarray) -> None:
        """
        Initialize embeddings and compute reference points.

        Args:
            data: Input data matrix
        """
        self.data = data
        self.resample(number_of_points=self.num_points)

    def resample(self, number_of_points: int) -> None:
        """
        Sample new reference points and compute distances.

        Args:
            number_of_points: Number of reference points to sample
        """
        self.indices = self.get_reference_points(
            self.data, number_of_points)
        reference_points = self.data[self.indices, :]
        euclidean = pairwise_distances(
            self.data,
            reference_points,
            metric='euclidean')

        if self.spearman:
            self.euclidean_ranks = torch.from_numpy(
                euclidean.argsort().argsort()).float()
        if self.pearson:
            self.euclidean = torch.from_numpy(euclidean)

    def _correlation_step(self) -> bool:
        """
        Advance the iteration counter, move the reference distances to the
        embedding's device, and tell whether the correlation term is active.

        Returns:
            True if the correlation loss should be applied in this iteration
        """
        device = self.embedding_.device
        if self.spearman and self.euclidean_ranks.device != device:
            self.euclidean_ranks = self.euclidean_ranks.to(device)
        if self.pearson and self.euclidean.device != device:
            self.euclidean = self.euclidean.to(device)

        self.epoch_for_comp = self.epoch_for_comp + 1
        return self.epoch_for_comp > self.epoch_to_start_correlation_loss

    def _loss(self) -> torch.Tensor:
        """
        Compute combined UMAP and correlation loss (torchdr < 0.3, autograd).

        Returns:
            Combined loss value
        """
        use_correlation = self._correlation_step()
        umap_loss = super()._loss()
        if use_correlation:
            correlation_loss = self.correlation_loss()
            return umap_loss + correlation_loss * self.correlation_loss_weight
        else:
            return umap_loss

    def _compute_gradients(self) -> torch.Tensor:
        """
        Add the correlation gradient to UMAP's closed-form gradients.

        From torchdr 0.3 on, UMAP no longer builds a loss for autograd: it
        computes its gradients in closed form in ``_compute_gradients``, so
        ``_loss`` is never called and the correlation term would be silently
        dropped. Here the correlation loss is differentiated with autograd and
        added to the UMAP gradients of the points this process updates.

        Returns:
            Gradients w.r.t. the embedding rows in ``chunk_indices_``
        """
        gradients = super()._compute_gradients()
        if gradients is None or not self._correlation_step():
            return gradients

        with torch.enable_grad():
            embedding = self.embedding_.detach().requires_grad_(True)
            loss = self.correlation_loss(embedding) * self.correlation_loss_weight
            (correlation_gradients,) = torch.autograd.grad(loss, embedding)

        chunk_indices = getattr(self, "chunk_indices_", None)
        if chunk_indices is not None:
            correlation_gradients = correlation_gradients[chunk_indices]
        return gradients + correlation_gradients.to(gradients.dtype)

    def correlation_loss(self, embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute correlation loss between embeddings.

        Args:
            embedding: Embedding to evaluate; defaults to ``self.embedding_``

        Returns:
            Correlation loss value
        """
        outputs = self.embedding_ if embedding is None else embedding
        reference_points = outputs[self.indices]
        output_distances = torch.cdist(outputs, reference_points)

        correlation_loss = 0
        if self.spearman:
            import torchsort
            output_ranks = torchsort.soft_rank(
                output_distances,
                regularization_strength=self.regularization_strength)
            correlation_loss = correlation_loss - \
                correlation(output_ranks, self.euclidean_ranks).mean()
        if self.pearson:
            correlation_loss = correlation_loss - \
                correlation(output_distances, self.euclidean).mean()

        if self.pearson and self.spearman:
            correlation_loss = correlation_loss / 2

        return correlation_loss
