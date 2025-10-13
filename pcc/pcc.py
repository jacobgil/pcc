import random
from typing import List, Optional

import numpy as np
import torch
import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from torchdr import UMAP


def correlation(
    pred: torch.Tensor, target: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
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
        num_epochs: int = 500,
        n_components: int = 2,
        beta: float = 5.0,
        k_epoch: int = 1,
        cluster: bool = True,
        batch_size=4096 * 2,
        temperature: float = 10.0,
        landmarks=None,
        random_state=42,
    ):
        """
        Initialize PCC.

        Args:
            num_points: Number of reference points to sample
            num_epochs: Number of optimization epochs
            n_components: Number of output dimensions
            beta: Weight of correlation loss
            k_epoch: Frequency of correlation loss computation
            cluster: Whether to use clustering
        """
        self.num_epochs = num_epochs
        self.num_points = num_points
        self.n_components = n_components
        self.clusters = None
        self.beta = beta
        self.k_epoch = k_epoch
        self.cluster = cluster
        self.batch_size = batch_size
        self.temperature = temperature
        self.indices = landmarks
        self.random_state = random_state

        self.seed_everything()

    def seed_everything(self):
        """
        Set random seeds for reproducibility.
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_reference_points(self, data: np.ndarray, Np: int) -> np.ndarray:
        """
        Reduces (NxD) data matrix from N to Np data points using random sampling.

        Args:
            data: Data matrix of shape [N, D]
            Np: Number of reference points

        Returns:
            Indices of selected reference points
        """
        N = data.shape[0]
        rng = np.random.RandomState(self.random_state)
        return rng.choice(list(range(N)), Np)

    def initialize_embeddings(self, X: np.ndarray, y: List[np.ndarray]) -> None:
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
                    torch.nn.Linear(self.n_components, num_clusters)
                )

                if torch.cuda.is_available():
                    layer = layer.cuda()
                self.visualiation_to_cluster.append(layer)

        self.data = X
        self.resample(number_of_points=self.num_points)

        self.visualization = 10 * torch.randn(len(self.data), self.n_components)
        if torch.cuda.is_available():
            self.visualization = self.visualization.cuda()

        self.visualization.requires_grad = True
        self.visualization = torch.nn.Parameter(self.visualization)
        params = [{"params": self.visualization, "weight_decay": 0}]
        if self.cluster:
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})

        self.optim = torch.optim.Adam(params, lr=1)

    def resample(self, number_of_points: int) -> None:
        """
        Sample new reference points and compute distances.

        Args:
            number_of_points: Number of reference points to sample
        """

        if self.indices is None:
            self.indices = self.get_reference_points(self.data, number_of_points)
        reference_points = self.data[self.indices, :]
        euclidean = pairwise_distances(self.data, reference_points, metric="euclidean")
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
        Compute one optimization epoch using mini-batches.

        Args:
            epoch: Current epoch number

        Returns:
            Updated embeddings
        """

        outputs = self.visualization
        batch_size = self.batch_size
        n_samples = outputs.shape[0]

        # Shuffle indices for random batching
        batch_indices = torch.randperm(n_samples)

        num_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_mask = batch_indices[start_idx:end_idx]

            batch_outputs = outputs[batch_mask]
            loss = 0

            # Get reference points for this batch

            batch_reference_points = outputs[self.indices]
            low_d_distances = torch.cdist(batch_outputs, batch_reference_points)

            high_d_distances = self.euclidean[batch_mask]

            if self.cluster:
                for i in range(len(self.clusters)):
                    layer = self.visualiation_to_cluster[i]
                    clusters = self.clusters[i]
                    batch_clusters = clusters[batch_mask]
                    cluster_loss = torch.nn.CrossEntropyLoss()(
                        layer(batch_outputs) / self.temperature, batch_clusters.long()
                    )
                    loss = loss + cluster_loss
                loss = loss / len(self.clusters)

            if epoch % self.k_epoch == self.k_epoch - 1:
                correlation_loss = 0

                high_d_distances = self.euclidean[batch_mask]
                correlation_loss = (
                    correlation_loss
                    - correlation(low_d_distances, high_d_distances).mean()
                )

                if correlation_loss > 0:
                    print(f"Correlation loss is positive: {correlation_loss}")

                alpha = -float(correlation_loss.detach().cpu().numpy())
                alpha = max(alpha, 1e-6)

                if self.cluster:
                    loss = loss + correlation_loss * self.beta / alpha
                else:
                    loss = correlation_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        if epoch % self.k_epoch == self.k_epoch - 1:
            # Linearly reduce k_epoch to 1 after 90% of epochs
            self.k_epoch = max(1, int(self.k_epoch * 0.92))

        return outputs.detach().cpu().numpy()


class PCUMAP(UMAP):
    """
    Principal Component UMAP - combines UMAP with correlation preservation.
    """

    def __init__(
        self,
        num_points: int = 1000,
        n_components: int = 2,
        epoch_to_start_correlation_loss: int = 10,
        correlation_loss_weight: float = 90000,
        lmc_batch_size: int = 4096 * 2 * 2 * 2 * 2,
        random_state=42,
        **kwargs,
    ):
        """
        Initialize PCUMAP.

        Args:
            num_points: Number of reference points to sample
            n_components: Number of output dimensions
            epoch_to_start_correlation_loss: Epoch to start computing correlation loss
            correlation_loss_weight: Weight of correlation loss
            **kwargs: Additional arguments passed to UMAP
        """

        super().__init__(**kwargs)
        self.epoch_for_comp = 0
        self.num_points = num_points
        self.n_components = n_components
        self.clusters = None
        self.epoch_to_start_correlation_loss = epoch_to_start_correlation_loss
        self.correlation_loss_weight = correlation_loss_weight
        self.lmc_batch_size = lmc_batch_size

    def get_reference_points(self, data: np.ndarray, Np: int) -> np.ndarray:
        """
        Reduces (NxD) data matrix from N to Np data points using random sampling.

        Args:
            data: Data matrix of shape [N, D]
            Np: Number of reference points

        Returns:
            Indices of selected reference points
        """
        N = data.shape[0]
        rng = np.random.RandomState(self.random_state)
        return rng.choice(list(range(N)), Np)

    def fit(self, X: np.ndarray, **kwargs) -> "PCUMAP":
        self.initialize_embeddings(X)
        return super().fit(X, **kwargs)

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.initialize_embeddings(X)
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
        self.indices = self.get_reference_points(self.data, number_of_points)
        reference_points = self.data[self.indices, :]
        euclidean = pairwise_distances(self.data, reference_points, metric="euclidean")

        self.euclidean = torch.from_numpy(euclidean)

    def _loss(self) -> torch.Tensor:
        """
        Compute combined UMAP and correlation loss.

        Returns:
            Combined loss value
        """

        self.epoch_for_comp = self.epoch_for_comp + 1

        umap_loss = super()._loss()
        if self.epoch_for_comp > self.epoch_to_start_correlation_loss:
            correlation_loss = self.correlation_loss()
            alpha = -float(correlation_loss.detach().cpu().numpy())
            alpha = max(alpha, 1e-6)
            return umap_loss + correlation_loss * self.correlation_loss_weight / alpha
        else:
            return umap_loss

    def correlation_loss(self) -> torch.Tensor:
        """
        Compute correlation loss between embeddings.

        Returns:
            Correlation loss value
        """
        outputs = self.embedding_
        reference_points = outputs[self.indices]

        correlation_loss = 0
        # Process in batches if data is larger than batch size
        if len(outputs) > self.lmc_batch_size:
            batch_losses = []
            for i in range(0, len(outputs), self.lmc_batch_size):
                batch_outputs = outputs[i : i + self.lmc_batch_size]
                batch_distances = torch.cdist(batch_outputs, reference_points)
                batch_euclidean = self.euclidean[i : i + self.lmc_batch_size]
                batch_euclidean = batch_euclidean.to(self.embedding_.device)
                batch_loss = correlation(batch_distances, batch_euclidean).mean()
                batch_losses.append(batch_loss)
            correlation_loss = correlation_loss - torch.stack(batch_losses).mean()
        else:
            # Handle devices in first epoch
            if self.epoch_for_comp == 0:
                self.euclidean = self.euclidean.to(self.embedding_.device)

            # Process all at once if data fits in batch
            output_distances = torch.cdist(outputs, reference_points)
            correlation_loss = (
                correlation_loss - correlation(output_distances, self.euclidean).mean()
            )

        return correlation_loss
