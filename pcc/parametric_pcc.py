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

class ParametricPCC:
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
        model: torch.nn.Module = None,
        x: np.ndarray = None,
        y: np.ndarray = None,
        lr: float = 0.1,
        warmup_epochs: int = 10,
        ema_decay: float = 0.999,  # new
    ):
        """
        Initialize PCC with learning rate scheduling and warmup.

        Args:
            num_points: Number of reference points to sample
            num_epochs: Number of optimization epochs
            n_components: Number of output dimensions
            beta: Weight of correlation loss
            k_epoch: Frequency of correlation loss computation
            cluster: Whether to use clustering
            warmup_epochs: Number of warmup epochs
        """
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_points = num_points
        self.n_components = n_components
        self.beta = beta
        self.k_epoch = k_epoch
        self.cluster = cluster
        self.batch_size = batch_size
        self.temperature = temperature
        self.indices = landmarks
        self.random_state = random_state
        self.model = model
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        #self.ema = EMA(model, decay=self.ema_decay)
        self.create_classifiers([c.max() + 1 for c in y])
        self.seed_everything()
        self.step_counter = 0
        self.scheduler = None

        self.x = x
        self.y = [torch.from_numpy(y[i]).cuda() for i in range(len(y))]

        self.resample(x)

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

    def create_classifiers(self, num_clusters: List[int]) -> None:
        """
        Initialize embeddings and optimization parameters.

        Args:
            X: Input data matrix
            y: List of cluster labels for each layer
        """
        
        self.visualiation_to_cluster = []

        if self.cluster:
            for k in num_clusters:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(self.n_components, k)
                )
                if torch.cuda.is_available():
                    layer = layer.cuda()
                self.visualiation_to_cluster.append(layer)
        params = [{"params": self.model.parameters(), "weight_decay": 0}]
        if self.cluster:
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})

        self.optim = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=self._lr_lambda
        )

    def _lr_lambda(self, epoch: int) -> float:
        """ Learning rate scheduler with warmup """
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        return 1.0

    def resample(self, data) -> None:
        """
        Sample new reference points and compute distances.
        """

        if self.indices is None:
            self.indices = self.get_reference_points(data, self.num_points)
        reference_points = data[self.indices, :]
        self.reference_points = reference_points
        self.reference_points = torch.from_numpy(reference_points)
        if torch.cuda.is_available():
            self.reference_points = self.reference_points.cuda()

        euclidean = pairwise_distances(data, reference_points, metric="euclidean")
        self.euclidean = torch.from_numpy(euclidean)
        if torch.cuda.is_available():
            self.euclidean = self.euclidean.cuda()

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.predict(data)

    def step(self) -> np.ndarray:
        """
        Compute one optimization epoch using mini-batches.

        Args:
            epoch: Current epoch number

        Returns:
            Updated embeddings
        """

        batch_size = self.batch_size
        n_samples = self.x.shape[0]

        # Shuffle indices for random batching
        batch_indices = torch.randperm(n_samples)

        num_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_mask = batch_indices[start_idx:end_idx]

            x = torch.from_numpy(self.x[batch_mask])
            if torch.cuda.is_available():
                x = x.cuda()
            
            batch_outputs = self.model(x)
            batch_reference_points = self.model(self.reference_points)

            loss = 0

            # Get reference points for this batch
            low_d_distances = torch.cdist(batch_outputs, batch_reference_points)

            high_d_distances = self.euclidean[batch_mask]

            if self.cluster:
                for i in range(len(self.y)):
                    layer = self.visualiation_to_cluster[i]
                    clusters = self.y[i]
                    batch_clusters = clusters[batch_mask]
                    layer_output = layer(batch_outputs)
                    cluster_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(
                        layer_output / self.temperature, batch_clusters.long()
                    )
                    loss = loss + cluster_loss
                loss = loss / len(self.y)

            if self.step_counter % self.k_epoch == self.k_epoch - 1:
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()

        if self.step_counter % self.k_epoch == self.k_epoch - 1:
            # Linearly reduce k_epoch to 1 after 90% of epochs
            self.k_epoch = max(1, int(self.k_epoch * 0.92))

        self.step_counter += 1