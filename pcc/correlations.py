import torch
import torch.nn.functional as F

# ----------------------------
# Utilities
# ----------------------------

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Double-center a Gram matrix: Kc = H K H, with H = I - 11^T / n."""
    n = K.size(0)
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    H = I - K.new_ones(n, n) / n
    return H @ K @ H

def _sqdist(X: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances (n x n), differentiable & stable."""
    # dist2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    n2 = (X * X).sum(dim=1, keepdim=True)
    dist2 = n2 + n2.t() - 2.0 * (X @ X.t())
    return dist2.clamp_min(0.0)

def _rbf_gram(X: torch.Tensor, sigma: torch.Tensor | float | None = None) -> torch.Tensor:
    """RBF (Gaussian) Gram matrix with bandwidth sigma (on *distance*, not squared)."""
    D2 = _sqdist(X)
    if sigma is None:
        # median heuristic on sqrt distances (detach to avoid noisy grads)
        with torch.no_grad():
            # use upper triangle without diag
            iu = torch.triu_indices(D2.size(0), D2.size(0), offset=1, device=X.device)
            med = torch.sqrt(D2[iu[0], iu[1]] + 1e-12).median()
            s = med if torch.isfinite(med) and med > 0 else torch.tensor(1.0, device=X.device, dtype=X.dtype)
        sigma = s
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=X.device, dtype=X.dtype)
    gamma = 1.0 / (2.0 * sigma.clamp_min(1e-12).pow(2))
    return torch.exp(-gamma * D2)

# ----------------------------
# HSIC (biased estimator)
# ----------------------------

def hsic_biased_from_grams(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    HSIC with biased estimator: (1/(n-1)^2) * tr(Kc @ Lc), where Kc, Lc are centered Gram matrices.
    K, L: (n x n) symmetric PSD Gram matrices (e.g., linear or RBF).
    Returns a scalar tensor.
    """
    n = K.size(0)
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return torch.trace(Kc @ Lc) / ((n - 1.0) ** 2 + 1e-12)

def hsic(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_x: str = "rbf",
    kernel_y: str = "rbf",
    sigma_x: float | None = None,
    sigma_y: float | None = None,
) -> torch.Tensor:
    """
    HSIC between two views X (n x dx) and Y (n x dy).
    kernel_* in {"rbf","linear"}; for "rbf" you may pass sigma_* (distance bandwidth).
    """
    if kernel_x == "linear":
        K = X @ X.t()
    elif kernel_x == "rbf":
        K = _rbf_gram(X, sigma_x)
    else:
        raise ValueError("kernel_x must be 'linear' or 'rbf'.")

    if kernel_y == "linear":
        L = Y @ Y.t()
    elif kernel_y == "rbf":
        L = _rbf_gram(Y, sigma_y)
    else:
        raise ValueError("kernel_y must be 'linear' or 'rbf'.")

    return hsic_biased_from_grams(K, L)

# ----------------------------
# CKA (Centered Kernel Alignment)
# ----------------------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Linear CKA (fast). Center across samples, then:
      CKA = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F).
    Returns a scalar in [0,1] (numerically ~[0,1]).
    """
    # Center across samples (dim=0)
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices
    XTX = Xc @ Xc.t()  # n x n
    YTY = Yc @ Yc.t()  # n x n
    XTY = Xc @ Yc.t()  # n x n
    
    # Compute CKA
    num = (XTY * XTY).sum()
    den = torch.sqrt((XTX * XTX).sum() * (YTY * YTY).sum() + eps)
    return (num / den).clamp(0.0, 1.0)

def rbf_cka(
    X: torch.Tensor, Y: torch.Tensor,
    sigma_x: float | None = None, sigma_y: float | None = None,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    RBF CKA via HSIC normalization:
      CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y)).
    Uses biased HSIC with RBF kernels (median heuristic if sigma_* is None).
    """
    hs_xy = hsic(X, Y, kernel_x="rbf", kernel_y="rbf", sigma_x=sigma_x, sigma_y=sigma_y)
    hs_xx = hsic(X, X, kernel_x="rbf", kernel_y="rbf", sigma_x=sigma_x, sigma_y=sigma_x)
    hs_yy = hsic(Y, Y, kernel_x="rbf", kernel_y="rbf", sigma_x=sigma_y, sigma_y=sigma_y)
    return (hs_xy / (torch.sqrt(hs_xx * hs_yy + eps))).clamp(min=0.0)