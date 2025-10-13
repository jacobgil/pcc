# fast_k_subspaces_quick.py
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def _kmeanspp_init(X: torch.Tensor, K: int, gen: torch.Generator) -> torch.Tensor:
    """
    k-means++-style seeding for affine means.
    X: [n, p] (device-aware)
    returns mu: [K, p]
    """
    n, p = X.shape
    mu = torch.empty((K, p), device=X.device, dtype=X.dtype)
    i0 = torch.randint(n, (1,), device=X.device, generator=gen).item()
    mu[0] = X[i0]

    d2 = ((X - mu[0])**2).sum(dim=1)  # [n]
    for k in range(1, K):
        probs = d2 / (d2.sum() + 1e-12)
        idx = torch.multinomial(probs.clamp_min(1e-12), 1, generator=gen).item()
        mu[k] = X[idx]
        d2 = torch.minimum(d2, ((X - mu[k])**2).sum(dim=1))
    return mu

def _proj_residuals_squared(X, mu, U, dmask):
    """
    Residuals:
      r^2 = ||X - mu_k||^2 - 1_{d_k=1} * ((X-mu_k)·U_k)^2
    X: [n, p]
    mu: [K, p]
    U:  [K, p]  (unit vectors; ignored when dmask[k]==0)
    dmask: [K] in {0,1} (float for broadcasting)
    returns resid: [n, K]
    """
    # base squared distances to means
    R = X.unsqueeze(1) - mu.unsqueeze(0)         # [n, K, p]
    base = (R * R).sum(dim=2)                    # [n, K]

    if U.numel() == 0:
        return base

    # SAFE & FAST: (X - mu_k) dot U_k for all n,k
    coef = torch.einsum('nkp,kp->nk', R, U)      # [n, K]
    proj2 = (coef * coef) * dmask.view(1, -1)    # zero when d_k==0

    return base - proj2

def fast_k_subspaces_quick(
    X,
    K,
    *,
    tau=0.20,           # quick rule: set d_k=1 if sigma2/sigma1 < tau else 0
    max_iter=50,
    n_init=5,
    trim_frac=0.0,      # e.g., 0.05 trims top 5% by residual
    affine=True,
    reseed_on_empty=True,
    device=None,
    dtype=torch.float32,
    verbose=False,
    seed=0,
    use_pca_lowrank=True,  # faster when nk >> p; uses torch.pca_lowrank
):
    """
    Fast k-subspaces with quick per-cluster dimension rule (d_k in {0,1}).
    - If d_k=0, cluster is a centroid (k-means-like).
    - If d_k=1, cluster is an affine line (mean + 1D PCA direction).
    """
    # Move to torch tensor & device
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=dtype)
    if device is None:
        device = X.device if X.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device=device, dtype=dtype)
    n, p = X.shape

    assert 0.0 <= float(trim_frac) < 0.5, "trim_frac should be in [0, 0.5)."
    best = None

    def _reseed(mu, U, dmask, z, resid_min, gen):
        counts = torch.bincount(z, minlength=K)
        empties = (counts == 0).nonzero(as_tuple=False).flatten()
        if empties.numel() == 0:
            return
        _, idxs = torch.topk(resid_min, k=empties.numel(), largest=True)
        for e, idx in zip(empties.tolist(), idxs.tolist()):
            mu[e] = X[idx] if affine else torch.zeros_like(mu[e])
            U[e].zero_()
            dmask[e] = 0.0

    def _pca1(Ck: torch.Tensor, gen: torch.Generator):
        """
        Returns (s1, s2, v1) where s1>=s2 are top singular values
        and v1 is the top right singular vector (feature direction).
        Ck is centered data: [nk, p].
        """
        if Ck.numel() == 0:
            return 0.0, 0.0, torch.zeros((Ck.shape[1],), device=Ck.device, dtype=Ck.dtype)

        if use_pca_lowrank:
            try:
                Uhat, Shat, Vhat = torch.pca_lowrank(Ck, q=min(2, min(*Ck.shape)))
                s1 = Shat[0].item() if Shat.numel() >= 1 else 0.0
                s2 = Shat[1].item() if Shat.numel() >= 2 else 0.0
                v1 = Vhat[:, 0] if Vhat.numel() > 0 else torch.zeros((Ck.shape[1],), device=Ck.device, dtype=Ck.dtype)
                return s1, s2, v1
            except RuntimeError:
                pass

        try:
            Uk, s, Vh = torch.linalg.svd(Ck, full_matrices=False)
        except RuntimeError:
            jitter = torch.randn_like(Ck, generator=gen) * 1e-6
            Uk, s, Vh = torch.linalg.svd(Ck + jitter, full_matrices=False)

        s1 = s[0].item() if s.numel() >= 1 else 0.0
        s2 = s[1].item() if s.numel() >= 2 else 0.0
        v1 = Vh[0] if Vh.numel() > 0 else torch.zeros((Ck.shape[1],), device=Ck.device, dtype=Ck.dtype)
        return s1, s2, v1

    def run_once(run_seed: int):
        # Device-aware generator (prevents the CUDA/CPU mismatch)
        gen = torch.Generator(device=device)
        gen.manual_seed(run_seed)

        # ----- init -----
        if affine:
            mu = _kmeanspp_init(X, K, gen)        # [K, p]
        else:
            mu = torch.zeros((K, p), device=X.device, dtype=X.dtype)

        # Start with all blobs (d_k=0); U arbitrary unit vectors
        dmask = torch.zeros(K, device=X.device, dtype=X.dtype)  # {0,1} as float for broadcast
        U = F.normalize(torch.randn((K, p), device=X.device, dtype=X.dtype, generator=gen), dim=1)

        # Warm E-step assignments
        resid = _proj_residuals_squared(X, mu, U, dmask)
        z = resid.argmin(dim=1)
        inliers = torch.ones(n, device=X.device, dtype=torch.bool)

        prev_loss = math.inf

        for it in range(max_iter):
            # ----- M-step -----
            for k in range(K):
                idx = (z == k) & inliers
                nk = int(idx.sum().item())
                if nk == 0:
                    continue

                Xk = X[idx]
                if affine:
                    muk = Xk.mean(dim=0)
                    Ck = Xk - muk
                else:
                    muk = torch.zeros_like(mu[k])
                    Ck = Xk

                s1, s2, v1 = _pca1(Ck, gen)
                if s1 > 0.0:
                    ratio = (s2 / (s1 + 1e-12)) if s2 > 0.0 else 0.0
                    dk = 1 if ratio < tau else 0
                else:
                    dk = 0

                dmask[k] = float(dk)
                mu[k] = muk
                U[k] = F.normalize(v1, dim=0) if dk == 1 else torch.zeros_like(U[k])

            # ----- E-step -----
            resid = _proj_residuals_squared(X, mu, U, dmask)
            z = resid.argmin(dim=1)
            rmin = resid.gather(1, z.view(-1, 1)).squeeze(1)

            # Trimming
            if trim_frac > 0.0:
                q = torch.quantile(rmin, 1.0 - float(trim_frac))
                inliers = rmin <= q
            else:
                inliers = torch.ones(n, device=X.device, dtype=torch.bool)

            # Reseed empties
            if reseed_on_empty:
                _reseed(mu, U, dmask, z, rmin, gen)

            # Loss on inliers
            loss = rmin[inliers].sum().item()
            if verbose:
                nz = torch.bincount(z, minlength=K).tolist()
                print(f"[iter {it:02d}] loss={loss:.6g}  inliers={int(inliers.sum())}/{n}  counts={nz}")

            # Convergence
            if prev_loss - loss < 1e-12 + (prev_loss if math.isfinite(prev_loss) else 1.0) * 1e-4:
                prev_loss = loss
                break
            prev_loss = loss

        return {
            "z": z.clone().to("cpu", torch.int64),
            "mu": mu.clone().to("cpu"),
            "U": U.clone().to("cpu"),
            "dmask": dmask.clone().to("cpu"),
            "loss": float(prev_loss),
            "inliers": inliers.clone().to("cpu"),
        }

    # Multiple restarts; keep best
    for j in range(n_init):
        res = run_once(seed + j)
        if (best is None) or (res["loss"] < best["loss"]):
            best = res

    return best


# ---------------------------
# Minimal usage example
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthetic: mix of blobs and lines in 2D
    n_per = 300
    # blob cluster
    X1 = 0.8 * torch.randn(n_per, 2, device=device)
    # line cluster
