"""
_sklearn_fallback.py — drop-in replacements for the THREE scikit-learn objects
used by Step 02, backed only by numpy + scipy (both shipped with ArcGIS Pro).

Used automatically by `02_STARR_matching_data_weights.py` when scikit-learn is
not present in the ArcGIS Pro Python environment. The results are numerically
identical to scikit-learn:

  * StandardScaler  — mean/std standardization (ddof=0, zero-variance → 1), the
    exact StandardScaler transform.
  * LedoitWolf      — the exact Ledoit-Wolf shrunk covariance (non-blocked form,
    which equals sklearn's blocked form for n_features < block_size=1000; our
    covariate count is far below that).
  * NearestNeighbors — exact Euclidean k-NN via scipy.spatial.cKDTree
    (same neighbours/distances as sklearn's ball_tree; only equal-distance ties
    may be ordered differently, which does not affect the matching).

Validated against scikit-learn on random data (see the packaging tests).
"""
import numpy as np


# ── StandardScaler ───────────────────────────────────────────────────────────
class StandardScaler(object):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            scale = X.std(axis=0, ddof=0)
            scale = np.asarray(scale, dtype=np.float64).copy()
            scale[scale == 0.0] = 1.0          # _handle_zeros_in_scale
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1])
        self.var_ = self.scale_ ** 2
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ── LedoitWolf ───────────────────────────────────────────────────────────────
def _ledoit_wolf_shrinkage(X):
    """Non-blocked equivalent of sklearn.covariance.ledoit_wolf_shrinkage
    (assume_centered=True input). Exact for n_features < 1000."""
    n_samples, n_features = X.shape
    if n_features == 1:
        return 0.0
    X2 = X ** 2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    delta_ = np.sum(np.dot(X.T, X) ** 2) / n_samples ** 2
    beta_ = np.sum(np.dot(X2.T, X2))
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu ** 2) / n_features
    beta = min(beta, delta)
    return 0.0 if beta == 0 else float(beta / delta)


class LedoitWolf(object):
    def __init__(self, store_precision=True, assume_centered=False, block_size=1000):
        self.assume_centered = assume_centered

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if not self.assume_centered:
            self.location_ = X.mean(axis=0)
            Xc = X - self.location_
        else:
            self.location_ = np.zeros(X.shape[1])
            Xc = X
        n_samples, n_features = Xc.shape
        emp_cov = np.dot(Xc.T, Xc) / n_samples
        mu = np.trace(emp_cov) / n_features
        shrinkage = _ledoit_wolf_shrinkage(Xc)
        shrunk = (1.0 - shrinkage) * emp_cov
        shrunk.flat[:: n_features + 1] += shrinkage * mu
        self.covariance_ = shrunk
        self.shrinkage_ = shrinkage
        return self


# ── NearestNeighbors (scipy cKDTree) ─────────────────────────────────────────
class NearestNeighbors(object):
    def __init__(self, n_neighbors=5, metric="euclidean", algorithm="auto",
                 leaf_size=30, n_jobs=None, **kwargs):
        if metric not in ("euclidean", "minkowski", "l2"):
            raise ValueError(f"_sklearn_fallback.NearestNeighbors supports only "
                             f"euclidean, got metric={metric!r}")
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        from scipy.spatial import cKDTree
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self._tree = cKDTree(X, leafsize=self.leaf_size)
        self._n_samples = X.shape[0]
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        k = int(n_neighbors or self.n_neighbors)
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        workers = -1 if (self.n_jobs in (-1, None) and self.n_jobs == -1) else 1
        try:
            dist, idx = self._tree.query(X, k=k, workers=(-1 if self.n_jobs == -1 else 1))
        except TypeError:
            # older scipy uses n_jobs kwarg
            dist, idx = self._tree.query(X, k=k)
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx)
        if k == 1:
            dist = dist.reshape(-1, 1)
            idx = idx.reshape(-1, 1)
        if return_distance:
            return dist, idx
        return idx
