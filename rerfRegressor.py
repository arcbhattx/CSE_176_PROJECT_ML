import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import r2_score

from rerf.rerfClassifier import rerfClassifier


class rerfRegressor(BaseEstimator, RegressorMixin):
    """
    Regression wrapper around rerfClassifier that discretizes the target into bins,
    fits a classifier, and returns the expected value under the predicted class probabilities.

    Parameters
    ----------
    n_bins : int (default=50)
        Number of bins to discretize the continuous target into.
    binning : {'quantile', 'uniform'} (default='quantile')
        How to place bin edges: 'quantile' places bins by quantiles, 'uniform' by equal-width.
    random_state : int or None
        Random seed for reproducibility (passed to rerfClassifier if not provided there).
    classifier params : other keyword args are passed to rerfClassifier (e.g., n_estimators).
    """

    def __init__(self, n_bins=50, binning="quantile", random_state=None, **classifier_params):
        self.n_bins = int(n_bins)
        self.binning = binning
        self.random_state = random_state
        self.classifier_params = classifier_params

    def _make_bins(self, y):
        if self.n_bins <= 1:
            raise ValueError("n_bins must be >= 2")
        if self.binning == "quantile":
            edges = np.quantile(y, np.linspace(0.0, 1.0, self.n_bins + 1))
        elif self.binning == "uniform":
            edges = np.linspace(np.min(y), np.max(y), self.n_bins + 1)
        else:
            raise ValueError("Unknown binning: choose 'quantile' or 'uniform'")

        edges = np.unique(edges)
        if edges.size < 2:
            raise ValueError("Not enough distinct bin edges; try fewer bins or different binning")
        return edges

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = np.asarray(y).ravel().astype(float)

        # create bins
        self.bin_edges_ = self._make_bins(y)
        if self.bin_edges_.size <= 2:
            # Degenerate: single value
            self._is_constant_ = True
            self.y_mean_ = y.mean()
            self.clf_ = None
            self.n_bins_ = 1
            self.bin_centers_ = np.array([self.y_mean_])
            return self

        labels = np.digitize(y, self.bin_edges_[1:-1], right=True)
        self.n_bins_ = int(self.bin_edges_.size - 1)

        # compute representative bin values (use mean inside each bin when present)
        bin_centers = np.empty(self.n_bins_, dtype=float)
        for i in range(self.n_bins_):
            members = y[labels == i]
            if members.size > 0:
                bin_centers[i] = members.mean()
            else:
                left = self.bin_edges_[i]
                right = self.bin_edges_[i + 1]
                bin_centers[i] = 0.5 * (left + right)
        self.bin_centers_ = bin_centers

        # prepare classifier kwargs
        clf_kwargs = dict(self.classifier_params)
        if "random_state" not in clf_kwargs:
            clf_kwargs["random_state"] = self.random_state

        self.clf_ = rerfClassifier(**clf_kwargs)
        self.clf_.fit(X, labels)

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(labels)
        return self

    def predict(self, X):
        X = check_array(X)
        if getattr(self, "_is_constant_", False):
            return np.full(shape=(X.shape[0],), fill_value=self.y_mean_, dtype=float)

        proba = np.asarray(self.clf_.predict_proba(X))
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)

        # ensure proba columns align with bin_centers
        if proba.shape[1] != self.bin_centers_.shape[0]:
            full_proba = np.zeros((proba.shape[0], self.bin_centers_.shape[0]), dtype=float)
            for idx, cls in enumerate(self.clf_.classes_):
                full_proba[:, int(cls)] = proba[:, idx]
            proba = full_proba

        preds = proba.dot(self.bin_centers_)
        return preds

    def score(self, X, y, sample_weight=None):
        y_true = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred, sample_weight=sample_weight)
