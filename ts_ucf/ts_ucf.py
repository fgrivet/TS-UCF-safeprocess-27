"""TS-UCF algorithm"""

import logging
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin

from . import core

logger = logging.getLogger(__name__)


class TS_UCF(OutlierMixin, BaseEstimator):
    """This version of TS-UCF consider that signals are already normalized to [-1, 1]
    and resampled to the same time grid as it takes a numpy array in input.
    The windowing step is performed in this class if it has not been done previously."""

    def __init__(self, n_min: int = 1, n_max: int = 20, L: int | Literal["auto"] = "auto", s: int | None = None, keep_internals=False) -> None:
        if n_min <= 0:
            raise ValueError(f"n_min should be positive. Received {n_min}.")
        if n_max < n_min:
            raise ValueError(f"n_max should greater than or equal to n_min={n_min}. Received {n_max}.")
        if L != "auto" and L <= 0:
            raise ValueError(f"L should be positive or 'auto'. Received {L}.")
        if s is not None and (s <= 0 or (L != "auto" and s > L)):
            raise ValueError(f"s should be positive and lower than or equal to L={self.L} or None. Received {s}.")

        self.n_min = n_min
        self.n_max = n_max
        self.n_val = n_max - n_min + 1
        self.L = L
        self.s = s
        self.keep_internals = keep_internals

        self.windows_train: None | np.ndarray = None
        self.k: None | int = None  # Number of channels
        if keep_internals:
            self.distances: None | np.ndarray = None

    def check_X_train(self, X: np.ndarray) -> np.ndarray:
        """X is either (n_samples, n_channel) or (n_windows, window_size=L, n_channel)
        The output is (n_windows, window_size=L, n_channel)"""
        if X.ndim == 2:
            if self.L == "auto":
                self.L = core.infer_window_size(X)
            windows = core.make_windows(X, self.L, self.s)

        elif X.ndim == 3:
            L = X.shape[1]
            if self.L == "auto":
                self.L = L
                logger.info("L was 'auto' and is now set to %d.", L)
            elif self.L != L:
                raise ValueError(f"Incompatible window sizes. Should be {self.L}. Received {L}.")
            windows = X

        else:
            raise ValueError(
                f"X is either size (n_samples, n_channel) or size (n_windows, window_size=L={self.L}, n_channel). Received size {X.shape}."
            )

        if self.s is None:
            self.s = self.L
        if self.s > self.L:
            raise ValueError(f"s should be positive and lower than or equal to L={self.L}. Received {self.s}.")

        return windows

    def check_X_test(self, X: np.ndarray) -> np.ndarray:
        assert self.L != "auto"
        assert self.k is not None

        if X.shape[-1] != self.k:
            raise ValueError(
                f"X is either size (n_samples, n_channel={self.k}) "
                f"or size (n_windows, window_size=L={self.L}, n_channel={self.k}). Received size {X.shape}."
            )

        if X.ndim == 2:
            windows = core.make_windows(X, self.L, self.s)

        elif X.ndim == 3:
            L = X.shape[1]
            if self.L != L:
                raise ValueError(f"Incompatible window sizes. Should be {self.L}. Received {L}.")
            windows = X

        else:
            raise ValueError(
                f"X is either size (n_samples, n_channel={self.k}) "
                f"or size (n_windows, window_size=L={self.L}, n_channel={self.k}). Received size {X.shape}."
            )

        return windows

    def fit(self, X: np.ndarray, y=None) -> "TS_UCF":
        self.windows_train = self.check_X_train(X)
        self.k = self.windows_train.shape[2]

        assert self.L != "auto"
        assert self.windows_train is not None
        assert self.k is not None

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.windows_train is not None
        assert self.k is not None

        windows_test = self.check_X_test(X)

        K_train = len(self.windows_train)
        K_test = len(windows_test)
        multi_distances = np.zeros((K_test, K_train))
        scores = np.zeros((self.k + 1, K_test, self.n_val))  # n_channel+1 (multivariate too), n_windows, n_val

        if self.keep_internals:
            self.distances = np.zeros((self.k + 1, K_test, K_train))  # n_channel+1 (multivariate too), n_windows_test, n_windows_train

        # Univariate analysis
        for n_channel in range(self.k):
            distances = core.pairwise_dist_rfft(windows_test[:, :, n_channel], self.windows_train[:, :, n_channel])
            scores[n_channel] = core.ucf_score(distances, self.n_min, self.n_max)
            multi_distances += distances

            if self.keep_internals:
                self.distances[n_channel] = distances  # type: ignore

        # Multivariate analysis
        scores[self.k] = core.ucf_score(multi_distances, self.n_min, self.n_max)

        if self.keep_internals:
            self.distances[self.k] = multi_distances  # type: ignore

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)

        # To be improved by detecting the exponential growth pattern of the scores. For now, these are temporary variables
        predictions = np.where(scores[:, :, 1] > 10, -1, 1)
        return predictions

    def impact_scores(self, sigma: float):
        if not self.keep_internals:
            raise ValueError("keep_internals should be True to compute impact_scores.")
        if self.distances is None:
            raise ValueError("Use TS_UCF.decision_function or TS_UCF.predict with keep_internals=True before.")
        if sigma > 1 or sigma < 0:
            raise ValueError(f"Sigma should be in [0, 1]. Received {sigma}.")

        return core.impact_scores(self.distances, sigma)
