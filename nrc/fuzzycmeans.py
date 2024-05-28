import numpy as np
import logging

logger = logging.getLogger(__name__)
class FuzzyCMeans:
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the Euclidean distance between two vectors.
        Args:
            x1: (n_features,) ndarray
            x2: (n_features,) ndarray

        Returns:
            float: the Euclidean distance between x1 and x2
        """
        return np.sqrt(
            np.sum((x1 - x2) ** 2, axis=-1)
        )  # sum over the features, not the samples

    _dist = _euclidean_distance

    def __init__(
        self, n_clusters: int = 3, max_iter: int = 100, m: float = 2, eps: float = 0.01
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.fitted = False
        self.eps = eps

    def _init_membership(self, n_samples: int) -> np.ndarray:
        """
        Initialize the membership matrix.
        Args:
            n_samples: int, the number of samples

        Returns:
            ndarray: the membership matrix of shape (n_clusters, n_samples)
        """
        r = np.random.rand(self.n_clusters, n_samples)
        return r / r.sum(
            0
        )  # normalize the rows to sum to 1 (sum accross a column is 1)

    def _compute_centroids(self, X: np.ndarray, membership: np.ndarray) -> np.ndarray:
        """
        Compute the centroids of the clusters.
        Args:
            X: (n_samples, n_features) ndarray
            membership: (n_clusters, n_samples) ndarray, the membership matrix

        Returns:
            ndarray: the centroids of the clusters
        """
        return np.dot(membership**self.m, X) / np.sum(
            membership**self.m, axis=1, keepdims=True
        )

    def _update_membership(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Update the membership matrix.
        Args:
            X: (n_samples, n_features) ndarray
            centroids: (n_clusters, n_features) ndarray

        Returns:
            ndarray: the updated membership matrix
        """
        distances = np.array(
            [self._dist(X, c) for c in centroids]
        )  # shape (n_clusters, n_samples)
        dsum_per_cluster = (
            distances.sum(0, keepdims=True) + 1e-8
        )  # avoid div by zero, shape (1, n_samples)
        normalized_dists = (
            distances / dsum_per_cluster + 1e-8
        )  # avoid div by zero, shape (n_clusters, n_samples)
        return 1 / (normalized_dists ** (2 / (self.m - 1)))

    def fit(self, X: np.ndarray):
        """
        Fit the model to the data matrix X.
        Args:
            X: (n_samples, n_features) ndarray

        Returns:
            self
        """
        assert (
            len(X.shape) == 2
        ), "X must be a 2D array with shape (n_samples, n_features), got shape {}".format(
            X.shape
        )
        if self.fitted:
            logging.warning(
                "The model has already been fitted. Re-fitting will overwrite the previous model."
            )

        self.fitted = True
        n_samples = X.shape[0]

        # Initialize the centroids and labels
        self.membership = self._init_membership(n_samples)
        self.centroids = self._compute_centroids(X, self.membership)

        for _ in range(self.max_iter):
            self.centroids = self._compute_centroids(X, self.membership)
            new_membership = self._update_membership(X, self.centroids)
            if np.allclose(new_membership, self.membership, atol=self.eps):
                logger.info("[FuzzyCMeans] Converged after {} iterations. Stopping early.".format(_))
                self.centroids = self._compute_centroids(
                    X, new_membership
                )  # update centroids one last time
                break
            self.membership = new_membership

        return self

    def get_hard_labels(self):
        """
        Get the hard labels of the data.
        Returns:
            ndarray: the hard labels of the data
        """
        assert self.fitted, "The model has not been fitted yet."
        return np.argmax(
            self.membership, axis=0
        )  # get the index of the maximum value along the rows
