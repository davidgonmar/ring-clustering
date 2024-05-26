import numpy as np
import logging
from nrc.fuzzycmeans import FuzzyCMeans

logging.basicConfig(level=logging.INFO)


class NoisyRingsClustering:
    def __init__(
        self,
        n_rings: int,
        q: float,
        convergence_eps: float = 0.01,
        max_iters: int = 200,
    ) -> None:
        self.n_rings = n_rings
        self.fitted = False
        self.q = q
        self.max_iters = max_iters
        self.convergence_eps = convergence_eps
        self.eps = 1e-10

    def _eucledian_dist(
        self, x: np.ndarray, y: np.ndarray, axis: int = 1
    ) -> np.ndarray:
        """
        Calculate the Euclidean distance between two arrays of points
        """
        if x.ndim != y.ndim:
            raise ValueError(
                "Shapes of x and y must be the same, got {} and {}".format(
                    x.shape, y.shape
                )
            )
        return np.sqrt(np.sum((x - y) ** 2, axis=axis, keepdims=False))

    def _dist_to_rings(
        self, x: np.ndarray, centers: np.ndarray, radius: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the distance of the points to the ring.
        The distance is the absolute difference between the Euclidean distance of the points to the center and the radius of the ring.

        Args:
            x: array of shape (n_samples, n_features)
            centers: array of shape (n_rings, n_features)
            radius: array of shape (n_rings)

        Returns:
            array of shape (n_samples)
        """
        # pass centers of shape (n_samples, n_features) to (n_rings, 1, n_features)
        # that'll be broadcasted to (n_rings, n_samples, n_features)
        # and reduced over the last axis (features axis)
        # so we'll get a matrix of shape (n_rings, n_samples) with the distances of each center to each sample
        dists = self._eucledian_dist(
            x[None, ...], centers[:, None, :], axis=2
        )  # shape (n_rings, n_samples)

        # returns the absolute difference between the distances and the radius
        return np.abs(dists - radius[:, None])  # shape (n_rings, n_samples)

    def get_new_memberships(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculate the new memberships of the samples to the rings

        Args:
            samples: array of shape (n_samples, n_features)

        Returns:
            array of shape (n_rings, n_samples)
        """

        ring_dists = (
            self._dist_to_rings(samples, self.centers, self.radii) + 1e-10
        )  # shape (n_rings, n_samples)

        # sum over the clusters, to compute, for each sample, the sum of the distances to each cluster
        div = np.sum(ring_dists, axis=0, keepdims=True)  # shape (1, n_samples)

        mem = ((ring_dists) ** (-1 / (self.q - 1))) / (
            (div ** (-1 / (self.q - 1)))
        )  # shape (n_rings, n_samples)

        x = mem / (
            np.sum(mem, axis=0, keepdims=True) + 1e-10
        )  # shape (n_rings, n_samples)

        assert x.sum(axis=0).all() == 1

        return x

    def get_new_radii(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculate the new radii of the rings

        Args:
            samples: array of shape (n_samples, n_features)

        Returns:
            array of shape (n_rings)
        """
        # warning! not ring distances, but distances to the centers !!!
        # self.memberships shape is (n_rings, n_samples)
        assert np.sum(self.memberships, axis=0).all() == 1
        center_dists = self._eucledian_dist(
            samples[None, ...], self.centers[:, None, :], axis=2
        ).astype(
            np.float64
        )  # shape (n_rings, n_samples)
        return np.sum((self.memberships**self.q) * center_dists, axis=1) / np.sum(
            self.memberships**self.q, axis=1
        )  # shape (n_rings)

    def get_new_centers(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculate the new centers of the rings

        Args:
            samples: array of shape (n_samples, n_features)

        Returns:
            array of shape (n_rings, n_features)
        """
        common = self.memberships**self.q  # shape (n_rings, n_samples)

        # abuse broadcasting :D
        return np.sum(
            common[..., None] * samples[None, ...],
            axis=1,
        ) / (
            np.sum(common, axis=1, keepdims=True) + self.eps
        )  # shape (n_rings, n_features)

    def _convergence_criterion(
        self, old_memberships: np.ndarray, new_memberships: np.ndarray
    ) -> bool:
        """
        Check if the memberships have converged
        """
        return np.allclose(old_memberships, new_memberships, atol=self.convergence_eps)

    def _initialize(self, x: np.ndarray) -> None:
        """
        Initialize the centers, radii and memberships of the rings

        Args:
            x: array of shape (n_samples, n_features)
        """
        # use kmeans to initialize the centers

        kmeans = FuzzyCMeans(n_clusters=self.n_rings, max_iter=300, m=1.2, eps=0.01)
        kmeans.fit(x)

        self.centers = kmeans.centroids.astype(
            np.float64
        )  # shape (n_rings, n_features)

        # initialize the radii as the average distance of the samples to the centers
        # (1, n_samples, n_features) x (n_rings, 1, n_features) -> (n_rings, n_samples, n_features)

        self.memberships = kmeans.membership.astype(
            np.float64
        )  # shape (n_rings, n_samples)

        # only take into account distances to the center of the cluster
        self.radii = self.get_new_radii(x)

    def fit(self, x: np.ndarray) -> None:
        assert x.ndim == 2, "Input data must be 2D, got shape {}".format(x.shape)
        n_samples, n_features = x.shape
        self.x = x
        self.fitted = True

        self._initialize(x)

        for it in range(self.max_iters):
            self.radii = self.get_new_radii(x)
            old_memberships = self.memberships
            self.memberships = self.get_new_memberships(x)

            self.centers = self.get_new_centers(x)

            if self._convergence_criterion(self.memberships, old_memberships):
                logging.info(
                    "Converged after {} iterations. Stopping early.".format(it)
                )
                break

    def prune_noise(
        self, threshold: float, reestimate_centers_and_radii: bool = True
    ) -> np.ndarray:
        """
        Prune the noise from the clusters

        Args:
            threshold: float, the threshold to prune the noise
        """
        # get the maximum membership for each sample

        # prune ones with distance to radius > averagedist + threshold

        ringdists = self._dist_to_rings(
            self.x, self.centers, self.radii
        )  # shape (n_rings, n_samples)
        avg_ringdist_per_cluster = np.mean(ringdists, axis=1)
        mask = np.ones_like(
            ringdists[1], dtype=bool
        )  # indicates which samples are not noise with a 1

        # noise -> distance to the center is larger than the (average distance to ring * threshold)
        for i in range(self.n_rings):
            # only take into account samples belonging to the cluster
            samplemasks = np.argmax(self.memberships, axis=0) == i
            mask = np.logical_and(
                mask, ringdists[i] < avg_ringdist_per_cluster[i] * threshold
            )
            mask = np.logical_or(mask, samplemasks)

        self.x = self.x[mask]

        self.memberships[:, np.logical_not(mask)] = -1
        if reestimate_centers_and_radii:
            old_memberships = self.memberships
            self.memberships = self.memberships[:, mask]
            self.centers = self.get_new_centers(self.x)
            self.radii = self.get_new_radii(self.x)

            self.memberships = old_memberships

        return mask

    def get_labels(self) -> np.ndarray:
        """
        Get the hard labels of the samples

        Returns:
            radii: array of shape (n_samples)
            centers: array of shape (n_samples, n_features)
            memberships: array of shape (n_samples) with the index of the cluster for each sample
        """
        return self.radii, self.centers, self.memberships
