import numpy as np
import logging
from nrc.fuzzycmeans import FuzzyCMeans
from typing import Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOISE = -1


class NoisyRingsClustering:
    def __init__(
        self,
        n_rings: int,
        q: float,
        convergence_eps: float = 0.01,
        max_iters: int = 200,
        noise_distance_threshold: float = 0.1,
        apply_noise_removal: bool = True,
        max_noise_checks: int = 20,
        init_method: Literal["fuzzycmeans", "concentric"] = "fuzzycmeans",
    ) -> None:
        self.n_rings = n_rings
        self.fitted = False
        self.q = q
        self.max_iters = max_iters
        self.convergence_eps = convergence_eps
        self.eps = 1e-10
        self.noise_distance_threshold = noise_distance_threshold
        self.max_noise_checks = max_noise_checks
        self.apply_noise_removal = apply_noise_removal
        self.init_method = init_method

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
        memberships = self.memberships
        # apply mask to the memberships
        memberships = memberships * self.noise_mask
        center_dists = self._eucledian_dist(
            samples[None, ...], self.centers[:, None, :], axis=2
        ).astype(
            np.float64
        )  # shape (n_rings, n_samples)
        return np.sum((memberships**self.q) * center_dists, axis=1) / np.sum(
            memberships**self.q + self.eps, axis=1
        )  # shape (n_rings)

    def get_new_centers(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculate the new centers of the rings

        Args:
            samples: array of shape (n_samples, n_features)

        Returns:
            array of shape (n_rings, n_features)
        """
        common = (
            self.memberships * self.noise_mask
        ) ** self.q  # shape (n_rings, n_samples)

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

        if self.init_method == "concentric":
            # in the case of concentric rings, we can initialize the centers as the mean of the samples
            self.centers = np.mean(x, axis=0)[None, ...].repeat(self.n_rings, axis=0)
            # initialize the radii as random values between min and max distance to the center
            max_dist = np.max(self._eucledian_dist(x, self.centers[0][None, ...]))
            min_dist = np.min(self._eucledian_dist(x, self.centers[0][None, ...]))
            self.radii = np.random.uniform(min_dist, max_dist, self.n_rings)
            self.memberships = np.random.rand(self.n_rings, x.shape[0])
            self.memberships = self.memberships / np.sum(
                self.memberships, axis=0
            )  # normalize the memberships
            self.noise_mask = np.ones_like(self.memberships, dtype=np.int32)
        elif self.init_method == "fuzzycmeans":
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

            self.noise_mask = np.ones_like(self.memberships, dtype=np.int32)
            # only take into account distances to the center of the cluster
            self.radii = self.get_new_radii(x)
        else:
            raise ValueError(
                "Invalid datadist value. Expected 'concentric' or 'fuzzycmeans', got {}".format(
                    self.init_method
                )
            )

    def fit(self, x: np.ndarray) -> None:
        assert x.ndim == 2, "Input data must be 2D, got shape {}".format(x.shape)
        self.x = x
        self.fitted = True

        self._initialize(x)
        noise_checks = 0
        last_noise_mask = np.zeros_like(self.noise_mask)
        for it in range(self.max_iters):
            self.last_iteration = it
            old_memberships = self.memberships
            self.memberships = self.get_new_memberships(x)
            radii, centers = self.get_new_radii(x), self.get_new_centers(x)
            self.radii, self.centers = radii, centers

            if self._convergence_criterion(self.memberships, old_memberships):
                noise_mask = self.get_noise_mask(x)
                if self.max_noise_checks > noise_checks and not np.allclose(
                    noise_mask, last_noise_mask
                ):
                    self.noise_mask = self.get_noise_mask(x)
                    logger.info(
                        "[NoisyRingsClustering] Converged partly after {} iterations. Recomputing noise mask and continuing. Total noise samples are {}".format(
                            it, np.sum(self.noise_mask == 0) / self.noise_mask.shape[0]
                        )
                    )
                    noise_checks += 1
                    last_noise_mask = self.noise_mask
                else:
                    logger.info(
                        "[NoisyRingsClustering] Converged after {} iterations. Stopping early.".format(
                            it
                        )
                    )
                    break

    def get_noise_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Mask where 1 means no noise and 0 means noise
        """
        # a point is considered noise if it's distance to all rings is

        if not self.apply_noise_removal:
            return np.ones_like(self.memberships)

        ringdists = self._dist_to_rings(
            x, self.centers, self.radii
        )  # shape (n_rings, n_samples)
        mask = np.logical_not(
            np.all(ringdists > (self.noise_distance_threshold), axis=0)
        ).astype(
            np.int32
        )  # shape (n_samples)

        mask = np.broadcast_to(
            mask, self.memberships.shape
        )  # shape (n_rings, n_samples)
        return mask

    def get_labels(self, include_mask: bool = True) -> np.ndarray:
        """
        Get the hard labels of the samples

        Returns:
            radii: array of shape (n_samples)
            centers: array of shape (n_samples, n_features)
            memberships: array of shape (n_samples) with the index of the cluster for each sample
        """
        hard_memberships = np.argmax(self.memberships, axis=0)
        # first, since noise mask is repeated across the axis 0, we can just take the first row
        noise_mask = self.noise_mask[0]
        for i in range(self.noise_mask.shape[0]):
            assert np.allclose(noise_mask, self.noise_mask[i])
        hard_memberships[noise_mask == 0] = NOISE
        logger.info("Total noise samples: {}".format(np.sum(noise_mask == 0)))
        return self.radii, self.centers, hard_memberships
