import numpy as np
import logging
from nrc.fuzzy_c_means import FuzzyCMeans
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
        """
        Noisy Rings Clustering algorithm

        Args:
            n_rings: the number of rings to cluster the data into
            q: the fuzziness parameter, must be greater than 1. Lower means less fuzziness.
            convergence_eps: the maximum difference between the memberships of two consecutive iterations to consider the algorithm has converged.
            max_iters: the maximum number of iterations to run the algorithm for.
            noise_distance_threshold: the maximum distance to consider a point as noise. Only used if apply_noise_removal is True.
            apply_noise_removal: whether to apply noise removal or not.
            max_noise_checks: the maximum number of times to recompute the noise mask after convergence.
            init_method: the method to initialize the centers and radii of the rings. Can be 'fuzzycmeans' or 'concentric'.
                'fuzzycmeans' initializes the centers and memberships using the Fuzzy C-Means algorithm.
                'concentric' initializes the centers as the mean of the samples and the radii as random values between the minimum and maximum distance to the center.
        """
        assert q > 1, "q must be greater than 1, got {}".format(q)
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

    def _euclidean_dist(
        self, x: np.ndarray, y: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        """
        Computes the Euclidean distances between two arrays of points

        Args:
            x: ndarray of shape (..., n_features)
            y: ndarray of shape (..., n_features)

        Returns:
            array of shape (...,) denoting the Euclidean distance between the points
        """
        return np.sqrt(np.sum((x - y) ** 2, axis=axis, keepdims=False))

    def _dist_to_rings(
        self, x: np.ndarray, centers: np.ndarray, radius: np.ndarray
    ) -> np.ndarray:
        """
        Computes the distance between the points and the rings.
        The distance is the absolute difference between the Euclidean distance of the points to the center and the radius of the ring.

        Args:
            x: ndarray of shape (n_samples, n_features)
            centers: ndarray of shape (n_rings, n_features)
            radius: ndarray of shape (n_rings)

        Returns:
            ndarray of shape (n_rings, n_samples)
        """
        # pass centers of shape (n_samples, n_features) to (n_rings, 1, n_features)
        # that'll be broadcasted to (n_rings, n_samples, n_features)
        # and reduced over the last axis (features axis)
        # so we'll get a matrix of shape (n_rings, n_samples) with the distances of each center to each sample
        dists = self._euclidean_dist(x[None, ...], centers[:, None, :], axis=2)

        # returns the absolute difference between the distances and the radius
        return np.abs(dists - radius[:, None])  # shape (n_rings, n_samples)

    def _get_new_memberships(self, samples: np.ndarray) -> np.ndarray:
        """
        Computes the new memberships of the samples to the rings, according to the formula.

        Args:
            samples: ndarray of shape (n_samples, n_features)

        Returns:
            ndarray of shape (n_rings, n_samples)
        """

        ring_dists = (
            self._dist_to_rings(samples, self.centers, self.radii) ** (2 / (self.q - 1))
            + self.eps
        )  # shape (n_rings, n_samples)
        # shapes (1, n_rings, n_samples) / (n_rings, 1, n_samples) -> (n_rings, n_rings, n_samples)
        term = ring_dists[None, :, :] / ring_dists[:, None, :]
        mem = 1 / np.sum(term, axis=0)
        return mem

    def _get_new_radii(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute the new radii of the rings, according to the formula.

        Args:
            samples: array of shape (n_samples, n_features)

        Returns:
            array of shape (n_rings) denoting the new radii of the rings
        """
        memberships = self.memberships
        # apply mask to the memberships, so that noise samples are not taken into account for the new radii computation
        memberships = memberships * self.noise_mask
        center_dists = self._euclidean_dist(
            samples[None, ...], self.centers[:, None, :], axis=2
        )  # shape (n_rings, n_samples)
        # compute the new radii according to the formula, not taking into account the noise samples
        return np.sum((memberships**self.q) * center_dists, axis=1) / np.sum(
            memberships**self.q + self.eps, axis=1
        )  # shape (n_rings)

    def _get_new_centers(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculate the new centers of the rings, according to the formula.

        Args:
            samples: array of shape (n_samples, n_features)

        Returns:
            array of shape (n_rings, n_features)
        """
        common = (
            self.memberships * self.noise_mask
        ) ** self.q  # shape (n_rings, n_samples)

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
        Checks if the memberships have converged.

        Args:
            old_memberships: array of shape (n_rings, n_samples)
            new_memberships: array of shape (n_rings, n_samples)
        """
        return np.allclose(old_memberships, new_memberships, atol=self.convergence_eps)

    def _initialize(self, x: np.ndarray) -> None:
        """
        Initialize the centers, radii and memberships of the rings.

        Args:
            x: array of shape (n_samples, n_features)
        """
        # use kmeans to initialize the centers

        if self.init_method == "concentric":
            # in the case of concentric rings, we can initialize the centers as the mean of the samples
            self.centers = np.mean(x, axis=0)[None, ...].repeat(self.n_rings, axis=0)
            # initialize the radii as random values between min and max distance to the center
            max_dist = np.max(self._euclidean_dist(x, self.centers[0][None, ...]))
            min_dist = np.min(self._euclidean_dist(x, self.centers[0][None, ...]))
            self.radii = np.random.uniform(min_dist, max_dist, self.n_rings)
            self.memberships = np.random.rand(self.n_rings, x.shape[0])
            self.memberships = self.memberships / np.sum(
                self.memberships, axis=0
            )  # normalize the memberships
            self.noise_mask = np.ones_like(self.memberships, dtype=np.int32)
        elif self.init_method == "fuzzycmeans":
            kmeans = FuzzyCMeans(n_clusters=self.n_rings, max_iter=3000, m=1.1)
            kmeans.fit(x)

            self.centers = kmeans.centroids  # shape (n_rings, n_features)

            # initialize the radii as the average distance of the samples to the centers
            self.memberships = kmeans.membership  # shape (n_rings, n_samples)

            self.noise_mask = np.ones_like(self.memberships, dtype=np.int32)
            # only take into account distances to the center of the cluster
            self.radii = self._get_new_radii(x)
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
            # update the memberships, radii and centers
            self.last_iter = it
            old_memberships = self.memberships
            self.memberships = self._get_new_memberships(x)
            self.radii, self.centers = self._get_new_radii(x), self._get_new_centers(x)

            # check for convergence on memberships
            if self._convergence_criterion(self.memberships, old_memberships):
                if not self.apply_noise_removal:
                    logger.info(
                        "[NoisyRingsClustering] Converged after {} iterations. Stopping early.".format(
                            it
                        )
                    )
                    break
                noise_mask = self._get_noise_mask(x)
                # check if the noise mask has changed and we haven't reached the maximum number of noise checks. If so, recompute and continue
                if self.max_noise_checks > noise_checks and not np.allclose(
                    noise_mask, last_noise_mask
                ):
                    self.noise_mask = noise_mask
                    logger.info(
                        "[NoisyRingsClustering] Converged partly after {} iterations. Recomputing noise mask and continuing. Total noise samples are {}".format(
                            it, np.sum(self.noise_mask == 0) / self.noise_mask.shape[0]
                        )
                    )
                    noise_checks += 1
                    last_noise_mask = self.noise_mask
                # if the noise mask hasn't changed, we can stop
                else:
                    logger.info(
                        "[NoisyRingsClustering] Converged after {} iterations. Stopping early.".format(
                            it
                        )
                    )
                    break

    def _get_noise_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Mask where 1 means no noise and 0 means noise

        Args:
            x: array of shape (n_samples, n_features)

        Returns:
            array of shape (n_rings, n_samples)
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

    def get_labels(self) -> np.ndarray:
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
        hard_memberships[noise_mask == 0] = NOISE
        logger.info("Total noise samples: {}".format(np.sum(noise_mask == 0)))
        return self.radii, self.centers, hard_memberships
