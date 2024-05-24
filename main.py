import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


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
                logging.info("Converged after {} iterations. Stopping early.".format(_))
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
        n_samples = samples.shape[0]
        n_features = samples.shape[1]
        center_dists = self._eucledian_dist(
            samples[None, ...], self.centers[:, None, :], axis=2
        )  # shape (n_rings, n_samples)
        radii = self.radii  # shape (n_rings)
        alpha = np.maximum(1 - radii[..., None] / (
            center_dists + self.eps
        ), radii[:, None])  # shape (n_rings, n_samples)

        print("alpha", alpha.shape, alpha)
        common = (self.memberships**self.q) * alpha  # shape (n_rings, n_samples)

        return np.sum(
            common.reshape(self.n_rings, n_samples, 1)
            * samples.reshape(1, n_samples, n_features),
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

        kmeans = FuzzyCMeans(n_clusters=self.n_rings, max_iter=500, m=2, eps=0.01)
        kmeans.fit(x)

        self.centers = kmeans.centroids  # shape (n_rings, n_features)

        # initialize the radii as the average distance of the samples to the centers
        # (1, n_samples, n_features) x (n_rings, 1, n_features) -> (n_rings, n_samples, n_features)

        # initialize the memberships as hard ones from the kmeans
        self.memberships = kmeans.membership

        dist = self._eucledian_dist(
            x[None, ...], self.centers[:, None, :], axis=2
        )  # shape (n_rings, n_samples)
        # only take into account distances to the center of the cluster
        self.radii = np.mean(dist, axis=1)  # shape (n_rings)
        print(dist, self.radii, self.centers)

    def fit(self, x: np.ndarray) -> None:
        assert x.ndim == 2, "Input data must be 2D, got shape {}".format(x.shape)
        n_samples, n_features = x.shape
        self.x = x
        self.fitted = True

        self._initialize(x)

        for it in range(self.max_iters):
            new_memberships = self.get_new_memberships(x)

            if self._convergence_criterion(self.memberships, new_memberships):
                self.memberships = new_memberships
                logging.info(
                    "Converged after {} iterations. Stopping early.".format(it)
                )
                break

            self.memberships = new_memberships
            self.radii = self.get_new_radii(x)
            self.centers = self.get_new_centers(x)

    def get_hard_labels(self) -> np.ndarray:
        """
        Get the hard labels of the samples

        Returns:
            radii: array of shape (n_samples)
            centers: array of shape (n_samples, n_features)
            memberships: array of shape (n_samples) with the index of the cluster for each sample
        """
        return self.radii, self.centers, np.argmax(self.memberships, axis=0)


if __name__ == "__main__":
    while True:
        def ds1():
            # create a ring dataset
            n_samples = 1000
            n_features = 2
            n_clusters = 2
            point = []
            POSRAND = 10

            for i in range(n_clusters):
                radius = 10 + np.random.uniform(-1, 1)
                theta = np.linspace(0, 2 * np.pi, n_samples // n_clusters)
                x = radius * np.cos(theta) + np.random.normal(
                    0, 0.4, size=(n_samples // n_clusters)
                )
                y = radius * np.sin(theta) + np.random.normal(
                    0, 0.4, size=(n_samples // n_clusters)
                )
                position = np.random.uniform(-POSRAND, POSRAND, size=(2))
                x += position[0]
                y += position[1]
                point.append(np.stack([x, y], axis=1))

            point = np.array(point)

            # merge dim 0 and 1
            point = point.reshape(-1, n_features)

            return point

        def ds2():
            # just a point on [10, 10] with some noise and radius 3
            n_samples = 1000
            n_features = 2
            n_clusters = 1
            point = []
            radius = 3
            theta = np.linspace(0, 2 * np.pi, n_samples)
            x = radius * np.cos(theta) + np.random.normal(0, 0.1, size=(n_samples))
            y = radius * np.sin(theta) + np.random.normal(0, 0.1, size=(n_samples))
            position = np.array([5, 5])
            x += position[0]
            y += position[1]
            point.append(np.stack([x, y], axis=1))
            point = np.array(point)

            # merge dim 0 and 1
            point = point.reshape(-1, n_features)

            return point

        point = ds1()

        n_clusters = 2
        ring_clustering = NoisyRingsClustering(
            n_clusters, q=1.2, convergence_eps=0.0001, max_iters=100
        )

        ring_clustering.fit(point)

        radii, centers, memberships = ring_clustering.get_hard_labels()

        import matplotlib.pyplot as plt

        plt.scatter(point[:, 0], point[:, 1], c=memberships)

        # draw centers
        plt.scatter(centers[:, 0], centers[:, 1], c="r", s=100, marker="x")

        # draw radii

        for i in range(n_clusters):
            circle = plt.Circle(centers[i], radii[i], color="r", fill=False)
            plt.gcf().gca().add_artist(circle)

        plt.show()

        print("done")
    