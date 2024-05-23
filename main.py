import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class RingCluster:
    position: np.ndarray
    radius: int

# assumes every point is in the cluster
def estimate_radius_and_center(ptrs: np.ndarray):
    center = ptrs.mean(axis=0)
    radius = ((ptrs - center) ** 2).sum(axis=1).mean() ** 0.5
    return center, radius

def loss(ptrs: np.ndarray, cluster: RingCluster):
    distances = np.linalg.norm(ptrs - cluster.position, axis=1)
    return np.mean(np.abs(distances - cluster.radius))

def individual_loss(ptrs: np.ndarray, cluster: RingCluster):
    distances = [np.linalg.norm(ptr - cluster.position) for ptr in ptrs]
    return np.abs(np.array(distances) - cluster.radius)

class RingClustering:
    def __init__(self):
        pass
    
    
    def _neighbourhood(self, ptrid, eps: float = 1):
        ptr = self.ptrs[ptrid]
        return np.linalg.norm(self.ptrs - ptr, axis=1) < eps
    
    def _expand_cluster(self, ptridx: int, cluster_id: int, eps: float = 1):
        seeds = np.where(self._neighbourhood(ptridx, eps))[0]
        for seed in seeds:
            if self.labels[seed] == -1:
                self.labels[seed] = cluster_id
            elif self.labels[seed] == 0:
                self.labels[seed] = cluster_id
                self._expand_cluster(seed, cluster_id, eps)

    def fit(self, ptrs: np.ndarray, n_initial_centers: int = 4):
        MIN_RADIUS = 1
        MAX_RADIUS = 20
        assert ptrs.ndim == 2, "expected ptrs to have shape (n_samples, n_features), got {}".format(ptrs.shape)
        # random positions enclosed in the ptrs
        n_dims = ptrs.shape[1]
        maxes = ptrs.max(axis=0, keepdims=False)
        mins = ptrs.min(axis=0, keepdims=False)

        rand_centers = np.random.uniform(mins, maxes, size=(n_initial_centers, n_dims))

        # random radius
        rand_radius = np.random.uniform(MIN_RADIUS, MAX_RADIUS, size=(n_initial_centers))

        self.clusters = [RingCluster(position, radius) for position, radius in zip(rand_centers, rand_radius)]
        self.ptrs = ptrs

        # compute losses for each cluster
        losses = []
        for cluster in self.clusters:
            losses.append(individual_loss(ptrs, cluster))
    
        losses = np.array(losses) # shape (n_clusters, n_samples)
        # select the cluster with the lowest loss for each point
        self.labels = np.argmin(losses, axis=0)

        for i in range(300):
            # update clusters
            for i, cluster in enumerate(self.clusters):
                cluster.ptrs = ptrs[self.labels == i]
                cluster.position, cluster.radius = estimate_radius_and_center(cluster.ptrs)
            
            # compute losses for each cluster
            losses = []
            for cluster in self.clusters:
                losses.append(individual_loss(ptrs, cluster))
        
            losses = np.array(losses) # shape (n_clusters, n_samples)
            # select the cluster with the lowest loss for each point
            new_labels = np.argmin(losses, axis=0)

            # now perform some kind of dbscan algorithm
            i = 0
            for c in self.clusters:
                ptrid = self.ptrs[new_labels == c]
                if len(ptrid) == 0:
                    continue
                self._expand_cluster(ptrid[0], i)
                i += 1

            if np.all(new_labels == self.labels):
                print("Converged after {} iterations".format(i))
                self.labels = new_labels
                break
        
            self.labels = new_labels
        
    
    def draw(self):
        assert self.ptrs.shape[1] == 2, "draw method only supports 2D data"

        for cluster in self.clusters:
            plt.scatter(cluster.position[0], cluster.position[1], c='r')
            circunference = plt.Circle(cluster.position, cluster.radius, color='r', fill=False)
            plt.gca().add_artist(circunference)
        
        # now, each label is a color
        colors = ['b', 'g', 'y', 'm', 'c', 'k']
        for i in range(len(self.clusters)):
            plt.scatter(self.ptrs[self.labels == i, 0], self.ptrs[self.labels == i, 1], c=colors[i])
        
        plt.show()



if __name__ == "__main__":
    # create a ring dataset
    n_samples = 1000
    n_features = 2
    n_clusters = 4
    point = []

    for i in range(n_clusters):
        radius = 10 + np.random.uniform(-1, 1)
        theta = np.linspace(0, 2 * np.pi, n_samples // n_clusters)
        x = radius * np.cos(theta) + np.random.normal(0, 0.1, size=(n_samples // n_clusters))
        y = radius * np.sin(theta) + np.random.normal(0, 0.1, size=(n_samples // n_clusters))
        position = np.random.uniform(-50, 50, size=(2))
        x += position[0]
        y += position[1]
        point.append(np.stack([x, y], axis=1))
    
    
    
    point = np.array(point)

    # merge dim 0 and 1
    point = point.reshape(-1, n_features)

    ring_clustering = RingClustering()

    ring_clustering.fit(point)

    ring_clustering.draw()

    print("done")