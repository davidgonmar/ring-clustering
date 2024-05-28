import numpy as np
import matplotlib.pyplot as plt
from nrc import NOISE
plt.rcParams["figure.dpi"] = 600

def plot_results(data: np.ndarray, radii: np.ndarray, centers: np.ndarray, labels: np.ndarray, cluster_base_colors: np.ndarray, show_noise: bool = False):
    """
    Plot the results of the clustering
    Args:
        radii: (n_rings) ndarray, the radii of the rings
        centers: (n_rings, 2) ndarray, the centers of the rings
        labels: (n_samples) ndarray, the labels of the samples
        cluster_base_colors: (n_rings, 3) ndarray, the base colors of the clusters
    """
    assert len(radii) == len(centers) == len(cluster_base_colors), "expected radii.shape[0] == centers.shape[0] == cluster_base_colors.shape[0], got {} == {} == {}".format(len(radii), len(centers), len(cluster_base_colors))

    circles = data
    vibrant_colors = cluster_base_colors
    # each cluster is assigned a vibrant color
    # then, the color of a point is a mix of the vibrant color of the cluster it belongs to according to the membership
    # remember membership is a matrix of shape (n_clusters, n_samples)
    # for each sample, color = vibrant_colors[label]

    # Assign a color to each point
    colors = np.array([vibrant_colors[labels[i]] if labels[i] != NOISE else [0, 0, 0] for i in range(len(labels))])


    # Plot the data
    plt.scatter(circles[:, 0], circles[:, 1], c=colors, s=10)



    # Draw circles with radii
    for i in range(len(radii)):
        circle = plt.Circle(centers[i], radii[i], fill=False, edgecolor="black")
        plt.gca().add_artist(circle)
        # draw the center of the circle
        plt.scatter(centers[i][0], centers[i][1], c="red", s=50)

    # Show the plot
    plt.axis("equal")
    plt.show()
