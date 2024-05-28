import numpy as np
import matplotlib.pyplot as plt
from nrc import NOISE
plt.rcParams["figure.dpi"] = 600
def plot_results(data: np.ndarray, radii: np.ndarray, centers: np.ndarray, labels: np.ndarray, cluster_base_colors: np.ndarray, show_noise: bool = False) -> plt.Figure:
    """
    Plot the results of the clustering
    Args:
        data: (n_samples, 2) ndarray, the data points
        radii: (n_rings) ndarray, the radii of the rings
        centers: (n_rings, 2) ndarray, the centers of the rings
        labels: (n_samples) ndarray, the labels of the samples
        cluster_base_colors: (n_rings, 3) ndarray, the base colors of the clusters
        show_noise: bool, whether to show noise points (default: False)
    """
    assert len(radii) == len(centers) == len(cluster_base_colors), "expected radii.shape[0] == centers.shape[0] == cluster_base_colors.shape[0], got {} == {} == {}".format(len(radii), len(centers), len(cluster_base_colors))
    purple = np.array([0.5, 0, 0.5])
    # Assign a color to each point
    colors = np.array([cluster_base_colors[labels[i]] if labels[i] != NOISE else purple for i in range(len(labels))])

    # Create a plot
    fig, ax = plt.subplots()

    # Plot the data
    ax.scatter(data[:, 0], data[:, 1], c=colors, s=10)

    # Draw circles with radii
    for i in range(len(radii)):
        circle = plt.Circle(centers[i], radii[i], fill=False, edgecolor=cluster_base_colors[i])
        ax.add_artist(circle)
        # Draw the center of the circle
        ax.scatter(centers[i][0], centers[i][1], c="black", s=10)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    return fig