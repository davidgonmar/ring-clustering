import numpy as np


def noisy_circle(
    base_position: np.ndarray, radius: float, n_samples: int, noise: float = 0.1
):  
    """
    Generate points on a circle with noise

    Args:
        base_position: the center of the circle
        radius: the radius of the circle
        n_samples: the number of samples to generate
        noise: the noise to add to the contour
    """
    theta = np.linspace(0, 2 * np.pi, n_samples)
    r = radius + noise * np.random.randn(n_samples)  # Add noise to the radius
    x = base_position[0] + r * np.cos(theta)
    y = base_position[1] + r * np.sin(theta)
    return np.stack([x, y], axis=1)


def random_circles(
    center_delimiters: np.ndarray, min_max_radius: np.ndarray, n_samples_per_circle: int, n_rings: int, noise=0.3
):  
    """
    Generate random circles within a rectangle

    Args:
        center_delimiters: a 4x2 array defining the corners of the rectangle
        min_max_radius: a 2-tuple defining the min and max radius of the circles
        n_samples_per_circle: the number of samples per circle
        n_rings: the number of circles to generate
        noise: the noise to add to the contour of the circles
    """
    assert center_delimiters.shape == (
        4,
        2,
    ), "center_delimiters must be a 4x2 array defining the corners of the rectangle"
    data = []

    # generate random centers within the rectangle
    min_x = min(center_delimiters[:, 0])
    max_x = max(center_delimiters[:, 0])
    min_y = min(center_delimiters[:, 1])
    max_y = max(center_delimiters[:, 1])
    centers = np.array(
        [
            [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
            for _ in range(n_rings)
        ]
    )
    # generate circles around the centers
    for center in centers:
        radius = np.random.uniform(min_max_radius[0], min_max_radius[1])
        circle_data = noisy_circle(center, radius, n_samples_per_circle, noise)
        data.append(circle_data)

    return np.concatenate(data, axis=0)


def random_noise(center_delimiters: np.ndarray, n_samples: int):
    """
    Generate random noise within a rectangle sampled uniformly

    Args:
        center_delimiters: a 4x2 array defining the corners of the rectangle
        n_samples: the number of samples to generate
    """
    min_x = min(center_delimiters[:, 0])
    max_x = max(center_delimiters[:, 0])
    min_y = min(center_delimiters[:, 1])
    max_y = max(center_delimiters[:, 1])
    x = np.random.uniform(min_x, max_x, n_samples)
    y = np.random.uniform(min_y, max_y, n_samples)
    return np.stack([x, y], axis=1)
