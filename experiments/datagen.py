import numpy as np


def noisy_circle(
    base_position: np.ndarray, radius: float, n_samples: int, noise: float = 0.1
):
    theta = np.linspace(0, 2 * np.pi, n_samples)
    r = radius + noise * np.random.randn(n_samples)  # Add noise to the radius
    x = base_position[0] + r * np.cos(theta)
    y = base_position[1] + r * np.sin(theta)
    return np.stack([x, y], axis=1)

def random_circles(
    center_delimiters, min_max_radius, n_samples_per_circle, n_rings, noise=0.3
):
    assert center_delimiters.shape == (
        4,
        2,
    ), "center_delimiters must be a 4x2 array defining the corners of the rectangle"
    data = []

    # Generate random centers within the rectangle
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
    for center in centers:
        radius = np.random.uniform(min_max_radius[0], min_max_radius[1])
        circle_data = noisy_circle(center, radius, n_samples_per_circle, noise)
        data.append(circle_data)

    return np.concatenate(data, axis=0)


def random_noise(center_delimiters, n_samples):
    # generates random noise between the center_delimiters
    min_x = min(center_delimiters[:, 0])
    max_x = max(center_delimiters[:, 0])
    min_y = min(center_delimiters[:, 1])
    max_y = max(center_delimiters[:, 1])
    x = np.random.uniform(min_x, max_x, n_samples)
    y = np.random.uniform(min_y, max_y, n_samples)
    return np.stack([x, y], axis=1)
