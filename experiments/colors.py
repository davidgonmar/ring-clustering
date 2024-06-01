import numpy as np

_vibrant_colors = np.array(
    [
        [0.8, 0.2, 0.2],
        [0.2, 0.8, 0.2],
        [0.2, 0.2, 0.8],
        [0.8, 0.8, 0.2],
        [0.8, 0.2, 0.8],
    ]
)


def get_vibrant_colors(n: int) -> np.ndarray:
    """
    Get n vibrant colors
    Args:
        n: int, the number of colors to get
    Returns:
        ndarray: the vibrant colors
    """
    return _vibrant_colors[:n].copy()
