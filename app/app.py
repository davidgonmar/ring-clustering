import sys

sys.path.append(".")
sys.path.append("..")

import streamlit as st
import numpy as np
from nrc import NoisyRingsClustering
from experiments.datagen import random_circles, random_noise
from experiments.draw import plot_results
from experiments.colors import get_vibrant_colors


def run_algorithm(
    params: dict, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Noisy Rings Clustering algorithm with the provided parameters
    and data.

    Args:
        params: the parameters for the algorithm
        data: the data to cluster

    Returns:
        the labels, the centers, and the radii of the clusters
    """
    nrc = NoisyRingsClustering(
        n_rings=params["n_rings"],
        q=params["q"],
        convergence_eps=params["convergence_eps"],
        max_iters=params["max_iters"],
        noise_distance_threshold=params["noise_distance_threshold"],
        apply_noise_removal=params["apply_noise_removal"],
        max_noise_checks=params["max_noise_checks"],
        init_method=params["init_method"],
    )
    nrc.fit(data)
    return nrc.get_labels()


st.title("Noisy Rings Clustering")

# Input parameters
n_rings = st.number_input("n_rings", value=3)
q = st.number_input("q", value=1.1)
convergence_eps = st.number_input("convergence_eps", value=1e-5)
max_iters = st.number_input("max_iters", value=10000)
noise_distance_threshold = st.number_input("noise_distance_threshold", value=100)
apply_noise_removal = st.checkbox("apply_noise_removal", value=True)
max_noise_checks = st.number_input("max_noise_checks", value=20)
init_method = st.selectbox("init_method", ["fuzzycmeans", "concentric"])

rect_size = st.number_input("rect_size", value=1200)

mode = st.selectbox("mode", ["excentric", "concentric"])
min_radius = st.number_input("min_radius", value=100)
max_radius = st.number_input("max_radius", value=400)

circle_noise = st.number_input("circle_noise", value=8)
background_noise = st.number_input("background_noise", value=20)
n_samples_per_ring = st.number_input("n_samples_per_ring", value=150)


if st.button("Run!"):
    # create the data
    half_rect_size = rect_size // 2
    excentric = np.array(
        [
            [-half_rect_size, -half_rect_size],
            [-half_rect_size, half_rect_size],
            [half_rect_size, -half_rect_size],
            [half_rect_size, half_rect_size],
        ]
    )
    concentric = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    noise_concentric_delim = np.array(
        [[-half_rect_size, -half_rect_size], [half_rect_size, half_rect_size]]
    )
    center_delim = concentric if mode == "concentric" else excentric

    minmax_radius = np.array([min_radius, max_radius])

    data = random_circles(
        center_delimiters=center_delim,
        min_max_radius=minmax_radius,
        n_samples_per_circle=n_samples_per_ring,
        n_rings=n_rings,
        noise=circle_noise,
    )

    data = np.concatenate(
        [
            data,
            random_noise(
                center_delimiters=noise_concentric_delim, n_samples=background_noise
            ),
        ],
        axis=0,
    )

    # run the algorithm
    res = run_algorithm(
        {
            "n_rings": n_rings,
            "q": q,
            "convergence_eps": convergence_eps,
            "max_iters": max_iters,
            "noise_distance_threshold": noise_distance_threshold,
            "apply_noise_removal": apply_noise_removal,
            "max_noise_checks": max_noise_checks,
            "init_method": init_method,
        },
        data,
    )

    # display
    fig = plot_results(
        data,
        res[0],
        res[1],
        res[2],
        get_vibrant_colors(n_rings),
    )
    st.pyplot(fig)
