import sys

sys.path.append(".")
sys.path.append("..")

import streamlit as st
import numpy as np
from nrc import NoisyRingsClustering
from experiments.datagen import random_circles, random_noise
from experiments.draw import plot_results
from experiments.colors import get_vibrant_colors


# Define the function to run when button is clicked
def run_algorithm(params, data):
    # Initialize the algorithm with the provided parameters
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

    # Run the algorithm
    nrc.fit(data)

    return nrc.get_labels()


# Streamlit App
st.title("Streamlit App with 8 Parameters")

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
# Button to run the algorithm
if st.button("Run Algorithm"):
    # Run the algorithm with the provided parameters

    # Generate some random data
    # ================ HELPERS ================
    half_rect_size = rect_size // 2
    EXCENTRIC = np.array(
        [
            [-half_rect_size, -half_rect_size],
            [-half_rect_size, half_rect_size],
            [half_rect_size, -half_rect_size],
            [half_rect_size, half_rect_size],
        ]
    )
    CONCENTRIC = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    NOISE_CONCENTRIC_DELIM = np.array(
        [[-half_rect_size, -half_rect_size], [half_rect_size, half_rect_size]]
    )

    CENTER_DELIMS = CONCENTRIC if mode == "concentric" else EXCENTRIC

    # ================ RINGS PARAMETERS ================

    MINMAX_RADIUS = np.array([min_radius, max_radius])
    N_RINGS = n_rings
    CIRCLES_NOISE = circle_noise
    N_SAMPLES_PER_RING = 150

    # ================ BG NOISE PARAMETERS ================
    N_BACKGROUND_NOISE = background_noise
    CENTER_DELIMS_NOISE = NOISE_CONCENTRIC_DELIM

    # ================ ALGORITHM PARAMETERS ================
    # "fuzzycmeans" or "concentric"
    INIT_METHOD = init_method
    FUZINESS_PARAM = q
    CONVERGENCE_EPS = convergence_eps
    MAX_ITERS = max_iters
    NOISE_DISTANCE_THRESHOLD = noise_distance_threshold
    APPLY_NOISE_REMOVAL = apply_noise_removal
    data = random_circles(
        center_delimiters=CENTER_DELIMS,
        min_max_radius=MINMAX_RADIUS,
        n_samples_per_circle=N_SAMPLES_PER_RING,
        n_rings=N_RINGS,
        noise=CIRCLES_NOISE,
    )

    data = np.concatenate(
        [
            data,
            random_noise(
                center_delimiters=CENTER_DELIMS_NOISE, n_samples=N_BACKGROUND_NOISE
            ),
        ],
        axis=0,
    )
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

    # Plot the results
    fig = plot_results(
        data,
        res[0],
        res[1],
        res[2],
        get_vibrant_colors(n_rings),
    )

    # Display the plot in Streamlit
    st.pyplot(fig)
