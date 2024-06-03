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


# Button to run the algorithm
if st.button("Run Algorithm"):
    # Run the algorithm with the provided parameters

    # Generate some random data
    # ================ HELPERS ================
    EXCENTRIC = np.array([[-600, -600], [-600, 600], [600, -600], [600, 600]])
    CONCENTRIC = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    EXCENTRIC_CLOSE = np.array([[-200, -200], [-200, 200], [200, -200], [200, 200]])
    NOISE_CONCENTRIC_DELIM = np.array(
        [[-1200, -1200], [-1200, 1200], [1200, -1200], [1200, 1200]]
    )

    # ================ RINGS PARAMETERS ================
    CENTER_DELIMS = EXCENTRIC

    MINMAX_RADIUS = np.array([100, 400])
    N_RINGS = 3
    CIRCLES_NOISE = 8
    N_SAMPLES_PER_RING = 150

    # ================ BG NOISE PARAMETERS ================
    N_BACKGROUND_NOISE = 20
    CENTER_DELIMS_NOISE = EXCENTRIC

    # ================ ALGORITHM PARAMETERS ================
    # "fuzzycmeans" or "concentric"
    INIT_METHOD = "fuzzycmeans"
    FUZINESS_PARAM = 1.1
    CONVERGENCE_EPS = 1e-5
    MAX_ITERS = 10000
    NOISE_DISTANCE_THRESHOLD = 70
    APPLY_NOISE_REMOVAL = True
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
