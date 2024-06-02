import timeit
from experiments.utils import load_experiment, ExperimentParams
from dataclasses import dataclass
import numpy as np
import logging
from collections import defaultdict
from tabulate import tabulate

# disable logging
logging.disable(logging.CRITICAL)


@dataclass
class ExperimentResults:
    error: float
    radii: list
    centers: list
    labels: list
    time: float
    iters: int

    params: ExperimentParams


def benchmark(func, *args, **kwargs):
    start = timeit.default_timer()
    res = func(*args, **kwargs)
    end = timeit.default_timer()
    return res, end - start


def geterror(radii, centers, labels, data):
    # error is defined as the sum of the distances between the centers and the radii for each ring

    error = 0
    total_points = 0

    for i in range(len(data)):
        ringidx = labels[i]
        if ringidx == -1:
            continue  # ignore noise points
        center = centers[ringidx]
        radius = radii[ringidx]

        error += np.abs(np.linalg.norm(data[i] - center) - radius)
        total_points += 1

    return error / total_points


def test_experiment(expname: str, subdolder: str):
    model, data, cfg = load_experiment(expname, subdolder)

    _, time = benchmark(lambda: model.fit(data))

    # Get the clustering results
    radii, centers, labels = model.get_labels()
    n_rings = model.n_rings
    error = geterror(radii, centers, labels, data)

    return ExperimentResults(
        error, radii, centers, labels, time, model.last_iter + 1, cfg
    )


def main(args):
    # set current working directory to current file
    # from ./experiments read all filenames with no json extension
    import os

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Get all the experiment files
    experiment_files = [
        f
        for f in os.listdir("./experiments/{}/".format(args.sf))
        if f.endswith(".json")
    ]

    # prune json extension
    experiment_names = [f[:-5] for f in experiment_files]
    experiment_subfolder = args.sf

    # Run all the experiments
    results = {}

    for expname in experiment_names:
        results[expname] = test_experiment(expname, experiment_subfolder)

    grouped_data = defaultdict(list)

    for expname in results:
        cfg = results[expname].params
        key = cfg
        grouped_data[key].append(
            (results[expname].error, results[expname].time, results[expname].iters)
        )

    # Compute averages and prepare data for tabulation
    final_data = []

    for key, values in grouped_data.items():
        n_rings = key.n_rings
        rings_noise = key.circles_noise
        background_noise = key.n_background_noise
        n_samples = len(values)
        avg_error = sum(v[0] for v in values) / n_samples
        avg_time = sum(v[1] for v in values) / n_samples
        avg_iters = sum(v[2] for v in values) / n_samples
        samples_per_ring = key.n_samples_per_ring
        final_data.append(
            [
                n_rings,
                samples_per_ring,
                rings_noise,
                background_noise,
                avg_error,
                avg_time,
                avg_iters,
                n_samples,
            ]
        )

    # Define table headers
    headers = [
        "N rings",
        "Samp per ring",
        "Rings noise",
        "Background noise",
        "Avg Error",
        "Avg Time",
        "Iters",
        "N Samples",
    ]

    # Print the table
    print(tabulate(final_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--sf", type=str, default="general", help="Subfolder to look for experiments"
    )
    main(parser.parse_args())
