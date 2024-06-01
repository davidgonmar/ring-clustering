import timeit
from experiments.utils import load_experiment
from dataclasses import dataclass
import numpy as np
import logging

# disable logging
logging.disable(logging.CRITICAL)


@dataclass
class ExperimentResults:
    error: float
    radii: list
    centers: list
    labels: list
    time: float


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


def test_experiment(expname: str):
    model, data, _ = load_experiment(expname)

    _, time = benchmark(lambda: model.fit(data))

    # Get the clustering results
    radii, centers, labels = model.get_labels()
    n_rings = model.n_rings
    error = geterror(radii, centers, labels, data)

    return ExperimentResults(error, radii, centers, labels, time)


def main():
    # set current working directory to current file
    # from ./experiments read all filenames with no json extension
    import os

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Get all the experiment files
    experiment_files = [f for f in os.listdir("./experiments") if f.endswith(".json")]

    # prune json extension
    experiment_names = [f[:-5] for f in experiment_files]

    # Run all the experiments
    results = {}

    for expname in experiment_names:
        results[expname] = test_experiment(expname)

    print(list(map(lambda x: (x, results[x].time, results[x].error), results.keys())))


if __name__ == "__main__":
    main()
