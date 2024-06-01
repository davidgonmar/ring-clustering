import timeit
from experiments.utils import load_experiment
from dataclasses import dataclass


@dataclass
class ExperimentResults:
    error: float
    radii: list
    centers: list
    memberships: list
    time: float


def benchmark(func, *args, **kwargs):
    start = timeit.default_timer()
    res = func(*args, **kwargs)
    end = timeit.default_timer()
    return res, end - start


def test_experiment(expname: str):
    model, data = load_experiment(expname)

    _, time = benchmark(model.fit(data))

    # Get the clustering results
    radii, centers, memberships = model.get_labels()
    n_rings = model.n_rings
    error = 1

    return ExperimentResults(error, radii, centers, memberships, time)


def main():
    # set current working directory to current file
    # from ./experiments read all filenames with no json extension
    import os
    import json

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Get all the experiment files
    experiment_files = [f for f in os.listdir("./experiments") if f.endswith(".json")]

    # prune json extension
    experiment_names = [f[:-5] for f in experiment_files]

    # Run all the experiments
    results = {}

    for expname in experiment_names:
        results[expname] = test_experiment(expname)

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
