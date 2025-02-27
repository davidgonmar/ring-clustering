import timeit
from experiments.utils import load_experiment, ExperimentParams
from dataclasses import dataclass
import numpy as np
import logging
from collections import defaultdict
from tabulate import tabulate


# disable logging
logging.disable(logging.CRITICAL)


configs = {
    "general": {
        "subfolder": "general",
        "cmd": [
            "n_rings",
            "samples_per_ring",
            "rings_noise",
            "background_noise",
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
        ],
        "latex": [
            "n_rings",
            "rings_noise",
            "background_noise",
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
        ],
    },
    "concentric": {
        "subfolder": "concentric",
        "cmd": [
            "n_rings",
            "samples_per_ring",
            "rings_noise",
            "background_noise",
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
        ],
        "latex": [
            "n_rings",
            "rings_noise",
            "background_noise",
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
        ],
    },
    "needle_in_haystack": {
        "subfolder": "needle_in_haystack",
        "cmd": [
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
            "background_noise",
        ],
        "latex": [
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
            "background_noise",
        ],
    },
    "bg_noise": {
        "subfolder": "bg_noise",
        "cmd": [
            "n_rings",
            "samples_per_ring",
            "rings_noise",
            "background_noise",
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
        ],
        "latex": [
            "n_rings",
            "rings_noise",
            "background_noise",
            "avg_error",
            "avg_time",
            "avg_iters",
            "n_experiments",
            "avg_detected_noise_n",
        ],
    },
}


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
    radii, centers, labels = model.get_labels()
    error = geterror(radii, centers, labels, data)

    return ExperimentResults(
        error, radii, centers, labels, time, model.last_iter + 1, cfg
    )


def main(args):
    # set current working directory to current file
    # from ./experiments read all filenames with no json extension
    import os

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    conf = configs[args.cfg]
    experiment_subfolder = conf["subfolder"]

    # Get all the experiment files
    experiment_files = [
        f
        for f in os.listdir("./experiments/{}/".format(experiment_subfolder))
        if f.endswith(".json")
    ]

    experiment_names = [f[:-5] for f in experiment_files]
    # prune json extension

    # Run all the experiments
    results = {}

    for expname in experiment_names:
        results[expname] = test_experiment(expname, experiment_subfolder)

    grouped_data = defaultdict(list)

    for expname in results:
        cfg = results[expname].params
        key = cfg
        detected_noise_n = len([l for l in results[expname].labels if l == -1])
        grouped_data[key].append(
            (
                results[expname].error,
                results[expname].time,
                results[expname].iters,
                detected_noise_n,
            )
        )

    # present the results
    final_data = []

    for key, values in grouped_data.items():
        n_rings = key.n_rings
        rings_noise = key.circles_noise
        background_noise = key.n_background_noise
        n_experiments = len(values)
        avg_error = sum(v[0] for v in values) / n_experiments
        avg_time = sum(v[1] for v in values) / n_experiments
        avg_iters = sum(v[2] for v in values) / n_experiments
        samples_per_ring = key.n_samples_per_ring
        avg_detected_noise_n = sum(v[3] for v in values) / n_experiments
        final_data.append(
            {
                "n_rings": n_rings,
                "samples_per_ring": samples_per_ring,
                "rings_noise": rings_noise,
                "background_noise": background_noise,
                "avg_error": avg_error,
                "avg_time": avg_time,
                "avg_iters": avg_iters,
                "n_experiments": n_experiments,
                "avg_detected_noise_n": avg_detected_noise_n,
                "q": key.q,
            }
        )

    headers = {
        "n_rings": "Number of rings",
        "samples_per_ring": "Samples per ring",
        "rings_noise": "Ring noise",
        "background_noise": "Background noise",
        "avg_error": "Avg. Error",
        "avg_time": "Avg. Runtime",
        "avg_iters": "Iterations",
        "n_experiments": "Experiments",
        "avg_detected_noise_n": "Avg. Detected Noise",
        "q": "q",
    }

    # filter to only show the columns we want
    final_data_cmd = [[d[h] for h in conf["cmd"]] for d in final_data]
    final_data_latex = [[d[h] for h in conf["latex"]] for d in final_data]
    headers_cmd = [headers[h] for h in conf["cmd"]]
    headers_latex = [headers[h] for h in conf["latex"]]
    cmd_table = tabulate(final_data_cmd, headers=headers_cmd, tablefmt="grid")
    latex_table = tabulate(
        final_data_latex, headers=headers_latex, tablefmt="latex_raw"
    )

    print(cmd_table)

    # save the results to a file, to be used in the report
    with open("results.txt", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--cfg", type=str, default="general", help="Subfolder to look for experiments"
    )
    main(parser.parse_args())
