import numpy as np
import json
from dataclasses import dataclass
from nrc import NoisyRingsClustering

np.random.seed(42)

DATA_PATH = "./data"
EXPERIMENTS_PATH = "./experiments"


@dataclass
class ExperimentParams:
    data_filename: str
    n_rings: int
    q: float
    convergence_eps: float
    max_iters: int
    noise_entropy_threshold: float
    max_noise_checks: int


def save_np_to_csv(data: np.ndarray, filename: str) -> None:
    np.savetxt(filename, data, delimiter=",")


def load_np_from_csv(filename: str) -> np.ndarray:
    return np.loadtxt(filename, delimiter=",")


def save_experiment_params(params: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(params, f)


def load_experiment_params(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


def save_experiment(model: NoisyRingsClustering, data: np.ndarray, name: str) -> None:
    params = ExperimentParams(
        data_filename=f"{DATA_PATH}/{name}.csv",
        n_rings=model.n_rings,
        q=model.q,
        convergence_eps=model.convergence_eps,
        max_iters=model.max_iters,
        noise_entropy_threshold=model.noise_entropy_threshold,
        max_noise_checks=model.max_noise_checks,
    )
    save_np_to_csv(data, params.data_filename)
    save_experiment_params(params.__dict__, f"{EXPERIMENTS_PATH}/{name}.json")


def load_experiment(name: str) -> (NoisyRingsClustering, np.ndarray):
    params = load_experiment_params(f"{EXPERIMENTS_PATH}/{name}.json")
    data = load_np_from_csv(params["data_filename"])
    model = NoisyRingsClustering(
        n_rings=params["n_rings"],
        q=params["q"],
        convergence_eps=params["convergence_eps"],
        max_iters=params["max_iters"],
        noise_entropy_threshold=params["noise_entropy_threshold"],
        max_noise_checks=params["max_noise_checks"],
    )
    return model, data
