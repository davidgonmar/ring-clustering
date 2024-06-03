import numpy as np
import json
from dataclasses import dataclass
from nrc import NoisyRingsClustering

np.random.seed(42)

DATA_PATH = "./data"
EXPERIMENTS_PATH = "./experiments"


@dataclass(frozen=True)
class ExperimentParams:
    # data
    data_filename: str

    # model
    n_rings: int
    q: float
    convergence_eps: float
    max_iters: int
    noise_distance_threshold: float
    max_noise_checks: int
    apply_noise_removal: bool
    init_method: str

    # generation params
    n_background_noise: int
    circles_noise: float
    n_samples_per_ring: int

    def __hash__(self):
        return hash(
            (
                self.n_rings,
                self.q,
                self.convergence_eps,
                self.max_iters,
                self.noise_distance_threshold,
                self.max_noise_checks,
                self.apply_noise_removal,
                self.init_method,
                self.n_background_noise,
                self.circles_noise,
                self.n_samples_per_ring,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, ExperimentParams):
            return False
        return (
            self.n_rings == other.n_rings
            and self.q == other.q
            and self.convergence_eps == other.convergence_eps
            and self.max_iters == other.max_iters
            and self.noise_distance_threshold == other.noise_distance_threshold
            and self.max_noise_checks == other.max_noise_checks
            and self.apply_noise_removal == other.apply_noise_removal
            and self.init_method == other.init_method
            and self.n_background_noise == other.n_background_noise
            and self.circles_noise == other.circles_noise
            and self.n_samples_per_ring == other.n_samples_per_ring
        )


def save_np_to_csv(data: np.ndarray, filename: str) -> None:
    """
    Save a numpy array to a csv file

    Args:
        data: the data to save
        filename: the filename to save to
    """
    np.savetxt(filename, data, delimiter=",")


def load_np_from_csv(filename: str) -> np.ndarray:
    """
    Load a numpy array from a csv file

    Args:
        filename: the filename to load from
    Returns:
        data: the loaded data
    """
    return np.loadtxt(filename, delimiter=",")


def save_experiment_params(params: dict, filename: str) -> None:
    """
    Save the experiment parameters to a json file

    Args:
        params: the parameters to save
        filename: the filename to save to
    """
    with open(filename, "w") as f:
        json.dump(params, f)


def load_experiment_params(filename: str) -> dict:
    """
    Load the experiment parameters from a json file

    Args:
        filename: the filename to load from
    Returns:
        params: the loaded parameters
    """
    with open(filename, "r") as f:
        return json.load(f)


def save_experiment(
    model: NoisyRingsClustering,
    data: np.ndarray,
    name: str,
    subfolder: str,
    extra_params: dict,
) -> None:
    """
    Save an experiment to disk, given a model, data, and extra params, to a given name and subfolder

    Args:
        model: the model to save
        data: the data to save
        name: the name of the experiment
        subfolder: the subfolder to save the experiment to
        extra_params: the extra parameters to save (n_background_noise, circles_noise, n_samples_per_ring)
    """
    params = ExperimentParams(
        data_filename=f"{DATA_PATH}/{subfolder}/{name}.csv",
        n_rings=model.n_rings,
        q=model.q,
        convergence_eps=model.convergence_eps,
        max_iters=model.max_iters,
        noise_distance_threshold=model.noise_distance_threshold,
        max_noise_checks=model.max_noise_checks,
        apply_noise_removal=model.apply_noise_removal,
        init_method=model.init_method,
        n_background_noise=extra_params["n_background_noise"],
        circles_noise=extra_params["circles_noise"],
        n_samples_per_ring=extra_params["n_samples_per_ring"],
    )
    save_np_to_csv(data, params.data_filename)
    save_experiment_params(
        params.__dict__, f"{EXPERIMENTS_PATH}/{subfolder}/{name}.json"
    )


def load_experiment(
    name: str, subfolder: str
) -> (NoisyRingsClustering, np.ndarray, ExperimentParams):
    """
    Load an experiment from disk, given a name and subfolder
    
    Args:
        name: the name of the experiment
        subfolder: the subfolder to load the experiment from
    Returns:
        model: the loaded model
        data: the loaded data
        config: the loaded configuration
    """
    params = load_experiment_params(f"{EXPERIMENTS_PATH}/{subfolder}/{name}.json")
    data = load_np_from_csv(params["data_filename"])
    model = NoisyRingsClustering(
        n_rings=params["n_rings"],
        q=params["q"],
        convergence_eps=params["convergence_eps"],
        max_iters=params["max_iters"],
        noise_distance_threshold=params["noise_distance_threshold"],
        max_noise_checks=params["max_noise_checks"],
        apply_noise_removal=params["apply_noise_removal"],
        init_method=params["init_method"],
    )
    config = ExperimentParams(**params)
    return model, data, config
