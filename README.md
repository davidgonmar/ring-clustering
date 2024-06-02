# Noisy Rings Clustering

This repository contains the code and report for the final project of the course "Artificial Intelligence" at the University of Seville.

## Library

The main algorithm is presented in the form of a library under the `nrc` folder.

## Report

Report is available in the `report` folder. It is written in LaTeX and can be compiled using a compatible LaTeX compiler.

## Experiments

Multiple experiments were conducted to test the algorithm, and the results are available on the paper.
To experiment with the algorithm, you can use the `experiments/experimen_playground.ipynb` notebook. It allows to generate random data based on your input,
run the algorithm and visualize the results. It also allows to save the experiment results to a file, to be benchmarked later. Furthermore, it allows to save
the figures as `pgf` files, to be included in a LaTeX document.

To rerun an individual experiment that was saved earlier, you can use the `experiments/run_experiment.ipynb` script. It allows to run an giving it subfolder and name. It also allows to save the plots as `pgf` files.

Experiments are saved in the `experiments/experiments` folder. Each family of experiments is saved into a subfolder, for example `experiments/experiments/test_this`. Each experiment file is saved as a `.json` file, and contains parameters that, with a fixed seed, allow to reproduce the experiment (both with the run_experiment script and the benchmarking script). The generated data is saved under `experiments/data/subfolder`, in csv files. Each experiment json file contains the path to the data file, so it can be loaded and used to run the algorithm.

### Benchmarking

To benchmark the algorithm, you can use the `experiments/benchmark.py` script. It takes as argument `cfg`, which is the family of experiments to benchmark. It will run all the experiments in the family, print out a table with the results, and save a `results.txt` file with a LaTeX table that can be included in the report.

In order to include a new benchmarking family, it must be added to the start of the code (it is a configuration dictionary that includes what to include in the printed table and the results.txt file). The benchmarking script will run all the experiments in the family, and print out the results in a table.

### Other

Other files are helpers, like data generators or helpers to plot the results.
