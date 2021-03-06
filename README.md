This repository contains the code to reproduce the experiments in our paper:
*Linear predictor on linearly-generated data with missing values: non consistency and solutions*.

The file **environment_lpm.yml** indicates the packages required to run the code as well as the
versions that were used.

The file **ground_truth.py** contains the parameter generation, data generation and Bayes rate functions.

The file **estimators.py** contains all the classes of estimators used.

The file **learning_curves.py** contains the code which runs the experiments.

The file **launch_experiment** takes `mixture1`, `mixture3`, or `selfmasked_proba` as argument. For example `python launch_experiment.py mixture1` launches the simulations for mixture1. Change this file if you want to change the values of the parameters tested for the simulations. Upon completion of this script, a csv file is saved that records the performances obtained.

The file **plot_curves**, **plot_MLP_scaling_n**, **plot_MLP_scaling_q**, **plot_boxplots** contain the code used to plot the figures based on the csv file obtained. **plot_curves** plots the learning curves.
