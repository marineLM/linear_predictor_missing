"""Running this file launches the simulations.
It can be edited to:
    - change the simulation settings desired,
    - change the estimators that should be tested for these simulations.
"""
import numpy as np
import pickle
import itertools
import argparse
from estimators import ConstantImputedLR,\
                       ExpandedLR,\
                       EMLR,\
                       ConstantImputedMLPR,\
                       MICELR
from learning_curves import run
from script_helper import add_MLP_method, choose_filename

parser = argparse.ArgumentParser()
parser.add_argument('data_type', help='type of simulation',
                    choices=['mixture1', 'mixture3', 'selfmasked_proba'])
args = parser.parse_args()

# Choice of parameters to simulate data.
data_descs = {
              'selfmasked_proba':
              {'prop_incomplete': 1, 'missing_rate': 0.25, 'prop_latent': 0.5,
               'lam': 0.5, 'mean': 0, 'noise': False},

              'mixture1':
              {'n_comp': 1, 'prob_comp': [1], 'mean_factor': 2,
               'prop_latent': 0.5, 'noise': False},

              'mixture3':
              {'n_comp': 3, 'prob_comp': [1/3, 1/3, 1/3], 'mean_factor': 2,
               'prop_latent': 0.5, 'noise': False}
    }


if __name__ == "__main__":

    n_iter = 10
    n_jobs = 10
    file_root = 'allresultsPPCA'
    methods = {
        'ConstantImputedLR': ConstantImputedLR,
        'ExpandedLR': ExpandedLR,
        'EMLR': EMLR,
        'MICELR': MICELR
    }
    est_params = {}

    if args.data_type == 'mixture3':
        n_sizes = [1e3, 5e3, 1e4, 2.5e4, 5e4, 1e5, 5e5]
        n_sizes = [int(i) for i in n_sizes]
        p_sizes = [4, 5, 6, 7, 8, 9, 10]
        for q in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            add_MLP_method('exponential', q, 1, False, methods, est_params)

    elif args.data_type == 'mixture1':
        n_sizes = [1e3, 5e3, 1e4, 2.5e4, 5e4, 1e5]
        n_sizes = [int(i) for i in n_sizes]
        p_sizes = [4, 6, 8, 10, 12, 14, 16]
        for q in [0.3, 0.5, 1, 2, 4]:
            add_MLP_method('linear', q, 1, False, methods, est_params)

    elif args.data_type == 'selfmasked_proba':
        n_sizes = [1e3, 5e3, 1e4, 2.5e4, 5e4, 1e5]
        n_sizes = [int(i) for i in n_sizes]
        p_sizes = [4, 6, 8, 10, 12, 14, 16]
        for q in [0.3, 0.5, 1, 2, 4]:
            add_MLP_method('linear', q, 1, False, methods, est_params)

    data_desc = data_descs[args.data_type]
    filename = choose_filename(file_root, n_sizes, p_sizes, args.data_type,
                               n_iter)
    rng = np.random.RandomState(123)

    run_params = {
        'n_iter': n_iter,
        'n_sizes': n_sizes,
        'p_sizes': p_sizes,
        'data_type': args.data_type,
        'data_desc': data_desc,
        'methods': methods,
        'est_params': est_params,
        'filename': filename,
        'rs': rng,
        'n_jobs': n_jobs}

    file = open('../results/' + filename, 'wb')
    pickle.dump(run_params, file)
    file.close()

    run(**run_params)
