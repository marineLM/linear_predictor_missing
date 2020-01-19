import pandas as pd
from sklearn.model_selection import train_test_split
from collections import namedtuple

from ground_truth import generate_data_mixture,\
                         generate_data_selfmasked_proba,\
                         bayes_rate, bayes_rate_monte_carlo, bayes_rate_r2,\
                         generate_toy_params_mixture,\
                         generate_toy_params_selfmasked_proba
from estimators import ConstantImputedMLPR

from joblib import Memory, Parallel, delayed
location = './cachedir'
memory = Memory(location, verbose=0)

# Result item to create the DataFrame in a consistent way.
ResultItem = namedtuple('ResultItem', ['method', 'iter', 'train_test', 'n',
                                       'p', 'mse', 'r2'])


@memory.cache
def run_one(X, y, method, params):
    print(method, X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    if params == "no_params":
        reg = method()
    else:
        reg = method(params)

    reg.fit(X_train, y_train)

    pred_test = reg.predict(X_test)
    pred_train = reg.predict(X_train)

    mse_train = ((y_train - pred_train)**2).mean()
    mse_test = ((y_test - pred_test)**2).mean()

    var_train = ((y_train - y_train.mean())**2).mean()
    var_test = ((y_test - y_test.mean())**2).mean()

    r2_train = 1 - mse_train/var_train
    r2_test = 1 - mse_test/var_test

    return {'train': {'mse': mse_train, 'r2': r2_train},
            'test': {'mse': mse_test, 'r2': r2_test}
            }


def get_results(i, g, est_params, methods):

    result_iter = []
    for X, y in g:
        n, p = X.shape

        for method, est in methods.items():

            if method in est_params:
                # necessary to copy for use in later iterations
                params = est_params[method].copy()

                if 'MLP' in method:
                    type_width = params.pop('type_width')
                    q = params.pop('width')
                    d = params.pop('depth')
                    if type_width == 'fixed':
                        hls = (q, )*d
                    elif type_width == 'linear':
                        hls = (int(q*p), )*d
                    elif type_width == 'exponential':
                        hls = (int(q*2**p), )*d
                    elif type_width == 'constant':
                        hls = (int(q), )*d
                    if hls[0] < 1:
                        continue
                    params['hidden_layer_sizes'] = hls

            else:
                params = "no_params"

            if method == 'ExpandedLR' and p > 10:
                print('ExpandedLR would take too much memory with p > 10')
                continue
            if method == 'ExpandedLR' and p > 9 and n > 1e5:
                print('ExpandedLR would take too much memory' +
                      'with p>9 and n>1e5')
                continue

            print(method)
            new_score = run_one(X, y, est, params)

            # The 0.75 factor comes from the fact that only 75% of the data is
            # used for training (the rest is used for testing).
            res_train = ResultItem(method=method, iter=i, train_test="train",
                                   n=0.75*n, p=p, **new_score["train"])
            res_test = ResultItem(method=method, iter=i, train_test="test",
                                  n=0.75*n, p=p, **new_score["test"])

            result_iter.extend([res_train, res_test])

    return result_iter


def run(n_iter, n_sizes, p_sizes, data_type, data_desc, est_params, methods,
        filename, rs, n_jobs=1):

    if 'mixture' in data_type:
        generate_params = generate_toy_params_mixture
        generate_data = generate_data_mixture
    else:
        generate_params = generate_toy_params_selfmasked_proba
        generate_data = generate_data_selfmasked_proba

    rs_params = rs.choice(1000)
    rs_gen = rs.choice(10000, n_iter)

    data_params = [generate_params(p, **data_desc, random_state=rs_params)
                   for p in p_sizes]

    pair_iter_gen = []
    for it in range(n_iter):
        for data_param in data_params:
            gen = generate_data(n_sizes, data_param, rs_gen[it])
            pair_iter_gen.append((it, gen))

    results = Parallel(n_jobs=n_jobs)(
        delayed(get_results)(it, list(g), est_params, methods)
        for it, g in pair_iter_gen
    )

    results = [item for result_iter in results for item in result_iter]
    scores_avg = pd.DataFrame(results)
    scores_avg.to_csv('../results/' + filename + '_sa.csv')

    if 'mixture' in data_type:
        br_mse = [bayes_rate(d_p) for d_p in data_params]
        br_r2 = [bayes_rate_r2(d_p) for d_p in data_params]
        data = {'p': p_sizes, 'mse': br_mse, 'r2': br_r2}
        br = pd.DataFrame(data)
        br.to_csv('../results/' + filename + '_br.csv')
