"""This file contains functions to:
    - compute the parameters used to generate simulated data,
    - generate simulated data using these parameters,
    - compute the Bayes rate of the pattern mixture model (both exact analytic
    expression and Monte Carlo approximation).
"""
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import norm
from math import sqrt, floor, log
from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)


def generate_toy_params_mixture(n_features, n_comp, prob_comp, mean_factor,
                                prop_latent, noise=False, random_state=None):
    """Creates parameters for generating data with `generate_data_mixture`.

    Parameters
    ----------
    n_features: int
        The number of features desired.

    n_comp: int
        The number of Gaussian components desired.

    prob_comp: array-like, shape (n_comp, )
        The ith entry is the probability that a sample is generated with the
        ith Gaussian component. Entries should sum to 1.

    mean_factor: float
        The mean of the ith multivariate gaussian is a vector with values 0 or
        mean_factor*var where var is the average variance of a gaussian
        component.

    prop_latent: float
        The number of latent factors used to generate the covariance matrix is
        prop_latent*n_features. The less factors the higher the correlations.
        Should be between 0 and 1.

    noise: boolean, optional, default False
        Whether or not the response should be generated with noise

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    # We want to limit to at most one Gaussian per missing data pattern,
    # plus the initialisation scheme of the means does not work otherwise.
    if n_comp > 2**n_features:
        raise ValueError("The number of components should be less or equal" +
                         "to the number of missing data patterns" +
                         "(i.e. n_comp <= 2**n_features)")

    if len(prob_comp) != n_comp:
        raise ValueError("prob_comp should be of size (n_comp, )," +
                         "got len(prob_comp)={} while n_comp={}".format(
                             len(prob_comp), n_comp))

    if sum(prob_comp) != 1:
        raise ValueError("prob_comp must sum to 1")

    rng = check_random_state(random_state)

    n_pat = np.empty((n_comp,), dtype=int)
    for i, p in enumerate(prob_comp):
        n_pat[i] = round(p*2**n_features)

    # Correction to ensure that the total number of patterns is correct
    n_pat[n_comp-1] = 2**n_features - n_pat[0:n_comp-1].sum()

    pat_to_comp = [np.repeat(i, n_pat[i]) for i in range(n_comp)]
    pat_to_comp = np.concatenate(pat_to_comp)
    rng.shuffle(pat_to_comp)

    probs = [prob_comp[i]/n_pat[i] for i in pat_to_comp]

    # Generate covariances
    # --------------------
    covs = []
    for _ in range(n_comp):
        B = rng.randn(n_features, int(prop_latent*n_features))
        cov = B.dot(B.T) + np.diag(rng.uniform(low=0.1, size=n_features))
        covs.append(cov)

    # Generate means
    # --------------
    means = []
    means.append(np.zeros((n_features, )))

    var = np.concatenate([np.diag(cov) for cov in covs])
    mean = mean_factor*np.mean(var)
    # start at 1 because the mean for the first component is all zeros.
    for i_comp in range(1, n_comp):
        new_mean = np.zeros((n_features, ))
        for j in range(floor(log(i_comp, 2))+1):
            if (1 << j) & i_comp:
                new_mean[j] = mean
        means.append(new_mean)

    beta = np.repeat(1., n_features + 1)

    if not noise:
        noise = 0
    else:
        noise = rng.chisquare(1)

    return n_features, pat_to_comp, probs, means, covs, beta, noise


def generate_toy_params_selfmasked_proba(n_features, prop_incomplete,
                                         missing_rate, prop_latent, lam,
                                         mean=0, noise=False,
                                         random_state=None):
    """Creates parameters for generating data with `generate_data_selfmasked`.

    Parameters
    ----------
    n_features: int
        The number of features desired.

    prop_incomplete: float
        The perccentage of features with missing entries.
        Should be between 0 and 1.

    missing_rate: int or array_like, shape (n_features, )
        The percentage of missing entries for each incomplete feature.
        It int, all features with missing values have the same missing rate.
        Entries should be between 0 and 1.

    prop_latent: float
        The number of latent factors used to generate the covariance matrix is
        prop_latent*n_feature. The less factots the higher the covariances.
        Should be between 0 and 1.

    lam: float
        Coefficient for the probit model which is used to add missing values.

    mean: float, optional, default 0
        Mean of the multivariate gaussian for all dimensions.

    noise: boolean, optional, default False
        Whether or not the response should be generated with noise

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    if missing_rate > 1 or missing_rate < 0:
        raise ValueError("missing_rate must be >= 0 and <= 1, got %s" %
                         missing_rate)

    if prop_incomplete > 1 or prop_incomplete < 0:
        raise ValueError("prop_incomplete must be >= 0 and <= 1, got %s" %
                         prop_incomplete)

    rng = check_random_state(random_state)

    # beta = rng.randn(n_features + 1)
    beta = np.repeat(1., n_features + 1)

    mean = np.repeat(mean, n_features)

    B = rng.randn(n_features, int(prop_latent*n_features))
    cov = B.dot(B.T) + np.diag(rng.uniform(low=0.1, size=n_features))

    n_incomplete_features = int(prop_incomplete*n_features)
    if isinstance(missing_rate, float):
        missing_rate = np.repeat(missing_rate, n_incomplete_features)

    # By default, missing values are incorporated in the first features.
    miss_index = np.arange(n_incomplete_features)
    lambda_0 = {}
    for i in miss_index:
        lambda_0[i] = (mean[i] - norm.ppf(missing_rate[i])*np.sqrt(
            1/lam**2+cov[i, i]))

    if not noise:
        noise = 0
    else:
        noise = rng.chisquare(1)

    return n_features, lam, lambda_0, mean, cov, beta, noise


def generate_data_mixture(n_sizes, data_params, random_state=None):
    """ Simulate Gaussian mixture data.

    Parameters
    ----------
    n_sizes: array_like
        The number of samples desired. Should be sorted in increasing order.

    data_params: tuple
        The parameters (means, covariances, ...) required to simulate
        Gaussian mixtures. These parametres can be obtained as the output of
        **generate_toy_params_mixture**

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    A generator that yields a sequence of datasets (X, y) with number of
    samples matching n_sizes. New samples are incrementally stacked to X and y
    so that larger datasets contain the samples of smaller ones.

    Usage
    -----
    for X, y in generate_data_mixture(n_sizes, p_sizes, data_params):
        print(X.shape, y.shape)
    """

    rng = check_random_state(random_state)
    n_features, ass, probs, means, covs, beta, noise = data_params

    X = np.empty((0, n_features))
    y = np.empty((0, ))
    current_size = 0

    for _, n_samples in enumerate(n_sizes):

        pattern_ids = rng.choice(2**n_features, p=probs,
                                 size=n_samples - current_size)
        current_M = [format(pat, 'b').zfill(n_features) for pat in pattern_ids]
        current_M = np.array(
            [np.array(list(s)).astype(int) for s in current_M])

        current_X = np.empty((n_samples-current_size, n_features))
        n_comp = len(means)
        for i_comp in range(n_comp):
            idx = np.where(ass[pattern_ids] == i_comp)[0]
            current_X[idx] = rng.multivariate_normal(
                mean=means[i_comp], cov=covs[i_comp],
                size=len(idx), check_valid='raise')

        current_y = beta[0] + current_X.dot(beta[1:]) + \
            noise * rng.randn(n_samples-current_size)

        np.putmask(current_X, current_M, np.nan)

        X = np.vstack((X, current_X))
        y = np.hstack((y, current_y))

        current_size = n_samples

        yield X, y


def generate_data_selfmasked_proba(n_sizes, data_params, random_state=None):
    """ Simulate Gaussian data with probit self masking

    Parameters
    ----------
    n_sizes: array_like
        The number of samples desired. Should be sorted in increasing order.

    data_params: tuple
        The parameters (means, covariances, ...) required to simulate
        Gaussian mixtures. These parametres can be obtained as the output of
        **generate_toy_params_selfmasked_proba**

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    A generator that yields a sequence of datasets (X, y) with number of
    samples matching n_sizes. New samples are incrementally stacked to X and y
    so that larger datasets contain the samples of smaller ones.

    Usage
    -----
    for X, y in generate_data_selfmasked_proba(n_sizes, p_sizes, data_params):
        print(X.shape, y.shape)
    """

    rng = check_random_state(random_state)

    n_features, lam, lambda_0, mean, cov, beta, noise = data_params

    X = np.empty((0, n_features))
    y = np.empty((0, ))
    current_size = 0

    for _, n_samples in enumerate(n_sizes):

        current_X = rng.multivariate_normal(
                mean=mean, cov=cov,
                size=n_samples-current_size,
                check_valid='raise')

        current_y = beta[0] + current_X.dot(beta[1:]) + \
            noise * rng.randn(n_samples-current_size)

        for j, l0 in lambda_0.items():
            X_j = current_X[:, j]
            prob = norm.cdf(lam*(X_j - l0))
            M_j = rng.binomial(n=1, p=prob, size=len(X_j))
            np.putmask(current_X[:, j], M_j, np.nan)

        X = np.vstack((X, current_X))
        y = np.hstack((y, current_y))

        current_size = n_samples

        yield X, y


class BayesPredictor():
    """The Bayes predictor for the Gaussian mixture model."""

    def __init__(self, data_params):
        self.data_params = data_params

    def fit(self, X, y):
        return self

    def predict(self, X):
        n_features, ass, probs, means, covs, beta, noise = self.data_params

        pred = []
        for x in X:
            # m = ''.join([str(mj) for mj in np.isnan(x).astype(int)])
            m = ''.join([str(mj) for mj in np.isnan(x).astype(int)])
            ind_m = int(m, 2)

            mu = means[ass[ind_m]]
            sigma = np.atleast_2d(covs[ass[ind_m]])

            obs = np.where(np.array(list(m)).astype(int) == 0)[0]
            mis = np.where(np.array(list(m)).astype(int) == 1)[0]

            predx = beta[0]
            if len(mis) > 0:
                predx += beta[mis + 1].dot(mu[mis])
            if len(obs) > 0:
                predx += beta[obs + 1].dot(x[obs])
            if len(obs) * len(mis) > 0:
                sigma_obs = sigma[np.ix_(obs, obs)]
                sigma_obs_inv = np.linalg.inv(sigma_obs)
                sigma_misobs = sigma[np.ix_(mis, obs)]

                predx += beta[mis + 1].dot(sigma_misobs).dot(
                    sigma_obs_inv).dot(x[obs] - mu[obs])

            pred.append(predx)

        return np.array(pred)


def bayes_rate_monte_carlo(data_params):
    """The Bayes risk computed based on repeated applications of the Bayes
    predictor"""
    reg = BayesPredictor(data_params)
    n_iter_mc = 30
    risk = 0.
    # var = 0.
    for _ in range(n_iter_mc):
        X, y = next(
            generate_data_mixture([10000], [data_params[0]], [data_params]))
        risk += np.mean((reg.predict(X) - y) ** 2)
        # var += np.mean((np.mean(y) - y) ** 2)
    risk /= n_iter_mc
    # var /= n_iter_mc

    # res = {'mse': float(risk), 'r2': float(1 - risk/var)}

    return float(risk)


bayes_rate_monte_carlo = memory.cache(bayes_rate_monte_carlo)


@memory.cache
def bayes_rate(data_params):
    """The Bayes risk computed based on the parameters of the model"""
    n_features, ass, probs, means, covs, beta, noise = data_params

    risk = noise ** 2
    for i in range(2**n_features):
        prob = probs[i]
        mu = means[ass[i]]
        sigma = np.atleast_2d(covs[ass[i]])

        m = bin(i).split('b')[1].zfill(n_features)

        obs = np.where(np.array(list(m)).astype(int) == 0)[0]
        mis = np.where(np.array(list(m)).astype(int) == 1)[0]

        factor = 0.
        if len(obs) == n_features:
            factor = 0.
        elif len(mis) == n_features:
            sigma_mis = sigma[np.ix_(mis, mis)]

            factor += beta[mis + 1].dot(sigma_mis).dot(beta[mis + 1])
            factor += (beta[mis + 1].dot(mu[mis])) ** 2

            sigma_obs = sigma[np.ix_(obs, obs)]
            sigma_obs_inv = np.linalg.inv(sigma_obs)
            sigma_misobs = sigma[np.ix_(mis, obs)]

            gamma_obs_0 = float(beta[mis + 1].dot(mu[mis]))
            gamma_obs = beta[mis + 1].dot(sigma_misobs).dot(sigma_obs_inv)

            factor += gamma_obs.dot(sigma_obs).dot(gamma_obs)
            factor += gamma_obs_0 ** 2
            factor += (gamma_obs.dot(mu[obs])) ** 2
            factor += 2 * gamma_obs_0 * gamma_obs.dot(mu[obs])
            factor -= 2 * gamma_obs.dot(sigma_misobs.T).dot(beta[mis + 1])
            factor -= 2 * gamma_obs_0 * beta[mis + 1].dot(mu[mis])
            factor -= 2 * ((gamma_obs.dot(mu[obs])) *
                           (beta[mis + 1].dot(mu[mis])))
        else:
            sigma_mis = sigma[np.ix_(mis, mis)]

            factor += beta[mis + 1].dot(sigma_mis).dot(beta[mis + 1])
            factor += (beta[mis + 1].dot(mu[mis])) ** 2

            sigma_obs = sigma[np.ix_(obs, obs)]
            sigma_obs_inv = np.linalg.inv(sigma_obs)
            sigma_misobs = sigma[np.ix_(mis, obs)]

            gamma_obs_0 = float(beta[mis + 1].dot(
                mu[mis] - sigma_misobs.dot(sigma_obs_inv).dot(mu[obs])))
            gamma_obs = beta[mis + 1].dot(sigma_misobs).dot(sigma_obs_inv)

            factor += gamma_obs.dot(sigma_obs).dot(gamma_obs)
            factor += gamma_obs_0 ** 2
            factor += (gamma_obs.dot(mu[obs])) ** 2
            factor += 2 * gamma_obs_0 * gamma_obs.dot(mu[obs])
            factor -= 2 * gamma_obs.dot(sigma_misobs.T).dot(beta[mis + 1])
            factor -= 2 * gamma_obs_0 * beta[mis + 1].dot(mu[mis])
            factor -= 2 * ((gamma_obs.dot(mu[obs])) *
                           (beta[mis + 1].dot(mu[mis])))

        risk += prob * factor

    return float(risk)


@memory.cache
def bayes_rate_r2(data_params):
    """The R2 Bayes rate computed based on the parameters of the model"""
    n_features, ass, probs, means, covs, beta, noise = data_params

    mean_of_means = np.zeros((n_features))

    for i in range(2**n_features):
        prob = probs[i]
        mu = means[ass[i]]
        mean_of_means += prob*mu

    risk = noise ** 2
    for i in range(2**n_features):
        prob = probs[i]
        mu = means[ass[i]]
        sigma = np.atleast_2d(covs[ass[i]])

        factor1 = beta[1:].dot(sigma).dot(beta[1:])

        sigma_mean = np.outer(mu - mean_of_means, mu - mean_of_means)
        factor2 = beta[1:].dot(sigma_mean).dot(beta[1:])

        risk += prob * (factor1 + factor2)

    br_mse = bayes_rate(data_params)

    br_r2 = 1 - br_mse/risk

    return float(br_r2)
