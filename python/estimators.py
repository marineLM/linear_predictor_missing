"""This file contains the estimators:
    - ConstantImputedLR
    - ExpandedLR
    - EMLR
    - ConstantImputedMLPR
    - MICELR
"""
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
norm = importr("norm")

numpy2ri.activate()
pandas2ri.activate()


class ConstantImputedLR():
    def __init__(self):
        self._reg = LinearRegression()

    def transform(self, X):
        T = X.copy()
        M = np.isnan(T)
        np.putmask(T, M, 0)
        T = np.hstack((T, M))
        return T

    def fit(self, X, y):
        T = self.transform(X)
        self._reg.fit(T, y)
        return self

    def predict(self, X):
        T = self.transform(X)
        return self._reg.predict(T)


class ExpandedLR():
    def __init__(self):
        """The intercept is in the matrix."""
        self._reg = RidgeCV(
            alphas=[1e-3, 1, 1e3], fit_intercept=True, normalize=False, cv=3)
        self._scaler = StandardScaler()

    def fit_transformer(self, X):
        _, n_features = X.shape
        # all possible patterns
        self.patterns = np.array(
                [list(bin(k).split('b')[1].zfill(n_features))
                 for k in range(2**n_features)]
                ).astype(int)

    def transform(self, X):
        n_samples, n_features = X.shape
        M = np.isnan(X)

        p_expanded = n_features*2**(n_features-1) + 2**n_features
        W = np.zeros((n_samples, p_expanded))

        current_j = 0
        for patt in self.patterns:
            ind_features = np.where(patt == 0)[0]
            ind_samples = np.where(np.all(M == patt, axis=1))[0]

            new_j = current_j+np.sum(patt == 0)+1
            ind_features_exp = np.arange(current_j+1, new_j)

            W[np.ix_(ind_samples, ind_features_exp)] = X[
                np.ix_(ind_samples, ind_features)]
            W[ind_samples, current_j] = 1
            current_j = new_j

        return W

    def fit(self, X, y):
        self.fit_transformer(X)
        T = self.transform(X)
        T = self._scaler.fit_transform(T)
        self._reg.fit(T, y)
        return self

    def predict(self, X):
        T = self.transform(X)
        T = self._scaler.transform(T)
        print(self._reg.alpha_)
        return self._reg.predict(T)


class EMLR(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        Z = np.hstack((y[:, np.newaxis], X))
        s = norm.prelim_norm(Z)
        thetahat = norm.em_norm(s, showits=False,
                                criterion=np.sqrt(np.finfo(float).eps))
        parameters = norm.getparam_norm(s, thetahat)
        self.mu_joint = np.array(parameters[0])
        self.Sigma_joint = np.array(parameters[1])

    def predict(self, X):
        pred = np.empty(X.shape[0])
        for i, x in enumerate(X):
            indices = np.where(~np.isnan(x))[0] + 1
            x_obs = x[~np.isnan(x)]

            mu_X = self.mu_joint[indices]
            Sigma_X = self.Sigma_joint[np.ix_(indices, indices)]
            mu_y = self.mu_joint[0]
            Sigma_yX = self.Sigma_joint[0, indices]

            if len(indices) == 0:
                pred[i] = mu_y
            elif len(indices) == 1:
                beta = (
                    mu_y - Sigma_yX * mu_X / Sigma_X,
                    Sigma_yX / Sigma_X)
                pred[i] = beta[0] + beta[1] * x_obs
            else:
                beta = (
                    mu_y - Sigma_yX.dot(np.linalg.inv(Sigma_X)).dot(mu_X),
                    Sigma_yX.dot(np.linalg.inv(Sigma_X)))
                pred[i] = beta[0] + beta[1].dot(x_obs)
        return pred


class ConstantImputedMLPR():
    def __init__(self, est_params={}):
        self.imputation = est_params['imputation']
        self.mask = est_params['mask']
        del est_params['mask'], est_params['imputation']

        est = MLPRegressor(**est_params)
        self._reg = GridSearchCV(
            est, param_grid={'alpha': [1e-1, 1e-2, 1e-4]}, cv=3)
        self._scaler = StandardScaler()

    def transform(self, X):
        T = X.copy()
        M = np.isnan(T)
        np.putmask(T, M, self.imputation)
        if self.mask:
            T = np.hstack((T, M))
        return T

    def fit(self, X, y):
        T = self.transform(X)
        T = self._scaler.fit_transform(T)
        self._reg.fit(T, y)
        return self

    def predict(self, X):
        T = self.transform(X)
        T = self._scaler.transform(T)
        return self._reg.predict(T)


class MICELR():
    def __init__(self):
        self._reg = LinearRegression()
        self._imp = IterativeImputer(random_state=0)

    def fit(self, X, y):
        T = self._imp.fit_transform(X)
        self._reg.fit(T, y)
        return self

    def predict(self, X):
        T = self._imp.transform(X)
        return self._reg.predict(T)
