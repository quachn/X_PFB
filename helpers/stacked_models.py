# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
import datetime
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import time
import copy

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

import catboost
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, PassiveAggressiveRegressor
import sklearn.linear_model as linear_model
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

SEED=42

# --- Average K models trained with K-folds by cloning from a same model
# --- The models are trained with early-stopping on the validation-fold => they might have differnt iterations
class KFoldsModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, model_ori, k, seed=42):
        self.model_ori = model_ori
        self.k = k
        self.seed=seed
        self.models_ = []
        self.R2s_validation = []

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = []
        self.R2s_validation = []
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        k = 1
        for train_index, holdout_index in kfold.split(X, y):
            t0 = time.time()
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_val, y_val = X.iloc[holdout_index], y.iloc[holdout_index]
            m = clone(self.model_ori)
            eval_set = {}
            if isinstance(m, catboost.CatBoostRegressor):
                eval_set = {'eval_set': (X_val, y_val)}
            m.fit(X_train, y_train, **eval_set)
            self.models_.append(m)
            self.R2s_validation.append(r2_score(y_true=y_val, y_pred=m.predict(X_val)))

            print(f"Fold {k}. R2_validation: {self.R2s_validation[-1]}. Time: {time.time()-t0}s")
            k += 1

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

# --- Averaged base models class
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

# --- Stacking averaged Models Class
# https://www.kaggle.com/getting-started/18153#post103381

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)