from dataclasses import dataclass
from typing import Any
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import catboost
import statsmodels.api as sm

@dataclass
class Result:
    name: str
    y_name: str
    mse_train: float = None
    R2_train: float = None
    mse_test: float = None
    R2_test: float = None
    fit_time: float = None
    model: Any = None
    def _to_list(self):
        return [getattr(self, att) for att in self._get_att_names()]

    def _get_att_names(self):
        return 'name y_name mse_train R2_train mse_test R2_test fit_time model'.split(' ')


def run_model(model, x_train, y_train, x_test, y_true):
    t0 = time.time()
    # if isinstance(model, sm.OLS):
    #     res = model.fit()
    # el
    if isinstance(model, catboost.CatBoostRegressor):
        fit_params = {'eval_set': (x_test, y_true)
                      }
        model.fit(x_train, y_train, **fit_params)
    elif isinstance(model, sm.OLS) or isinstance(model, sm.RLM):
        ols_res = model.fit()
    else:
        model.fit(x_train, y_train)
    fit_time = time.time() - t0

    if isinstance(model, sm.OLS) or isinstance(model, sm.RLM):
        model = ols_res
    y_pred_train = model.predict(x_train)
    mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    r2_train = r2_score(y_true=y_train, y_pred=y_pred_train)

    y_pred_test = model.predict(x_test)
    mse_test = mean_squared_error(y_true=y_true, y_pred=y_pred_test)
    r2_test = r2_score(y_true=y_true, y_pred=y_pred_test)
    return {
        'mse_train': mse_test, 'R2_train': r2_train,
        'mse_test': mse_test, 'R2_test': r2_test,
        'fit_time': fit_time,
        'model': model}


def run_new_model(name,
                  model, y_name,
                  x_train, y_train,
                  x_test, y_test,
                  results=[]):
    """
    Runs the given model and adds the new result into results.
    """

    res = run_model(model=model,
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_true=y_test)
    new_entry = Result(name=name, y_name=y_name, **res)
    # print(new_entry)
    results.append(new_entry)
    # Flatten results as dataframe
    results_df = flatten_results(results)
    return results, results_df


def flatten_results(results):
    return pd.DataFrame([r._to_list() for r in results], columns=results[0]._get_att_names())

def results_df_to_list(res_df):
    res_list = []
    for ix, row in res_df.iterrows():
        name_vals = dict(row)
        res_list.append(Result(**name_vals))

    return res_list

def save_results(id, results):
    # save_results('13.catboost.tuned', ALL_RESULTS)
    pickle.dump(results, open(r'c:\zzz\ParisFireBrigade\ALL_RESULTS.{id}.pickle'.format(id=id), 'wb'))
    res_df = flatten_results(results)
    res_df.to_csv('c:\zzz\ParisFireBrigade\ALL_RESULTS_DF.{id}.csv'.format(id=id))


def load_results(id):
    res = pickle.load(open(r'c:\zzz\ParisFireBrigade\ALL_RESULTS.{id}.pickle'.format(id=id), 'rb'))
    res_df = flatten_results(res)
    return res, res_df

# Makes residuals dataframe
def make_residuals_df(model, x_test, y_true):
  res_df = pd.DataFrame(
    {'y_true':y_true,
     'y_pred': model.predict(x_test)
    })
  res_df['resid'] = res_df.y_pred - res_df.y_true
  resid_stddev = res_df['resid'].std()
  res_df['resid_standardized'] = res_df['resid']/resid_stddev
  res_df = pd.concat([res_df, x_test], axis=1)
  return res_df

def r2_score_all(models,
                 x_test_sp_list,
                 y_test_sp):
    y_names = y_test_sp.columns[-3:].tolist()
    res_dfs = [None] * 3  # List of residual dataframe
    r2s = []
    # R2 of Y0 and Y1
    for i in range(2):
        x_test_sp = x_test_sp_list[i]
        res_dfs[i] = make_residuals_df(model=models[i],
                                       x_test=x_test_sp,
                                       y_true=y_test_sp[y_names[i]])
        r2 = r2_score(y_true=res_dfs[i]['y_true'], y_pred=res_dfs[i]['y_pred'])
        r2s.append(r2)
    # R2 for Y2
    y_name_3 = y_names[-1]
    y_pred_3 = res_dfs[0]['y_pred'] + res_dfs[1]['y_pred']
    r2 = r2_score(y_true=y_test_sp[y_name_3], y_pred=y_pred_3)
    r2s.append(r2)

    for i in range(3):
      print(f'R2 of {y_names[i]}: {r2s[i]}')
    # Mean of R2(Y0) and R2(y1), the metric used for submission ranking
    r2 = np.mean(r2s[:-1])
    print('Mean R2 (Y0, Y1):', r2)
    r2s.append(r2)
    return r2s, res_dfs

def create_submit_csv(models, data_dir, feats_4y0, data_prep_dir='data/prepared/'):
    x_test = pd.read_csv(data_prep_dir + 'x_test.csv.zip', compression='zip')
    x_test_4y0 = x_test[feats_4y0]
    x_test_ori = pd.read_csv(data_dir + 'x_test.csv')
    # models = [ALL_RESULTS_FINAL[0].model, ALL_RESULTS_FINAL[1].model]
    # Create a submission file
    submit = pd.DataFrame([], columns=['emergency vehicle selection',
                                       'delta selection-departure',
                                       'delta departure-presentation',
                                       'delta selection-presentation'])
    submit['emergency vehicle selection'] = x_test_ori['emergency vehicle selection']
    # submit['delta selection-departure'] = y_train['delta selection-departure'].median()
    submit['delta selection-departure'] = models[0].predict(x_test_4y0)
    submit['delta departure-presentation'] = models[1].predict(x_test)
    submit['delta selection-presentation'] = submit['delta selection-departure'] + submit[
        'delta departure-presentation']
    submit.set_index('emergency vehicle selection', inplace=True)
    return submit