'''
https://github.com/catboost/tutorials/blob/master/classification/classification_with_parameter_tuning_tutorial.ipynb
https://github.com/hyperopt/hyperopt/wiki/FMin

See function train_best_model
'''
import numpy as np
import catboost as cb
import hyperopt
import os, sys


class MyObjective(object):
    def __init__(self, dataset, const_params, fold_count, eval_metric='R2'):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._evaluated_count = 0
        self._eval_metric = eval_metric  # 'R2' for regression, 'Accuracy' for classification

    def _to_catboost_params(self, hyper_params):
        return hyper_params
        # return {
        #     'learning_rate': hyper_params['learning_rate'],
        #     'depth': hyper_params['depth'],
        #     'l2_leaf_reg': hyper_params['l2_leaf_reg']}

    # hyperopt optimizes an objective using `__call__` method (e.g. by doing
    # `foo(hyper_params)`), so we provide one
    def __call__(self, hyper_params):
        # join hyper-parameters provided by hyperopt with hyper-parameters
        # provided by the user
        params = self._to_catboost_params(hyper_params)
        params.update(self._const_params)

        print('evaluating params={}'.format(params), file=sys.stdout)
        sys.stdout.flush()

        # we use cross-validation for objective evaluation, to avoid overfitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._fold_count,
            partition_random_seed=42,
            verbose=False)

        # scores returns a dictionary with mean and std (per-fold) of metric
        # value for each cv iteration, we choose minimal value of objective
        # mean (though it will be better to choose minimal value among all folds)
        # because noise is additive
        metric = f'test-{self._eval_metric}-mean'  # metric to be retrieved from CV
        max_mean_score = np.max(scores[metric])
        print('evaluated score={}'.format(max_mean_score), file=sys.stdout)

        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)

        # negate because hyperopt minimizes the objective
        return {'loss': -max_mean_score, 'status': hyperopt.STATUS_OK}


def find_best_hyper_params(dataset,
                           const_params, parameter_space,
                           fold_count,
                           eval_metric,
                           max_evals=100,
                           ):
    # we are going to optimize these three parameters, though there are a lot more of them (see CatBoost docs)
    # parameter_space = {
    #     'learning_rate': hyperopt.hp.uniform('learning_rate', 0.1, 1.0),
    #     'depth': hyperopt.hp.randint('depth', 7),
    #     'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1, 10)
    #     }
    objective = MyObjective(dataset=dataset,
                            const_params=const_params,
                            fold_count=fold_count,
                            eval_metric=eval_metric)
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.rand.suggest,
        max_evals=max_evals,
        rstate=np.random.RandomState(seed=42)
    )
    return best


def train_best_model(X, y,
                     const_params, parameter_space,
                     fold_count,
                     eval_metric,
                     max_evals=100, use_default=False):
    '''
    Example
    have_gpu = True
    # skip hyper-parameter optimization and just use provided optimal parameters
    use_optimal_pretrained_params = None
    # number of iterations of hyper-parameter search
    hyperopt_iterations = 5

    # Constant params
    const_params = dict({
        'task_type': 'GPU' if have_gpu else 'CPU',
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        # 'custom_metric': ['AUC'],
        'iterations': 8000,
        'random_seed': 42})
    # Search param space
    parameter_space = {
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.1, 1.0),
        'depth': hyperopt.hp.randint('depth', 7),
        'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1, 10)
    }

    model, params = train_best_model(
        X=x_train_sp, y=y_train_sp[y_name_bin],
        const_params=const_params,
        parameter_space=parameter_space,
        max_evals=hyperopt_iterations,
        fold_count=3,
        eval_metric='R2',
        use_default=use_optimal_pretrained_params)
    print('best params are {}'.format(params), file=sys.stdout)

    :param X: train X
    :param y: labels
    :param const_params: constant params (dict)
    :param parameter_space: search params (dict)
    :param max_evals: max nb of iterations in hyperopt search
    :param fold_count: Cross validation fold count,
    :param eval_metric: metric name used for optimization ('R2', 'Accuracy', ...).
    :param use_default: dict, don't search just use it as the best params
    :return:
    - The best model (catboost)
    - The best params (dict)
    '''
    # convert pandas.DataFrame to catboost.Pool to avoid converting it on each
    # iteration of hyper-parameters optimization
    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes != np.float)[0])

    if use_default:
        # pretrained optimal parameters
        best = use_default
    else:
        best = find_best_hyper_params(dataset=dataset,
                                      const_params=const_params, parameter_space=parameter_space,
                                      fold_count=fold_count,
                                      eval_metric=eval_metric,
                                      max_evals=max_evals)

    # merge subset of hyper-parameters provided by hyperopt with hyper-parameters
    # provided by the user
    hyper_params = best.copy()
    hyper_params.update(const_params)

    # drop `use_best_model` because we are going to use entire dataset for
    # training of the final model
    hyper_params.pop('use_best_model', None)

    model = cb.CatBoostClassifier(**hyper_params)
    model.fit(dataset, verbose=False)

    return model, hyper_params


if __name__ == '__main__':
    pass