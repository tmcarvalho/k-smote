"""Predictive performance
This script will test the predictive performance of the data sets.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
# %% evaluate a model
def evaluate_model(x_train, x_test, y_train, y_test):
    """Evaluatation

    Args:
        x_train (pd.DataFrame): dataframe for train
        x_test (pd.DataFrame): dataframe for test
        y_train (np.int64): target variable for train
        y_test (np.int64): target variable for test
    Returns:
        tuple: dictionary with validation, train and test results
    """
    
    seed = np.random.seed(1234)

    # initiate models
    rf = RandomForestClassifier(random_state=seed)
    booster = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=seed)
    reg = LogisticRegression(random_state=seed)

    # set parameterisation
    param1 = {}
    param1['classifier__n_estimators'] = [100, 250, 500]
    param1['classifier__max_depth'] = [4, 7, 10]
    param1['classifier'] = [rf]

    param2 = {}
    param2['classifier__n_estimators'] = [100, 250, 500]
    param2['classifier__max_depth'] = [4, 7, 10]
    param2['classifier__learning_rate'] = [0.1, 0.01]
    param2['classifier'] = [booster]

    param3 = {}
    param3['classifier__C'] = np.logspace(-4, 4, 3)
    param3['classifier__max_iter'] = [1000000, 100000000]
    param3['classifier'] = [reg]

    # define metric functions
    scoring = {
        'gmean': make_scorer(geometric_mean_score),
        'acc': 'accuracy',
        'bal_acc': 'balanced_accuracy',
        'f1': 'f1',
        'f1_weighted': 'f1_weighted',
        'roc_auc_curve': make_scorer(roc_auc_score, max_fpr=0.001, needs_proba=True),
        }

    pipeline = Pipeline([('classifier', rf)])
    params = [param1, param2, param3]

    print("Start modeling with CV")
    # Train the grid search model
    gs = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='acc',
        return_train_score=True,
        n_jobs=-1).fit(x_train, y_train)

    # validation = {}
    score_cv = {
    'model':[],
    'test_accuracy': [], 'test_f1_weighted':[], 'test_gmean':[], 'test_roc_auc':[]
    }
    # Store results from grid search
    validation = pd.DataFrame(gs.cv_results_)
    validation['model'] = validation['param_classifier']
    validation['model'] = validation['model'].apply(lambda x: 'Random Forest' if 'RandomForest' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'XGBoost' if 'XGB' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Logistic Regression' if 'Logistic' in str(x) else x)

    validation_head = validation.sort_values(['rank_test_roc_auc_curve'],ascending=False).groupby('model').head(1).reset_index(drop=True)
    
    # get best models for prediction on test
    clf_1st_best = gs.best_estimator_.set_params(**validation_head.loc[0, 'params']).fit(x_train, y_train)
    clf_1st = clf_1st_best.predict(x_test)
    clf_2nd_best = gs.best_estimator_.set_params(**validation_head.loc[1, 'params']).fit(x_train, y_train)
    clf_2nd = clf_2nd_best.predict(x_test)
    clf_3rd_best = gs.best_estimator_.set_params(**validation_head.loc[2, 'params']).fit(x_train, y_train)
    clf_3rd = clf_3rd_best.predict(x_test)

    for i, clf in enumerate([clf_1st, clf_2nd, clf_3rd]):
        score_cv['model'].append(validation_head.loc[i, 'model'])
        score_cv['test_accuracy'].append(accuracy_score(y_test, clf))
        score_cv['test_f1_weighted'].append(f1_score(y_test, clf, average='weighted'))
        score_cv['test_gmean'].append(geometric_mean_score(y_test, clf))
        score_cv['test_roc_auc'].append(roc_auc_score(y_test, clf))

    ##################### OUT OF SAMPLE ##########################
    # apply best cv result in all training data (without CV - out of sample)
    # Train the grid search model
    print("Start modeling without CV")
    gs_outofsample = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=[(slice(None), slice(None))],
        scoring=scoring,
        refit='acc',
        return_train_score=True,
        n_jobs=-1).fit(x_train, y_train)

    outofsample_val = pd.DataFrame(gs_outofsample.cv_results_)
    outofsample_val['model'] = outofsample_val['param_classifier']
    outofsample_val['model'] = outofsample_val['model'].apply(lambda x: 'Random Forest' if 'RandomForest' in str(x) else x)
    outofsample_val['model'] = outofsample_val['model'].apply(lambda x: 'XGBoost' if 'XGB' in str(x) else x)
    outofsample_val['model'] = outofsample_val['model'].apply(lambda x: 'Logistic Regression' if 'Logistic' in str(x) else x)

    outofsample_val_head = outofsample_val.sort_values(['rank_test_roc_auc_curve'],ascending=False).groupby('model').head(1).reset_index(drop=True)

    # Predict on train data with best params
    clf_1st_best_out = gs_outofsample.best_estimator_.set_params(**outofsample_val_head.loc[0, 'params']).fit(x_train, y_train)
    clf_1st_out = clf_1st_best_out.predict(x_test)
    clf_2nd_best_out = gs_outofsample.best_estimator_.set_params(**outofsample_val_head.loc[1, 'params']).fit(x_train, y_train)
    clf_2nd_out = clf_2nd_best_out.predict(x_test)
    clf_3rd_best_out = gs_outofsample.best_estimator_.set_params(**outofsample_val_head.loc[2, 'params']).fit(x_train, y_train)
    clf_3rd_out = clf_3rd_best_out.predict(x_test)

    score_outofsample_val = {
        'model':[],
        'test_accuracy': [], 'test_f1_weighted':[], 'test_gmean':[], 'test_roc_auc':[]
    }
    # Store predicted results in out of sample
    for i, clf in enumerate([clf_1st_out, clf_2nd_out, clf_3rd_out]):
        score_outofsample_val['model'].append(outofsample_val_head.loc[i, 'model'])
        score_outofsample_val['test_accuracy'].append(accuracy_score(y_test, clf))
        score_outofsample_val['test_f1_weighted'].append(f1_score(y_test, clf, average='weighted'))
        score_outofsample_val['test_gmean'].append(geometric_mean_score(y_test, clf))
        score_outofsample_val['test_roc_auc'].append(roc_auc_score(y_test, clf))
    
    score_cv = pd.DataFrame(score_cv)
    score_outofsample_val = pd.DataFrame(score_outofsample_val)

    return [validation, score_cv, outofsample_val, score_outofsample_val]
