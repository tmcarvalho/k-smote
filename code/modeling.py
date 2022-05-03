"""Predictive performance
This script will test the predictive performance of the data sets.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import make_scorer, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# %% evaluate a model
def evaluate_model(
    x: pd.DataFrame,
    y: np.int64,
    neirest_neighbours: np.int64,
    privacy_risks: list,
    key_vars):
    """Evaluatation

    Args:
        x (pd.DataFrame): dataframe to train
        y (np.int64): target variable
        neirest_neighbours (np.int64): neirest neighbours from smote
        privacy_risk (list): record linkage result with 50%, 75% and 100% score

    Returns:
        tuple: dictionary with validation and test results
    """

    # split data 80/20
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    seed = np.random.seed(1234)

    n_feat = x_train.shape[1]

    # set parameters
    param_grid_rf = {
        'n_estimators': [100, 250, 500],
        'max_depth': [4, 6, 8, 10]
    }
    param_grid_bc = {
        'n_estimators': [100, 250, 500],
        'max_features': [0.7]
    }
    param_grid_xgb = {
        'n_estimators': [100, 250, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.001, 0.0001]
    }
    param_grid_lreg = {'C': np.logspace(-4, 4, 3),
                       'max_iter': [10000, 100000]
                       }
    param_grid_nnet = {'hidden_layer_sizes': [[n_feat], [n_feat // 2],
                                            [int(n_feat * (2 / 3))],
                                            [n_feat, n_feat // 2],
                                            [n_feat, int(n_feat * (2 / 3))],
                                            [n_feat // 2, int(n_feat * (2 / 3))],
                                            [n_feat, n_feat // 2, int(n_feat * (2 / 3))]
                                            ],
                       'alpha': [1e-3, 1e-4],
                       'max_iter': [10000, 100000]
                       }
    # define metric functions
    scoring = {
        'gmean': make_scorer(geometric_mean_score),
        'acc': 'accuracy',
        'bal_acc': 'balanced_accuracy',
        'f1': 'f1',
        'f1_weighted': 'f1_weighted'}

    # create the parameter grid
    gs_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=seed),
        param_grid=param_grid_rf,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='f1_weighted',
        return_train_score=True)
    gs_bc = GridSearchCV(
        estimator=BaggingClassifier(random_state=seed),
        param_grid=param_grid_bc,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='f1_weighted',
        return_train_score=True)
    gs_xgb = GridSearchCV(
        estimator=XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=seed),
        param_grid=param_grid_xgb,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='f1_weighted',
        return_train_score=True)
    gs_lreg = GridSearchCV(
        estimator=LogisticRegression(random_state=seed),
        param_grid=param_grid_lreg,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='f1_weighted',
        return_train_score=True)
    gs_nnet = GridSearchCV(
        estimator=MLPClassifier(random_state=seed),
        param_grid=param_grid_nnet,
        cv=RepeatedKFold(n_splits=5, n_repeats=2),
        scoring=scoring,
        refit='f1_weighted',
        return_train_score=True)

    # List of pipelines for ease of iteration
    grids = [gs_rf, gs_bc, gs_xgb, gs_lreg, gs_nnet]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {
        0: 'Random Forest',
        1: 'Bagging',
        2: 'Boosting',
        3: 'Logistic Regression',
        4: 'Neural Network'}

    # Fit the grid search objects
    # print('Performing model optimizations...')

    validation = test = {}
    for idx, gs in enumerate(grids):
        print(f'\nEstimator: {grid_dict[idx]}')
        # Performing cross validation to tune parameters for best model fit
        gs.fit(x_train, y_train)
        # Best params
        # print(f'Best params: {gs.best_params_}')
        # Best training data accuracy
        # print(f'Best training accuracy: {gs.best_score_}')
        # Store results from grid search
        validation['cv_results_' + str(grid_dict[idx])] = gs.cv_results_
        validation['cv_results_' + str(grid_dict[idx])]['neirest_neighbours']\
            = neirest_neighbours
        validation['cv_results_' + str(grid_dict[idx])]['privacy_risk_50']\
            = privacy_risks[0]
        validation['cv_results_' + str(grid_dict[idx])]['privacy_risk_75']\
            = privacy_risks[1]    
        validation['cv_results_' + str(grid_dict[idx])]['privacy_risk_100']\
            = privacy_risks[2]
        validation['cv_results_' + str(grid_dict[idx])]['key_vars']\
            = key_vars    

        # Predict on test data with best params
        y_pred = gs.predict(x_test)
        # Test data accuracy of model with best params
        # print(f'Test set accuracy score for best params:
        # {f1_score(y_test, y_pred, average='weighted')}')
        # Store results from grid search
        test[str(grid_dict[idx])] = f1_score(y_test, y_pred, average='weighted')
        test['neirest_neighbours'] = neirest_neighbours
        test['privacy_risk_50'] = privacy_risks[0]
        test['privacy_risk_75'] = privacy_risks[1]
        test['privacy_risk_100'] = privacy_risks[2]
        test['key_vars'] = key_vars

    return validation, test
