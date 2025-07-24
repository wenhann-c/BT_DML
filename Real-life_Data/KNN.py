import pandas as pd
import numpy as np
from sklearn import neighbors as nb
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
import pickle

def get_data():
            
    df_raw = pd.read_csv("data/401k.csv")
    variables_needed = ['e401', 'net_tfa', 'age', 'inc', 'fsize', 'educ', 'marr', 'twoearn', 'db', 'pira', 'hown']
    df = df_raw[variables_needed]

    return df['net_tfa'].values, df['e401'].values, df.drop(['e401', 'net_tfa'], axis=1).values

def knn_cv(y_data, d_data, x_data, cv=5):
    
    knn_model_treat = nb.KNeighborsRegressor()
    knn_model_control = nb.KNeighborsRegressor()

    param_grid = {

        'n_neighbors': [35,40,45,50,55],
        'p': [1,2,3,4]

    }
    
    grid_search_treat = GridSearchCV(estimator=knn_model_treat, param_grid=param_grid, cv=cv, n_jobs=1,
                                        scoring='neg_mean_squared_error', verbose=1)
    grid_search_control = GridSearchCV(estimator=knn_model_control, param_grid=param_grid, cv=cv, n_jobs=1,
                                        scoring='neg_mean_squared_error', verbose=1)
    
    knn_params_dict = {}

    x_treat = x_data[d_data == 1]
    y_treat = y_data[d_data == 1]
    x_control = x_data[d_data == 0]
    y_control = y_data[d_data == 0]

    grid_search_treat.fit(X=x_treat, y=y_treat)
    grid_search_control.fit(X=x_control, y=y_control)
    knn_params_dict['treat'] = grid_search_treat.best_params_
    knn_params_dict['control'] = grid_search_control.best_params_

    return knn_params_dict

y_data, d_data, x_data = get_data()
opt_params_knn = knn_cv(y_data, d_data, x_data)
print(f'Cross-validation done')

with open('opt_params_knn.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_knn, pickle_file)

def get_models(knn_params_dict):

    model_treat = nb.KNeighborsRegressor()
    model_treat.set_params(**knn_params_dict['treat'])

    model_control = nb.KNeighborsRegressor()
    model_control.set_params(**knn_params_dict['control'])

    return model_treat, model_control

def knn(y_data, d_data, x_data, model_treat, model_control, B=100, alpha=0.05):

    N = len(y_data)

    x_treat = x_data[d_data == 1]
    y_treat = y_data[d_data == 1]
    x_control = x_data[d_data == 0]
    y_control = y_data[d_data == 0]

    model_treat_ite = clone(model_treat)
    model_control_ite = clone(model_control)
    model_treat_ite.fit(X=x_treat, y=y_treat)
    model_control_ite.fit(X=x_control, y=y_control)

    mu1_hat = model_treat_ite.predict(x_data)
    mu0_hat = model_control_ite.predict(x_data)
    ite_hat = mu1_hat - mu0_hat
    ate_hat = np.mean(ite_hat)

    ate_bootstrap = []
    for _ in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        x_b = x_data[idx]
        d_b = d_data[idx]
        y_b = y_data[idx]

        x_b_treat = x_b[d_b == 1]
        y_b_treat = y_b[d_b == 1]
        x_b_control = x_b[d_b == 0]
        y_b_control = y_b[d_b == 0]

        # If there are not enough treated/control units in a bootstrap sample, skip
        if len(x_b_treat) < 10 or len(x_b_control) < 10:
            continue

        model_treat_b = clone(model_treat)
        model_control_b = clone(model_control)
        model_treat_b.fit(X=x_b_treat, y=y_b_treat)
        model_control_b.fit(X=x_b_control, y=y_b_control)

        mu1_b = model_treat_b.predict(x_b)
        mu0_b = model_control_b.predict(x_b)
        ite_b = mu1_b - mu0_b
        ate_b = np.mean(ite_b)
        ate_bootstrap.append(ate_b)

    ate_bootstrap = np.array(ate_bootstrap)
    se_hat = np.std(ate_bootstrap, ddof=1)
    lower = np.percentile(ate_bootstrap, 100 * alpha / 2)
    upper = np.percentile(ate_bootstrap, 100 * (1 - alpha / 2))
    CI = np.array([lower, upper])

    results_dict = {
    'ate_estimate': ate_hat,
    'se_estimate': se_hat,
    'CI': CI,
    }

    return results_dict

model_treat, model_control = get_models(opt_params_knn)
results_dict = knn(y_data, d_data, x_data, model_treat, model_control)
print(f'KNN done')

with open('results_knn.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)