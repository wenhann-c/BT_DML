import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from data_generation import get_data

def xgb_cv(y_data, d_data, x_data, cv=5):
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=0)

    param_grid = {

        'n_estimators': [50, 100, 150],
        'max_depth': [2],
        'subsample': [0.5],
        'learning_rate': [0.05, 0.1, 0.15],
        'reg_lambda': [0.01, 0.1, 1]
    }
    
    grid_search_l = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    grid_search_m = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    xgb_params_dict = {}

    grid_search_l.fit(X=x_data, y=y_data)
    xgb_params_dict['l'] = grid_search_l.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    xgb_params_dict['m'] = grid_search_m.best_params_

    return xgb_params_dict

sample_sizes = [500, 1000, 2000]
n_MC = 3
opt_params_xgb = {}

for N in sample_sizes:
    rng = np.random.default_rng(seed=19)
    opt_params_xgb_N = {}
    
    for j in range(n_MC): 
        y_data, d_data, x_data = get_data(N, rng)
        opt_params_xgb_N[j] = xgb_cv(y_data, d_data, x_data)

    opt_params_xgb[N] = opt_params_xgb_N
    print(f'Cross-validation done for N={N}')

with open('opt_params_xgb.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_xgb, pickle_file)