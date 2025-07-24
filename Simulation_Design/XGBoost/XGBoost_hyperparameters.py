import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from data_generation import get_data

def xgb_cv(y_data, d_data, x_data, cv=5):
    
    xgb_model_l = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    xgb_model_m = xgb.XGBClassifier(objective='binary:logistic', seed=42)

    param_grid_l = {

        'n_estimators': [300,325,350],
        'max_depth': [2],
        'subsample': [0.8],
        'learning_rate': [0.05,0.1,0.15],
        'reg_lambda': [5,10,15]
    }
    
    param_grid_m = param_grid_l.copy()
    param_grid_m['n_estimators'] = [75,100,125]
    
    grid_search_l = GridSearchCV(estimator=xgb_model_l, param_grid=param_grid_l, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    grid_search_m = GridSearchCV(estimator=xgb_model_m, param_grid=param_grid_m, cv=cv, n_jobs=1,
                                 scoring='neg_brier_score', verbose=1)
    
    xgb_params_dict = {}

    grid_search_l.fit(X=x_data, y=y_data)
    xgb_params_dict['l'] = grid_search_l.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    xgb_params_dict['m'] = grid_search_m.best_params_

    return xgb_params_dict

N = 1000
n_trial = 10
opt_params_xgb = {}
sum_l = {}
sum_m = {}
rng = np.random.default_rng(seed=42)
opt_params_xgb_N = {}

for j in range(n_trial):
    y_data, d_data, x_data = get_data(N, rng)
    opt_params_xgb_N[j] = xgb_cv(y_data, d_data, x_data)

for i in range(n_trial):
    
    for key, value in opt_params_xgb_N[i]['l'].items():
        if key in sum_l:
            sum_l[key] += value/n_trial
        else:
            sum_l[key] = value/n_trial

    for key, value in opt_params_xgb_N[i]['m'].items():
        if key in sum_m:
            sum_m[key] += value/n_trial
        else:
            sum_m[key] = value/n_trial

keys_to_round = ['n_estimators', 'max_depth', 'reg_lambda']
keys_dec_to_round = ['learning_rate', 'subsample']

for key in keys_to_round:
    sum_l[key] = round(sum_l[key])
    sum_m[key] = round(sum_m[key])

for key in keys_dec_to_round:
    sum_l[key] = round(sum_l[key],2)
    sum_m[key] = round(sum_m[key],2)

opt_params_xgb['l'] = sum_l
opt_params_xgb['m'] = sum_m

print(f'Cross-validation done for N={N}')

with open('opt_params_xgb.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_xgb, pickle_file)