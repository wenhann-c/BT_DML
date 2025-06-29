import numpy as np
from sklearn import ensemble as ens
import pickle
from sklearn.model_selection import GridSearchCV
from data_generation import get_data

def rf_cv(y_data, d_data, x_data, cv=5):
    
    rf_model = ens.RandomForestRegressor(random_state=42, max_features='sqrt')

    param_grid = {

        'n_estimators': [100,120,140],
        'max_depth': [15,25],
        'min_samples_leaf' : [2],
        'min_samples_split' : [3]
    }
    
    grid_search_l = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    grid_search_m = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    rf_params_dict = {}

    grid_search_l.fit(X=x_data, y=y_data)
    rf_params_dict['l'] = grid_search_l.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    rf_params_dict['m'] = grid_search_m.best_params_

    return rf_params_dict

N = 10000
n_trial = 10
opt_params_rf = {}
sum_l = {}
sum_m = {}
rng = np.random.default_rng(seed=42)
opt_params_rf_N = {}
    
for j in range(n_trial):
    y_data, d_data, x_data = get_data(N, rng)
    opt_params_rf_N[j] = rf_cv(y_data, d_data, x_data)

for i in range(n_trial):
    
    for key, value in opt_params_rf_N[i]['l'].items():
        if key in sum_l:
            sum_l[key] += value/n_trial
        else:
            sum_l[key] = value/n_trial

    for key, value in opt_params_rf_N[i]['m'].items():
        if key in sum_m:
            sum_m[key] += value/n_trial
        else:
            sum_m[key] = value/n_trial

keys_to_round = ['n_estimators','max_depth','min_samples_leaf','min_samples_split']

for key in keys_to_round:
    sum_l[key] = round(sum_l[key])
    sum_m[key] = round(sum_m[key])

opt_params_rf['l'] = sum_l
opt_params_rf['m'] = sum_m

print(f'Cross-validation done for N={N}')

with open('opt_params_rf.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_rf, pickle_file)