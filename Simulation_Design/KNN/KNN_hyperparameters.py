import numpy as np
from sklearn import neighbors as nb
import pickle
from sklearn.model_selection import GridSearchCV
from data_generation import get_data

def knn_cv(y_data, d_data, x_data, cv=5):
    
    knn_model_treat = nb.KNeighborsRegressor()
    knn_model_control = nb.KNeighborsRegressor()

    param_grid = {

        'n_neighbors': [1,3,5],
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

N = 1000
n_trial = 10
opt_params_knn = {}
sum_t = {}
sum_c = {}
rng = np.random.default_rng(seed=42)
opt_params_knn_N = {}
    
for j in range(n_trial):
    y_data, d_data, x_data = get_data(N, rng)
    opt_params_knn_N[j] = knn_cv(y_data, d_data, x_data)

for i in range(n_trial):
    
    for key, value in opt_params_knn_N[i]['treat'].items():
        if key in sum_t:
            sum_t[key] += value/n_trial
        else:
            sum_t[key] = value/n_trial

    for key, value in opt_params_knn_N[i]['control'].items():
        if key in sum_c:
            sum_c[key] += value/n_trial
        else:
            sum_c[key] = value/n_trial

keys_to_round = ['n_neighbors','p']

for key in keys_to_round:
    sum_t[key] = round(sum_t[key])
    sum_c[key] = round(sum_c[key])

opt_params_knn['treat'] = sum_t
opt_params_knn['control'] = sum_c

print(f'Cross-validation done for N={N}')

with open('opt_params_knn.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_knn, pickle_file)