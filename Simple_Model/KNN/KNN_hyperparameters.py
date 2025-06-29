import numpy as np
from sklearn import neighbors as nb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import GridSearchCV
from data_generation import get_data

def knn_cv(y_data, d_data, x_data, cv=5):
    
    knn_model = nb.KNeighborsRegressor()

    param_grid = {

        'n_neighbors': [5,10,15,20,25,30],
        'p': [2,3,4]

    }
    
    grid_search_l = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    grid_search_m = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    knn_params_dict = {}

    grid_search_l.fit(X=x_data, y=y_data)
    knn_params_dict['l'] = grid_search_l.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    knn_params_dict['m'] = grid_search_m.best_params_

    return knn_params_dict

N = 10000
n_trial = 10
opt_params_knn = {}
sum_l = {}
sum_m = {}
rng = np.random.default_rng(seed=42)
opt_params_knn_N = {}
    
for j in range(n_trial):
    y_data, d_data, x_data = get_data(N, rng)
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_data_quad = poly_features.fit_transform(x_data)
    scaler_x = StandardScaler()
    x_data_scaled = scaler_x.fit_transform(x_data_quad)
    opt_params_knn_N[j] = knn_cv(y_data, d_data, x_data_scaled)

for i in range(n_trial):
    
    for key, value in opt_params_knn_N[i]['l'].items():
        if key in sum_l:
            sum_l[key] += value/n_trial
        else:
            sum_l[key] = value/n_trial

    for key, value in opt_params_knn_N[i]['m'].items():
        if key in sum_m:
            sum_m[key] += value/n_trial
        else:
            sum_m[key] = value/n_trial

keys_to_round = ['n_neighbors','p']

for key in keys_to_round:
    sum_l[key] = round(sum_l[key])
    sum_m[key] = round(sum_m[key])

opt_params_knn['l'] = sum_l
opt_params_knn['m'] = sum_m

print(f'Cross-validation done for N={N}')

with open('opt_params_knn.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_knn, pickle_file)