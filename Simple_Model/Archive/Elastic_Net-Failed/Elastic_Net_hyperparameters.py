import numpy as np
from sklearn import linear_model as lm
import pickle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from data_generation import get_data

def eln_cv(y_data, d_data, x_data):
    
    eln_model_l = lm.ElasticNetCV(l1_ratio=[.01, .05, .1, .5, .7, .9, .95, .99, 1], alphas=np.logspace(-10, 0, 100), cv=5, max_iter=10000, tol=1e-4, n_jobs=1)
    eln_model_m = lm.ElasticNetCV(l1_ratio=[.01, .05, .1, .5, .7, .9, .95, .99, 1], alphas=np.logspace(-10, 0, 100), cv=5, max_iter=10000, tol=1e-4, n_jobs=1)
    eln_params_dict = {}

    eln_model_l.fit(X=x_data, y=y_data)
    eln_params_dict['l'] = {'alpha': eln_model_l.alpha_, 'l1_ratio': eln_model_l.l1_ratio_}
    eln_model_m.fit(X=x_data, y=d_data)
    eln_params_dict['m'] = {'alpha': eln_model_m.alpha_, 'l1_ratio': eln_model_m.l1_ratio_}

    return eln_params_dict

N = 10000
n_trial = 10
opt_params_eln = {}
sum_l = {}
sum_m = {}
rng = np.random.default_rng(seed=42)
opt_params_eln_N = {}
    
for j in range(n_trial): 
    y_data, d_data, x_data = get_data(N, rng)
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_data_quad = poly_features.fit_transform(x_data)
    scaler = StandardScaler()
    x_data_quad_stand = scaler.fit_transform(x_data_quad)
    opt_params_eln_N[j] = eln_cv(y_data, d_data, x_data_quad_stand)

for i in range(n_trial):
    
    for key, value in opt_params_eln_N[i]['l'].items():
        if key in sum_l:
            sum_l[key] += value/n_trial
        else:
            sum_l[key] = value/n_trial

    for key, value in opt_params_eln_N[i]['m'].items():
        if key in sum_m:
            sum_m[key] += value/n_trial
        else:
            sum_m[key] = value/n_trial

# keys_to_round = ['alpha']
keys_dec_to_round = ['l1_ratio']

# for key in keys_to_round:
#     sum_l[key] = round(sum_l[key])
#     sum_m[key] = round(sum_m[key])

for key in keys_dec_to_round:
    sum_l[key] = round(sum_l[key],2)
    sum_m[key] = round(sum_m[key],2)

opt_params_eln['l'] = sum_l
opt_params_eln['m'] = sum_m

print(f'Cross-validation done for N={N}')

with open('opt_params_eln.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_eln, pickle_file)