import numpy as np
import xgboost as xgb
import pickle
from data_generation import get_data
from dml_algorithm import mm_ate, dml_ate

with open('opt_params_xgb.pkl', 'rb') as pickle_file:
    opt_params_xgb = pickle.load(pickle_file)

def get_models(xgb_params_dict):

    model_l = xgb.XGBRegressor(objective='reg:squarederror', seed=19)
    model_l.set_params(**xgb_params_dict['l'])

    model_m = xgb.XGBRegressor(objective='reg:squarederror', seed=19)
    model_m.set_params(**xgb_params_dict['m'])

    return model_l, model_m

def mc_simulation(N, n_MC=3):
    
    rng = np.random.default_rng(seed=42)
    ate_estimates = np.empty((n_MC, 2))
    sigma_estimates = np.empty((n_MC, 1))
    CIs = np.empty((n_MC, 2))
    rmses = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        ate_estimates[j, 0] = mm_ate(y_data, d_data, x_data)
        model_l, model_m = get_models(opt_params_xgb[N][j])
        ate_estimates[j, 1], sigma_estimates[j,0], CIs[j, :], rmses[j, :] = dml_ate(y_data, d_data, x_data, model_l, model_m)

    return [ate_estimates, sigma_estimates, CIs, rmses]

results_dict = {}

for N in opt_params_xgb.keys():
    results_dict[N] = mc_simulation(N)
    print(f'MC simulation done for N={N}')

with open('results_ate_xgb.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)