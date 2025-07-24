import numpy as np
import xgboost as xgb
import pickle
from data_generation import get_data
from dml_algorithm import dml

with open('opt_params_xgb.pkl', 'rb') as pickle_file:
    opt_params_xgb = pickle.load(pickle_file)

def get_models(xgb_params_dict):

    model_l = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    model_l.set_params(**xgb_params_dict['l'])

    model_m = xgb.XGBClassifier(objective='binary:logistic', seed=42)
    model_m.set_params(**xgb_params_dict['m'])

    return model_l, model_m

def mc_simulation(N=1000, n_MC=100):
    
    rng = np.random.default_rng(seed=42)
    ate_estimates = np.empty((n_MC, 1))
    se_estimates = np.empty((n_MC, 1))
    CIs = np.empty((n_MC, 2))
    mses = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        model_l, model_m = get_models(opt_params_xgb)
        ate_estimates[j, 0], se_estimates[j,0], CIs[j, :], mses[j, :] = dml(y_data, d_data, x_data, model_l, model_m)

    results_dict = {
    'ate_estimates': ate_estimates,
    'se_estimates': se_estimates,
    'CIs': CIs,
    'mses': mses
    }

    return results_dict

results_dict = mc_simulation()
print(f'MC simulation done for N=1000')

with open('results_xgb.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)