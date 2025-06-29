import numpy as np
from sklearn import ensemble as ens
import pickle
from data_generation import get_data
from dml_algorithm import dml

with open('opt_params_rf.pkl', 'rb') as pickle_file:
    opt_params_rf = pickle.load(pickle_file)

def get_models(rf_params_dict):

    model_l = ens.RandomForestRegressor(random_state=42, max_features='sqrt')
    model_l.set_params(**rf_params_dict['l'])

    model_m = ens.RandomForestRegressor(random_state=42, max_features='sqrt')
    model_m.set_params(**rf_params_dict['m'])

    return model_l, model_m

def mc_simulation(N=10000, n_MC=10):
    
    rng = np.random.default_rng(seed=42)
    ate_estimates = np.empty((n_MC, 1))
    se_estimates = np.empty((n_MC, 1))
    CIs = np.empty((n_MC, 2))
    mses = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        model_l, model_m = get_models(opt_params_rf)
        ate_estimates[j, 0], se_estimates[j,0], CIs[j, :], mses[j, :] = dml(y_data, d_data, x_data, model_l, model_m)

        results_dict = {
        'ate_estimates': ate_estimates,
        'se_estimates': se_estimates,
        'CIs': CIs,
        'mses': mses
        }

    return results_dict

results_dict = {}

results_dict = mc_simulation()
print(f'MC simulation done for N=10000')

with open('results_rf.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)