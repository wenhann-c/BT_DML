import numpy as np
from sklearn import neighbors as nb
import pickle
from data_generation import get_data
from KNN import knn

with open('opt_params_knn.pkl', 'rb') as pickle_file:
    opt_params_knn = pickle.load(pickle_file)

def get_models(knn_params_dict):

    model_treat = nb.KNeighborsRegressor()
    model_treat.set_params(**knn_params_dict['treat'])

    model_control = nb.KNeighborsRegressor()
    model_control.set_params(**knn_params_dict['control'])

    return model_treat, model_control

def mc_simulation(N=1000, n_MC=100):
    
    rng = np.random.default_rng(seed=42)
    ate_estimates = np.empty((n_MC, 1))
    se_estimates = np.empty((n_MC, 1))
    CIs = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        model_treat, model_control = get_models(opt_params_knn)
        ate_estimates[j, 0], se_estimates[j,0], CIs[j, :] = knn(y_data, d_data, x_data, model_treat, model_control)
        
    results_dict = {
    'ate_estimates': ate_estimates,
    'se_estimates': se_estimates,
    'CIs': CIs,
    }

    return results_dict

results_dict = mc_simulation()
print(f'MC simulation done for N=1000')

with open('results_knn.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)