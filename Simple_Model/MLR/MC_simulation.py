import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pickle
from data_generation import get_data
from mlr import mlr

def mc_simulation(N=10000, n_MC=10):
    
    rng = np.random.default_rng(seed=42)
    ate_estimates = np.empty((n_MC, 1))
    se_estimates = np.empty((n_MC, 1))
    CIs = np.empty((n_MC, 2))
    mses = np.empty((n_MC, 2))

    for j in range(n_MC):
        y_data, d_data, x_data = get_data(N, rng)
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        x_data_quad = poly_features.fit_transform(x_data)
        ate_estimates[j, 0], se_estimates[j,0], CIs[j, :], mses[j, :] = mlr(y_data, d_data, x_data_quad)

        results_dict = {
        'ate_estimates': ate_estimates,
        'se_estimates': se_estimates,
        'CIs': CIs,
        'mses': mses
        }

    return results_dict

results_dict = mc_simulation()
print(f'MC simulation done for N=10000')

with open('results_mlr.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)