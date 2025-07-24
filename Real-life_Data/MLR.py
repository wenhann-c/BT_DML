import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import pickle

def get_data():
            
    df_raw = pd.read_csv("data/401k.csv")
    variables_needed = ['e401', 'net_tfa', 'age', 'inc', 'fsize', 'educ', 'marr', 'twoearn', 'db', 'pira', 'hown']
    df = df_raw[variables_needed]

    return df['net_tfa'].values, df['e401'].values, df.drop(['e401', 'net_tfa'], axis=1).values

def mlr(y_data, d_data, x_data, alpha=0.05):

    model = LinearRegression(fit_intercept=False)
    N = len(y_data)
    X = np.hstack([np.ones((N, 1)), d_data.reshape(-1,1), x_data])
    p = X.shape[1]

    fitted_model = model.fit(X=X,y=y_data)
    b_hat = fitted_model.coef_[1]

    y_hat = fitted_model.predict(X)
    y_resid = y_data - y_hat
    sigma2_hat = (y_resid @ y_resid) / (N - p)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_matrix = sigma2_hat * XtX_inv
    se_hat = np.sqrt(cov_matrix[1,1])
    quantile = norm.ppf(1 - alpha/2)
    CI = np.array([b_hat - quantile*se_hat, b_hat + quantile*se_hat])

    results_dict = {
    'ate_estimate': b_hat,
    'se_estimate': se_hat,
    'CI': CI,
    }

    return results_dict

y_data, d_data, x_data = get_data()
results_dict = mlr(y_data, d_data, x_data)
print(f'MLR done')

with open('results_mlr.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)