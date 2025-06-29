import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.base import is_regressor
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression

from data_generation import g, d

def mm_ate(y_data, d_data, x_data):
    return np.mean(((y_data - g(x_data)) * (d_data - d(x_data))) - ((d_data - d(x_data)) * (d_data - d(x_data))))

def dml_ate(y_data, d_data, x_data, model_l, model_m, K=5, alpha=0.05):

    kf = KFold(n_splits=K, shuffle=False)
    
    b_check_list = []
    scores_list = []
    rmse_list = []
    
    for (train_indices, test_indices) in kf.split(X=x_data, y=y_data):
        y_train, d_train, x_train = y_data[train_indices], d_data[train_indices], x_data[train_indices]
        y_test, d_test, x_test = y_data[test_indices], d_data[test_indices],x_data[test_indices]
        
        model_l.fit(X=x_train,y=y_train)
        l_hat = model_l.predict(x_test)
        y_resid = y_test - l_hat
        y_resid = y_resid.reshape(-1,1)

        model_m.fit(X=x_train,y=d_train)
        m_hat = model_m.predict(x_test)
        d_resid = d_test - m_hat
        d_resid = d_resid.reshape(-1,1)

        reg = LinearRegression(fit_intercept=False).fit(X=d_resid,y=y_resid)
        theta = reg.coef_
        b_check_list.append(theta)

        # scores_a = - (d_test - m_hat) * (d_test - m_hat)
        # scores_b = (y_test - l_hat) * (d_test - m_hat)
        scores = (y_test - l_hat - theta*(d_test - m_hat)) * (d_test - m_hat)
        scores_list.append(scores)

        rmse_l = root_mean_squared_error(g(x_test),l_hat)
        rmse_m = root_mean_squared_error(d(x_test),m_hat)
        rmse_list.append([rmse_l,rmse_m])

    b_hat = np.mean(b_check_list)
    sigma_hat = np.mean(scores_list)
    N = len(y_data)
    quantile = norm.ppf(1 - alpha/2)
    CI = np.array([b_hat - quantile*sigma_hat/np.sqrt(N), b_hat + quantile*sigma_hat/np.sqrt(N)])

    rmse = np.mean(rmse_list, axis=0)

    return b_hat, sigma_hat, CI, rmse