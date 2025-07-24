import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.base import clone

def dml(y_data, d_data, x_data, model_l, model_m, K=5, alpha=0.05):

    kf = StratifiedKFold(n_splits=K, shuffle=False)
    
    b_list = []
    var_list = []
    mse_list = []
    
    for (train_indices, test_indices) in kf.split(X=x_data, y=d_data):
        y_train, d_train, x_train = y_data[train_indices], d_data[train_indices], x_data[train_indices]
        y_test, d_test, x_test = y_data[test_indices], d_data[test_indices], x_data[test_indices]

        model_l_fold = clone(model_l)
        model_m_fold = clone(model_m)

        model_l_fold.fit(X=x_train,y=y_train)
        l_hat = model_l_fold.predict(x_test)
        y_resid = y_test - l_hat
        y_resid = y_resid.reshape(-1,1)

        model_m_fold.fit(X=x_train,y=d_train)
        m_hat = model_m_fold.predict_proba(x_test)[:,1]
        d_resid = d_test - m_hat
        d_resid = d_resid.reshape(-1,1)

        score_a = np.multiply(d_resid, d_resid)
        score_b = np.multiply(d_resid, y_resid)
        b_hat = np.mean(score_b) / np.mean(score_a)
        b_list.append(b_hat)

        residuals = y_resid - (b_hat * d_resid)
        sigma2_hat = np.sum(residuals ** 2)/(len(residuals) - 1)
        var_hat = sigma2_hat / np.sum(score_a)
        var_list.append(var_hat)

        mse_l = mean_squared_error(y_test,l_hat)
        mse_m = mean_squared_error(d_test,m_hat)
        mse_list.append([mse_l,mse_m])

    b_hat_cv = np.mean(b_list)
    var_hat_cv = np.sum(var_list) / (K ** 2)
    quantile = norm.ppf(1 - alpha/2)
    se_hat_cv = np.sqrt(var_hat_cv)
    CI = np.array([b_hat_cv - quantile*se_hat_cv, b_hat_cv + quantile*se_hat_cv])

    mse = np.mean(mse_list, axis=0)

    return b_hat_cv, se_hat_cv, CI, mse