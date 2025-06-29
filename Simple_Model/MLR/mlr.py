import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

def mlr(y_data, d_data, x_data, K=5, alpha=0.05):

    kf = KFold(n_splits=K, shuffle=True)
    
    b_check_list = []
    mse_list = []
    
    for (train_indices, test_indices) in kf.split(X=d_data, y=y_data):
        y_train, d_train, x_train_raw = y_data[train_indices], d_data[train_indices], x_data[train_indices]
        y_test, d_test, x_test_raw= y_data[test_indices], d_data[test_indices], x_data[test_indices]

        scaler_x = StandardScaler()

        x_train = scaler_x.fit_transform(x_train_raw)

        x_test = scaler_x.transform(x_test_raw)
        
        model_g = LinearRegression(fit_intercept=False)

        model_l = LinearRegression().fit(X=x_train,y=y_train)
        l_hat = model_l.predict(x_test)
        y_resid = y_test - l_hat
        y_resid = y_resid.reshape(-1,1)

        model_m = LinearRegression().fit(X=x_train,y=d_train)
        m_hat = model_m.predict(x_test)
        d_resid = d_test - m_hat
        d_resid = d_resid.reshape(-1,1)

        model_g.fit(X=d_resid,y=y_resid)
        coef = model_g.coef_
        b_check_list.append(coef)

        mse_l = mean_squared_error(y_test,l_hat)
        mse_m = mean_squared_error(d_test,m_hat)
        mse_list.append([mse_l,mse_m])


    b_hat = np.mean(b_check_list)
    sigma_hat = np.std(b_check_list, ddof=1)
    N = len(y_data)
    quantile = norm.ppf(1 - alpha/2)
    se_hat = sigma_hat/np.sqrt(N)
    CI = np.array([b_hat - quantile*se_hat, b_hat + quantile*se_hat])
    
    mse = np.mean(mse_list, axis=0)

    return b_hat, se_hat, CI, mse