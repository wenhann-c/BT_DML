import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

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

    return b_hat, se_hat, CI