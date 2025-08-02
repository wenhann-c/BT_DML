import numpy as np
from sklearn.base import clone

def knn(y_data, d_data, x_data, model_treat, model_control, B=100, alpha=0.05):

    N = len(y_data)

    x_treat = x_data[d_data == 1]
    y_treat = y_data[d_data == 1]
    x_control = x_data[d_data == 0]
    y_control = y_data[d_data == 0]

    model_treat_ite = clone(model_treat)
    model_control_ite = clone(model_control)
    model_treat_ite.fit(X=x_treat, y=y_treat)
    model_control_ite.fit(X=x_control, y=y_control)

    mu1_hat = model_treat_ite.predict(x_data)
    mu0_hat = model_control_ite.predict(x_data)
    ite_hat = mu1_hat - mu0_hat
    ate_hat = np.mean(ite_hat)

    ate_bootstrap = []
    for _ in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        x_b = x_data[idx]
        d_b = d_data[idx]
        y_b = y_data[idx]

        x_b_treat = x_b[d_b == 1]
        y_b_treat = y_b[d_b == 1]
        x_b_control = x_b[d_b == 0]
        y_b_control = y_b[d_b == 0]

        # If there are not enough treated/control units in a bootstrap sample, skip
        if len(x_b_treat) < 10 or len(x_b_control) < 10:
            continue

        model_treat_b = clone(model_treat)
        model_control_b = clone(model_control)
        model_treat_b.fit(X=x_b_treat, y=y_b_treat)
        model_control_b.fit(X=x_b_control, y=y_b_control)

        mu1_b = model_treat_b.predict(x_data)
        mu0_b = model_control_b.predict(x_data)
        ite_b = mu1_b - mu0_b
        ate_b = np.mean(ite_b)
        ate_bootstrap.append(ate_b)

    ate_bootstrap = np.array(ate_bootstrap)
    se_hat = np.std(ate_bootstrap, ddof=1)
    lower = np.percentile(ate_bootstrap, 100 * alpha / 2)
    upper = np.percentile(ate_bootstrap, 100 * (1 - alpha / 2))
    CI = np.array([2 * ate_hat - upper , 2 * ate_hat - lower])

    return ate_hat, se_hat, CI