import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
import pickle

def get_data():
            
    df_raw = pd.read_csv("data/401k.csv")
    variables_needed = ['e401', 'net_tfa', 'age', 'inc', 'fsize', 'educ', 'marr', 'twoearn', 'db', 'pira', 'hown']
    df = df_raw[variables_needed]

    return df['net_tfa'].values, df['e401'].values, df.drop(['e401', 'net_tfa'], axis=1).values

def xgb_cv(y_data, d_data, x_data, cv=5):
    
    xgb_model_l = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    xgb_model_m = xgb.XGBClassifier(objective='binary:logistic', seed=42)

    param_grid_l = {

        'n_estimators': [80,100,120,140],
        'max_depth': [2],
        'subsample': [0.8],
        'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.3],
        'reg_lambda': [0.01,0.1,1,10,100,1000]
    }
    
    param_grid_m = param_grid_l.copy()
    
    grid_search_l = GridSearchCV(estimator=xgb_model_l, param_grid=param_grid_l, cv=cv, n_jobs=1,
                                 scoring='neg_mean_squared_error', verbose=1)
    
    grid_search_m = GridSearchCV(estimator=xgb_model_m, param_grid=param_grid_m, cv=cv, n_jobs=1,
                                 scoring='neg_brier_score', verbose=1)
    
    xgb_params_dict = {}

    grid_search_l.fit(X=x_data, y=y_data)
    xgb_params_dict['l'] = grid_search_l.best_params_
    grid_search_m.fit(X=x_data, y=d_data)
    xgb_params_dict['m'] = grid_search_m.best_params_

    return xgb_params_dict

y_data, d_data, x_data = get_data()
opt_params_xgb = xgb_cv(y_data, d_data, x_data)
print(f'Cross-validation done')

with open('opt_params_xgb.pkl', 'wb') as pickle_file:
    pickle.dump(opt_params_xgb, pickle_file)

def get_models(xgb_params_dict):

    model_l = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    model_l.set_params(**xgb_params_dict['l'])

    model_m = xgb.XGBClassifier(objective='binary:logistic', seed=42)
    model_m.set_params(**xgb_params_dict['m'])

    return model_l, model_m

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
        sigma2_hat = np.mean(residuals ** 2)
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

    results_dict = {
    'ate_estimate': b_hat_cv,
    'se_estimate': se_hat_cv,
    'CI': CI,
    'mse': mse
    }

    return results_dict

model_l, model_m = get_models(opt_params_xgb)
results_dict = dml(y_data, d_data, x_data, model_l, model_m)
print(f'XGBoost done')

with open('results_xgb.pkl', 'wb') as pickle_file:
    pickle.dump(results_dict, pickle_file)