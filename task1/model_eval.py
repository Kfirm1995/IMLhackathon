import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
# import xgboost as xgb

import pickle
import matplotlib.pyplot as plt
from task1.parse import *
from sklearn.model_selection import train_test_split
# from xgboost
# from parse import get_train_test


def fit(model, x_train, y_train):
    return model.fit(x_train, y_train)


def plot_results(all_rss, model_names):
    fig = plt.figure(figsize=(10, 5))
    plt.bar(model_names, all_rss, color='maroon', width=0.4)
    plt.xlabel("model")
    plt.ylabel("rss")
    plt.title("RSS per model")
    plt.show()
    plt.waitforbuttonpress(-1)


def save_rss(fitted_model, model_name, rss):
    print(f"model rss: \n {rss}")
    with open(f'{model_name}_{rss}.pkl', 'wb') as fid:
        pickle.dump(fitted_model, fid)


def main():
    df = load_data("sample_set.csv")
    df, y_revenue, y_vote_avg = clean_data(df, stage='train')
    x_train, x_test, y_train, y_test= train_test_split(df, y_vote_avg, test_size=0.25, random_state=42)
    linear_regression = LinearRegression()
    ridge_regression = Ridge()
    ridge_cv = RidgeCV()
    kernel_ridge = KernelRidge()
    svr = SVR()
    DecisionTreeRegressor(random_state=0)
    models = [linear_regression, ridge_regression, ridge_cv, kernel_ridge, svr, DecisionTreeRegressor]
    model_names = ['linear_regression', 'ridge_regression', 'ridge_cv', 'kernel_ridge', 'svr', 'regression_tree']
    fitted_models = []
    all_rss = []
    for model, model_name in zip(models, model_names):
        fitted_model = model.fit(X=x_train, y=y_train)
        fitted_models.append(fitted_model)
        y_pred = fitted_model.predict(x_test)
        print(f"MSE: model {model_name} : --------------------- {np.sqrt(mean_squared_error(y_pred, y_test))} ----------------------\n")
        rss = fitted_model.score(x_test, y_test)
        all_rss.append(rss)
        save_rss(fitted_model=fitted_model, model_name=model_name, rss=rss)

    plot_results(all_rss=all_rss, model_names=model_names)


if __name__ == '__main__':
    main()