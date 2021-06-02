import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pickle
# from parse import get_train_test


def fit(model, x_train, y_train):
    return model.fit(x_train, y_train)


def evaluate(models, x_test, y_test):

    pass

def save_rss(fitted_model, model_name, x_test, y_test):
    # TODO: return results.
    rss = fitted_model.score(x_test, y_test)
    print(f"model rss: \n {rss}")
    with open(f'{model_name}_{rss}.pkl', 'wb') as fid:
        pickle.dump(fitted_model, fid)

    # load it again
    # with open('my_dumped_classifier.pkl', 'rb') as fid:
    #     gnb_loaded = pickle.load(fid)


def save_results(fitted_model, model_name, results):
    pass


def main():
    x_train, x_test, y_train, y_test = None, None, None, None  # get_train_test() when implemented.
    linear_regression = LinearRegression()
    ridge_regression = Ridge()
    ridge_cv = RidgeCV()
    kernel_ridge = KernelRidge()
    svr = SVR()
    DecisionTreeRegressor(random_state=0)
    models = [linear_regression, ridge_regression, ridge_cv, kernel_ridge, svr, DecisionTreeRegressor]
    model_names = ['linear_regression', 'ridge_regression', 'ridge_cv', 'kernel_ridge', 'svr', 'regression_tree']
    for model, model_name in zip(models, model_names):
        fitted_model = fit(model=model, x_train=x_train, y_train=y_train)
        results = evaluate(fitted_model=fitted_model, x_test=x_test, y_test=y_test)
        save_results(fitted_model, model_name, results)
    pass


if __name__ == '__main__':
    # TODO:
    # train_data, test_data = ...
    train_data, test_data = None, None
    main(train_data, test_data)