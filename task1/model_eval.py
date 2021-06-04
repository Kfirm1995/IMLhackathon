import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb

import pickle
import matplotlib.pyplot as plt
from task1.parse import *
from sklearn.model_selection import train_test_split
# from xgboost
# from parse import get_train_test


def fit(model, x_train, y_train):
    return model.fit(x_train, y_train)


def plot_results(all_rss, model_names, title):
    fig = plt.figure(figsize=(10, 5))
    plt.bar(model_names, all_rss, color='maroon', width=0.4)
    plt.xlabel("model")
    plt.ylabel("MSE")
    plt.title(title)
    plt.show()
    plt.waitforbuttonpress(10)


def save_mse(fitted_model, model_name, mse, pred_type):
    mse = mse/1e6 if pred_type == 'revenue' else mse
    ls = os.listdir("models")
    files = [f for f in ls if f.startswith(f'{pred_type}_{model_name}_')]
    if not files:
        with open(f'models/{pred_type}_{model_name}_{mse}.pkl', 'wb') as fid:
            pickle.dump(fitted_model, fid)
    else:
        best = files[0]
        mse_best = float(best.split("_")[-1][:-4])
        if mse <= mse_best:
            os.remove(os.path.join("models", best))
            with open(f'models/{pred_type}_{model_name}_{mse}.pkl', 'wb') as fid:
                pickle.dump(fitted_model, fid)


def main():
    df = load_data("movies_dataset.csv")
    df, y_revenue, y_vote_avg = clean_data(df, stage='train')
    for i in range(30):
        linear_regression = LinearRegression()
        random_forest = RandomForestRegressor()
        ridge_regression = Ridge()
        adaboost = AdaBoostRegressor(random_state=0, n_estimators=10)
        decision_tree = DecisionTreeRegressor(random_state=0)
        models = [adaboost, linear_regression, random_forest, ridge_regression, decision_tree]
        model_names = ['Adaboost', 'Linear_regression', 'Random_forest', 'Ridge_regression', 'Decision_tree']
        fitted_models = []
        all_mse_vote_avg = []
        all_mse_revenue = []
        for pred_type, y_response in zip(['average_vote', 'revenue'], [y_vote_avg, y_revenue]):
            x_train, x_test, y_train, y_test = train_test_split(df, y_response, test_size=0.10, random_state=42)
            for model, model_name in zip(models, model_names):
                fitted_model = model.fit(X=x_train, y=y_train)
                fitted_models.append(fitted_model)
                y_pred = fitted_model.predict(x_test)
                print(f"-----------Prediction: {pred_type}----------- Model:{model_name} -----------------------")
                mse = np.sqrt(mean_squared_error(y_pred, y_test))
                print(f"MSE: {mse}\n")
                rss = fitted_model.score(x_test, y_test)
                print(f"score: {rss}\n")
                if pred_type == 'average_vote':
                    all_mse_vote_avg.append(mse)
                else:
                    all_mse_revenue.append(mse)
                save_mse(fitted_model=fitted_model, model_name=model_name, mse=mse, pred_type=pred_type)

    # plot_results(all_rss=all_mse_vote_avg, model_names=model_names, title="average_vote")
    # plot_results(all_rss=all_mse_revenue, model_names=model_names, title="revenue")


if __name__ == '__main__':
    main()