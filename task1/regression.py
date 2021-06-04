################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
from task1.parse import load_data, clean_data
import os
import pickle
import sys
import numpy as np
from os.path import join


class RevenueModel(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        zero_mask = np.zeros(len(data))
        x_test = clean_data(data, stage='test').drop('status', axis=1)
        y_pred = self.model.predict(x_test)
        y_pred = np.where(condition=data['status'] != 'Released', x=y_pred, y=zero_mask)
        return y_pred


def predict(csv_file="movies_dataset.csv"):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    df = load_data(csv_file)
    model_pickles_vote_avg = [f for f in os.listdir("models/") if f.startswith('average_vote_')]
    model_pickles_revenue = [f for f in os.listdir("models/") if f.startswith('revenue_')]
    model_pickles_vote_avg.sort(key=lambda f: float(f.split("_")[-1][:-4]))
    model_pickles_revenue.sort(key=lambda f: float(f.split("_")[-1][:-4]))
    sklearn_avg_vote = pickle.load(open(join("models", model_pickles_vote_avg[0]), 'rb'))
    sklearn_revenue = pickle.load(open(join("models", model_pickles_revenue[0]), 'rb'))
    model_revenue = RevenueModel(model=sklearn_revenue)
    y_rev = model_revenue.predict(data=df)
    print(y_rev)
    # print(y_avg_vote)
    # return y_rev, y_avg_vote


if __name__ == "__main__":
    predict()



