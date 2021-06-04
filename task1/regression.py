################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
from task1.parse import load_data, clean_data
import os
import pickle
import numpy as np
from os.path import join
from task1.parse import MEAN_REVENUE, MEAN_AVERAGE_VOTE
from sklearn.metrics import mean_squared_error


class Model(object):
    """
    A wrapper to sklearn models1, allows to include heuristics.
    """
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def predict(self, data):
        try:
            x_test = clean_data(data, stage='test')
            for response_col in ['revenue', 'vote_average']:
                if response_col in x_test.columns:
                    x_test = x_test.drop(response_col, axis=1)
            y_pred = self.model.predict(x_test)
            y_pred[np.where(data['status'] != 'Released')] = 0
            y_pred[np.where(data['budget'] <= 0)] = 0
            if self.name == 'average_vote':
                y_pred = np.around(y_pred, decimals=1)
            return y_pred
        except:
            return np.full(len(data), MEAN_REVENUE) if self.name == 'revenue' else np.full(len(data), MEAN_AVERAGE_VOTE)


def predict(csv_file="movies_dataset_part2.csv"):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    df = load_data(csv_file)
    y_true_rev = df['revenue']
    y_true_vote_avg = df['vote_average']
    model_pickles_vote_avg = [f for f in os.listdir("models1/") if f.startswith('average_vote_')]
    model_pickles_revenue = [f for f in os.listdir("models1/") if f.startswith('revenue_')]
    model_pickles_vote_avg.sort(key=lambda f: float(f.split("_")[-1][:-4]))
    model_pickles_revenue.sort(key=lambda f: float(f.split("_")[-1][:-4]))
    sklearn_avg_vote = pickle.load(open(join("models1", model_pickles_vote_avg[0]), 'rb'))
    sklearn_revenue = pickle.load(open(join("models1", model_pickles_revenue[0]), 'rb'))
    model_revenue = Model(model=sklearn_revenue, name='revenue')
    model_avg_vote = Model(model=sklearn_avg_vote, name='average_vote')
    y_rev = model_revenue.predict(data=df)
    y_avg_vote = model_avg_vote.predict(data=df)
    print(f'y_avg:     {np.sqrt(mean_squared_error(y_avg_vote, y_true_vote_avg))}')
    print(f'y_rev:     {np.sqrt(mean_squared_error(y_rev, y_true_rev))}')
    # print(y_avg_vote, y_rev)
    return y_avg_vote, y_rev


if __name__ == "__main__":
    predict()



