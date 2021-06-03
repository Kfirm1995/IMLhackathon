import numpy as np
import pandas as pd
from task1.parse import *
from task1.model_eval import *


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    df = load_data(csv_file)
    df, y_revenue, y_vote_avg = clean_data(df, stage='train')



if __name__ == '__main__':
    predict("sample_set.csv")



