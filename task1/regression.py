
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

import pandas as pd
import numpy as np


def load_data(filename) -> pd.DataFrame:
    """
    Load house prices dataset and preprocess data.
    :param filename: Path to house prices dataset
    :return: Design matrix (including intercept) and response vector (prices)
    """
    return pd.read_csv(filename).dropna().drop_duplicates()


def clean_data(df: pd.DataFrame):
    df["zipcode"] = df["zipcode"].astype(int)

    for c in ["id", "lat", "long", "date"]:
        df = df.drop(c, 1)

    for c in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built",
              "sqft_living15", "sqft_lot15"]:
        df = df[df[c] > 0]
    for c in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[c] >= 0]

    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 15))]

    df["recently_renovated"] = np.where(
        df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 70), 1,
        0)
    df = df.drop("yr_renovated", 1)

    df["decade_built"] = (df["yr_built"] / 10).astype(int)
    df = df.drop("yr_built", 1)

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='decade_built_', columns=['decade_built'])

    # Removal of outliers (Notice that there exists methods for better defining outliers
    # but for this course this suffices
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["sqft_lot15"] < 500000]

    df.insert(0, 'intercept', 1, True)
    return df.drop("price", 1), df.price



def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    df = load_data(csv_file)




