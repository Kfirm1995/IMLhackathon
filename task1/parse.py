import pandas as pd
import numpy as np
from ast import literal_eval



def load_data(filename) -> pd.DataFrame:
    """
    Load house prices dataset and preprocess data.
    :param filename: Path to house prices dataset
    :return: Design matrix (including intercept) and response vector (prices)
    """
    return pd.read_csv(filename)


# todo remove revenue zero lines
# todo dummies for belongs to collection
# todo add feature quarter of release


def clean_data(df: pd.DataFrame):

    # dropping duplicates
    df = df.drop_duplicates()

    # removing ids
    df = df.drop("id", 1)

    # adding column of boolean belongs to collection
    df[["is_belongs_to_collection"]] = df[["belongs_to_collection"]].notnull().astype(int)

    # df['belongs_to_collection'] = df['belongs_to_collection'].fillna(dict).apply(literal_eval)
    # literal_eval(df[[]])
    # df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: [e['id'] for e in x] if isinstance(x, list) else [])



    # budget stay as it is

    # genre
    df = get_dummies_for_uniques(df, 'genres')

    # change homepage feature to boolean is com
    # df['homepage'] = df['homepage'].apply(lambda x: 1 if ".com" in x else 0)


    # production companies
    # df = get_dummies_for_uniques(df, 'production_companies') //todo improve



    # y
    y = df.revenue

    return df, y









    # for c in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built",
    #           "sqft_living15", "sqft_lot15"]:
    #     df = df[df[c] > 0]
    # for c in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
    #     df = df[df[c] >= 0]
    #
    # df = df[df["waterfront"].isin([0, 1]) &
    #         df["view"].isin(range(5)) &
    #         df["condition"].isin(range(1, 6)) &
    #         df["grade"].isin(range(1, 15))]
    #
    # df["recently_renovated"] = np.where(
    #     df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 70), 1,
    #     0)
    # df = df.drop("yr_renovated", 1)
    #
    # df["decade_built"] = (df["yr_built"] / 10).astype(int)
    # df = df.drop("yr_built", 1)
    #
    # df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    # df = pd.get_dummies(df, prefix='decade_built_', columns=['decade_built'])
    #
    # # Removal of outliers (Notice that there exists methods for better defining outliers
    # # but for this course this suffices
    # df = df[df["bedrooms"] < 20]
    # df = df[df["sqft_lot"] < 1250000]
    # df = df[df["sqft_lot15"] < 500000]
    #
    # df.insert(0, 'intercept', 1, True)


def get_dummies_for_uniques(df, feature: str):
    df['new_genres'] = df[feature].apply(literal_eval)
    df['genre_names'] = df['new_genres'] \
        .apply(lambda x: [e['name'] for e in x] if isinstance(x, list) else [])
    genres_df = pd.DataFrame(df['genre_names'].tolist())
    stacked_genres = genres_df.stack()
    raw_dummies = pd.get_dummies(stacked_genres)
    genre_dummies = raw_dummies.sum(level=0)
    df = pd.concat([df, genre_dummies], axis=1)
    return df