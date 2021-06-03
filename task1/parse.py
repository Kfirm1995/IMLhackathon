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
# tod


def clean_data(df: pd.DataFrame):
    df = handle_first(df)

    df = handle_id(df)
    df = handle_belongs_to_collection(df)
    df = handle_budget(df)
    df = handle_genres(df)
    df = handle_homepage(df)
    df = handle_original_languages(df)
    df = handle_original_title(df)
    df = handle_overview(df)
    df = handle_vote_average(df)
    df = handle_vote_count(df)
    df = handle_production_companies(df)
    df = handle_production_countries(df)
    df = handle_release_date(df)
    df = handle_runtime(df)
    df = handle_spoken_languages(df)
    df = handle_status(df)
    df = handle_tagline(df)
    df = handle_title(df)
    df = handle_keywords(df)
    df = handle_cast(df)
    df = handle_crew(df)
    df = handle_revenue(df)
    # drop original title

    # production companies
    # df = get_dummies_for_uniques(df, 'production_companies') //todo improve

    # y
    y = df.revenue

    return df, y


###########

def handle_first(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    # todo validate with roy and shemesh
    # df = df[df['release_date'].notna()]
    # df = df[df['revenue'] > 0]
    return df

def handle_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("id", 1)
    return df


def handle_belongs_to_collection(df: pd.DataFrame) -> pd.DataFrame:
    df[["is_belongs_to_collection"]] = df[["belongs_to_collection"]].notnull().astype(int)
    return df


def handle_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    leave budget as it is
    :param df:
    :return:
    """
    return df


def handle_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    encode one hot by genres
    :param df:
    :return:
    """
    df = get_dummies_for_uniques(df, 'genres', value='name')
    return df


def handle_homepage(df: pd.DataFrame) -> pd.DataFrame:
    """
    change it to boolean by if .com exist in the homepage zero else
    :param df:
    :return:
    """
    df['homepage'] = df['homepage'].map(lambda x: 1 if '.com' in str(x) else 0)
    return df


def handle_original_languages(df: pd.DataFrame) -> pd.DataFrame:
    """
    encode onehot by languages (limited by 150)
    :param df:
    :return:
    """
    df = encode_one_hot(df, 'original_language')
    return df


def handle_original_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropping title
    :param df:
    :return:
    """
    df = df.drop("original_title", 1)
    return df


def handle_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropping overview
    :param df:
    :return:
    """
    df = df.drop("overview", 1)
    return df


def handle_vote_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    leave vote_avg as it is
    :param df:
    :return:
    """
    return df


def handle_vote_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    leave vote_count as it is
    :param df:
    :return:
    """
    return df

# todo shemesh
def handle_production_companies(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_production_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    one hot encoding by get dummies
    :param df:
    :return:
    """
    df = get_dummies_for_uniques(df, feature="production_countries", value="iso_3166_1")
    return df


def handle_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    adding quarter feature
    :param df:
    :return:
    """
    df = df[df['release_date'].notna()]
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['quarter'] = df['release_date'].dt.quarter

    return df


def handle_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    very correlated feature, leave it as it is
    :param df:
    :return:
    """
    return df


def handle_spoken_languages(df: pd.DataFrame) -> pd.DataFrame:
    """
    encode by one hot with get dummies by languages
    :param df:
    :return:
    """
    df = get_dummies_for_uniques(df, 'spoken_languages', 'iso_639_1')
    return df


def handle_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropping
    :param df:
    :return:
    """
    df = df.drop("status", 1)
    return df


def handle_tagline(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropping
    :param df:
    :return:
    """
    df = df.drop("tagline", 1)
    return df


def handle_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropping
    :param df:
    :return:
    """
    df = df.drop("title", 1)
    return df


def handle_keywords(df: pd.DataFrame) -> pd.DataFrame:
    # todo counter of keywords
    return df


def handle_cast(df: pd.DataFrame) -> pd.DataFrame:
    # todo first actor
    return df


def handle_crew(df: pd.DataFrame) -> pd.DataFrame:
    # todo director
    return df


def handle_revenue(df: pd.DataFrame) -> pd.DataFrame:

    return df


###########################

def get_dummies_for_uniques(df: pd.DataFrame, feature: str, value: str):
    df['new'] = df[feature].apply(literal_eval)
    df['names'] = df['new'].apply(lambda x: [e[value] for e in x] if isinstance(x, list) else [])
    df = encode_one_hot(df, 'names')

    # cleaning after done
    for col in ['new', 'names', feature]:
        df = df.drop(col, 1)
    return df


def encode_one_hot(df, feature: str):
    genres_df = pd.DataFrame(df[feature].tolist())
    stacked_genres = genres_df.stack()
    raw_dummies = pd.get_dummies(stacked_genres)
    genre_dummies = raw_dummies.sum(level=0)
    df = pd.concat([df, genre_dummies], axis=1)
    return df

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
