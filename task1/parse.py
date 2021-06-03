import datetime

import pandas as pd
import numpy as np
from ast import literal_eval
from task1.utils import *


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


def clean_data(df: pd.DataFrame, stage='train'):
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
    y_revenue = df.revenue
    y_vote_avg = df.vote_average

    return df, y_revenue, y_vote_avg


###########


def handle_first(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df[df['revenue'].notna()]
    df = df[df['revenue'] >= 1000]
    return df


def handle_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("id", 1)
    return df


def handle_belongs_to_collection(df: pd.DataFrame) -> pd.DataFrame:
    # todo ordinary collection
    df[["is_belongs_to_collection"]] = df[["belongs_to_collection"]].notnull().astype(int)
    return df


def handle_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    leave budget as it is
    :param df:
    :return:
    """
    # todo add log budget
    df['log_budget'] = df['budget'].map(lambda x: 0 if float(x) < 2 else math.log(float(x)))
    df['budget/runtime'] = df['budget'] / df['runtime']
    return df


def handle_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    encode one hot by genres
    :param df:
    :return:
    """
    # todo ensure robustness
    df = get_dummies_for_uniques(df, 'genres', value='name')
    return df


def handle_homepage(df: pd.DataFrame) -> pd.DataFrame:
    """
    change it to boolean by if .com exist in the homepage zero else
    :param df:
    :return:
    """
    df['has_homepage'] = df['homepage'].apply(lambda x: 0 if pd.isna(x) else 1)
    df = df.drop('homepage', axis=1)
    return df


def handle_original_languages(df: pd.DataFrame) -> pd.DataFrame:
    """
    encode onehot by languages (limited by 150)
    :param df:
    :return:
    """
    # todo check most occurrences of languages and change to 1 just on them
    top_languages = get_top_n_freq_values(df, 5, 'original_language')
    df['original_language'] = df['original_language'].map(lambda x: 1 if x in top_languages else 0)
    return df


def handle_original_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    dropping title
    :param df:
    :return:
    """
    # todo add length of title string and check corellation (maybe drop)
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
    # todo change to y response
    return df


def handle_vote_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    leave vote_count as it is
    :param df:
    :return:
    """
    # todo check correlation
    return df


def handle_production_companies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a map from company of production to average vote and average revenue.
    :param df: data.
    :return: data['company'] is dropped and instead df['company_id_vote_batch'] and df['company_id_revenue_batch']
    are added
    """
    dic_rev = json.load(open("memory_maps/company_id_map_to_rev.json"))
    dic_vote = json.load(open("memory_maps/company_id_map_to_vote.json"))

    def score_by_json_st_rev(json_st):
        company_id_row = re.sub("[^0-9]", "", json_st.split(',')[0])
        if company_id_row == '':
            return 1
        return dic_rev[company_id_row]

    def score_by_json_st_vote(json_st):
        company_id_row = re.sub("[^0-9]", "", json_st.split(',')[0])
        if company_id_row == '':
            return 1
        return dic_vote[company_id_row]

    df['company_id_revenue_batch'] = df['production_companies'].map(lambda x: score_by_json_st_rev(x))
    df['company_id_vote_batch'] = df['production_companies'].map(lambda x: score_by_json_st_vote(x))
    df = df.drop('production_companies', axis=1)
    # plot_corr_heatmap(df)
    return df


def handle_production_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    one hot encoding by get dummies
    :param df:
    :return:
    """
    # todo correlate offline and remain top five + one hot
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
    df['year'] = pd.DatetimeIndex(df['release_date']).year

    # adding days passed released
    today = datetime.datetime.now()
    df['days_from_release'] = df['release_date'].map(lambda x: (today - pd.to_datetime(x)).days)

    df = df.drop("release_date", 1)

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

    # todo correlate offline and remain top five + one hot
    df = df.drop('spoken_languages', axis=1)
    # df = get_dummies_for_uniques(df, 'spoken_languages', 'iso_639_1')
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
    df = df.drop("keywords", 1)
    return df


def handle_cast(df: pd.DataFrame) -> pd.DataFrame:
    # todo has superstar? find on internet on hot of this
    return df


def handle_crew(df: pd.DataFrame) -> pd.DataFrame:
    # todo has good director?
    return df


def handle_revenue(df: pd.DataFrame) -> pd.DataFrame:
    return df


###########################

def get_dummies_for_uniques(df: pd.DataFrame, feature: str, value: str):
    df['new'] = df[feature].apply(literal_eval)
    df['names'] = df['new'].apply(lambda x: [e[value] for e in x] if isinstance(x, list) else [])
    # write_best_langs(df)
    df = encode_one_hot(df, 'names')

    # cleaning after done
    for col in ['new', 'names', feature]:
        df = df.drop(col, 1)
    # plot_corr_heatmap(df)
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
