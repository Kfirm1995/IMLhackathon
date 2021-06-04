import datetime
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import pickle
from ast import literal_eval
import math
from task1.utils import *

# default mean values for outliers
MEAN_BUDGET = 23_587_185
MEAN_RUNTIME = 107
MEAN_VOTE_COUNT = 1391
MEDIAN_DATE = "01/01/2005"

# Global Members
TOP_ORIGINAL_DIC = json.load(open("memory_maps/top_original_languages.json"))
REV_DIC = json.load(open("memory_maps/company_id_map_to_rev.json"))
VOTE_DIC = json.load(open("memory_maps/company_id_map_to_vote.json"))
TOP_ACTOR_SET = pickle.load(open("memory_maps/top_actor_set.pickle", 'rb'))
TOP_ACTOR_DIC = json.load(open("memory_maps/top_director.json", 'r'))
GENRES_DIC = json.load(open("memory_maps/top_director.json", 'r'))
INFLATION_DF = pd.read_csv("inflation_data.csv")


def load_data(filename) -> pd.DataFrame:
    """
    Load house prices dataset and preprocess data.
    :param filename: Path to house prices dataset
    :return: Design matrix (including intercept) and response vector (prices)
    """
    return pd.read_csv(filename)


def clean_data(df: pd.DataFrame, stage='train'):
    df = handle_first(df, stage)
    df = handle_id(df)
    df = handle_belongs_to_collection(df, stage)
    df = handle_runtime(df, stage)
    df = handle_budget(df, stage)
    df = handle_genres(df, stage)
    df = handle_homepage(df)
    df = handle_original_languages(df)
    df = handle_release_date(df, stage)
    df = handle_original_title(df)
    df = handle_overview(df)
    df = handle_vote_count(df, stage)
    df = handle_production_companies(df)
    df = handle_production_countries(df)
    df = handle_spoken_languages(df)
    df = handle_status(df, stage)
    df = handle_tagline(df)
    df = handle_title(df)
    df = handle_keywords(df)
    df = handle_cast(df)
    df = handle_crew(df)
    df = handle_revenue(df)
    if stage == 'train':
        y_revenue = df.revenue
        y_vote_avg = df.vote_average
        df = df.drop(['revenue', 'vote_average'], axis=1)
        return df, y_revenue, y_vote_avg
    else:
        return df


def handle_first(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage == 'train':
        df = df[df['revenue'].notna()]
        df = df.drop_duplicates()
        df['revenue'] = df['revenue'].apply(lambda x: 0 if not str(x).isnumeric() else x)
        df = df[df['revenue'] >= 1000]
    return df


def handle_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("id", 1)
    return df


def handle_belongs_to_collection(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    df[["belongs_to_collection"]] = df[["belongs_to_collection"]].notnull().astype(int)
    return df


def handle_budget(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    leave budget as it is
    :param df:
    :return:
    """
    df = df[df['budget'].notna()]
    df['budget'] = df['budget'].apply(lambda x: 0 if not str(x).isnumeric() else x)
    df['log_budget'] = df['budget'].map(lambda x: 0 if float(x) < 2 else math.log(float(x)))
    df['runtime'].replace(0, MEAN_RUNTIME, inplace=True)
    df['budget/runtime'] = df['budget'] / df['runtime']
    df['budget/runtime'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df


def handle_genres(df: pd.DataFrame, stage) -> pd.DataFrame:
    """
    encode one hot by genres
    :param df:
    :return:
    """
    # validate

    df['genres'] = df['genres'].fillna("[]")

    for g in GENRES_DIC:
        df[g] = df['genres'].apply(lambda x: 1 if g in x else 0)

    df = df.drop("genres", axis=1)
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
    df['original_language'] = df['original_language'].fillna("")
    df['original_language'] = df['original_language'].map(lambda x: 1 if x in TOP_ORIGINAL_DIC else 0)
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


def handle_vote_count(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    leave vote_count as it is
    :param df:
    :param stage:
    :return:
    """
    df['vote_count'] = df['vote_count'].fillna(value=MEAN_VOTE_COUNT)
    df['vote_count'] = df['vote_count'].apply(lambda x: MEAN_VOTE_COUNT if not str(x).isnumeric() else x)
    return df


def handle_production_companies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a map from company of production to average vote and average revenue.
    :param df: data.
    :return: data['company'] is dropped and instead df['company_id_vote_batch'] and df['company_id_revenue_batch']
    are added
    """

    def score_by_json_st(json_st, dic: dict):
        try:
            company_id_row = re.sub("[^0-9]", "", json_st.split(',')[0])
            if company_id_row not in dic:
                return 1
            return dic[company_id_row]
        except Exception:
            return 1

    df['company_id_revenue_batch'] = df['production_companies'].map(lambda x: score_by_json_st(x, REV_DIC))
    df['company_id_vote_batch'] = df['production_companies'].map(lambda x: score_by_json_st(x, VOTE_DIC))
    df = df.drop('production_companies', axis=1)
    return df


def handle_production_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    one hot encoding by get dummies
    :param df:
    :return:
    """
    df = df.drop('production_countries', axis=1)
    return df


def handle_release_date(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    adding quarter feature
    adding decade
    adding days from release
    :param df:
    :return:
    """
    if stage == 'train':
        df = df[df['release_date'].notna()]
    try:
        df['release_date'] = pd.to_datetime(df['release_date'])
        # df['quarter'] = df['release_date'].dt.quarter
        # df['month'] = pd.DatetimeIndex(df['release_date']).month
        # # plot_corr_heatmap(df)
        # df['year'] = pd.DatetimeIndex(df['release_date']).year
        # # adding decade
        # df['decade'] = df['year'].map(lambda x: x - x % 10)
    except Exception:
        df['release_date'] = '2005-11-12'
    finally:
        df['quarter'] = df['release_date'].dt.quarter
        df['month'] = pd.DatetimeIndex(df['release_date']).month
        # plot_corr_heatmap(df)
        df['year'] = pd.DatetimeIndex(df['release_date']).year
        df['decade'] = df['year'].map(lambda x: x - x % 10)

    today = datetime.datetime.now()
    df['days_from_release'] = df['release_date'].map(lambda x: (today - pd.to_datetime(x)).days)
    df = df.drop("release_date", 1)
    return df


def handle_runtime(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    very correlated feature, leave it as it is
    :param df:
    :return:
    """
    if stage == 'train':
        df = df.dropna(axis=0, subset=['runtime'])
    else:
        df['runtime'] = df['runtime'].fillna(value=MEAN_RUNTIME)
        df['runtime'] = df['runtime'].apply(lambda x: MEAN_RUNTIME if not str(x).isnumeric() else x)
    return df


def handle_spoken_languages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dropping
    :param df:
    :return:
    """
    df = df.drop('spoken_languages', axis=1)
    return df


def handle_status(df: pd.DataFrame, stage) -> pd.DataFrame:
    """
    dropping
    :param df:
    :return:
    """
    if stage == "train":
        df['status'] = df['status'].apply(lambda x: np.nan if x != 'Released' else x)
        df = df.dropna(axis=0, subset=['status'])
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
    actors = list(TOP_ACTOR_SET)
    df['cast'] = df['cast'].fillna("[]")
    # dummies of actors
    for actor in actors:
        df[actor] = df['cast'].apply(lambda x: 1 if actor in x else 0)
    df = df.drop('cast', axis=1)
    return df


def handle_crew(df: pd.DataFrame) -> pd.DataFrame:
    df['crew'] = df['crew'].fillna("[]")
    for d in TOP_ACTOR_DIC.keys():
        df[d] = df['crew'].apply(lambda x: math.ceil(TOP_ACTOR_DIC[d] / 25) if d in x else 0)
    df = df.drop('crew', axis=1)
    return df


def handle_revenue(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """
    create 2 feeatures:
        1) inflation rate from 1900 to day depend on th eyear of relase
        2) usd value in rleation to usd value in 1900
    :param df:
    :return:
    """
    # cleaning year
    df['temp_year'] = df['year'].apply(lambda x: np.where(str(x).isdigit(), x, '1900'))
    df['temp_year'] = df['temp_year'].apply(lambda x: np.where(int(x) < 1900, x, '1900'))
    df['temp_year'] = df['temp_year'].map(lambda x: float(x) - 1900)
    df['inflation_rate'] = df['temp_year'].map(lambda x: INFLATION_DF.at[int(x), 'inflation_rate'])
    df['usd_1900'] = df['temp_year'].map(lambda x: INFLATION_DF.at[min(int(x), 120), 'amount'])
    df['usd_1900'].replace(0, 1, inplace=True)
    df['budget_1900'] = df['budget'] / df['usd_1900']
    df = df.drop("temp_year", 1)
    df = df.drop("usd_1900", 1)
    print(df.corr()['revenue'])
    return df
