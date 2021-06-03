import datetime
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import pickle
from ast import literal_eval
import math
from task1.utils import *

# Global Members
top_original_dic = json.load(open("memory_maps/top_original_languages.json"))
rev_dic = json.load(open("memory_maps/company_id_map_to_rev.json"))
vote_dic = json.load(open("memory_maps/company_id_map_to_vote.json"))
top_actor_set = pickle.load(open("memory_maps/top_actor_set.pickle", 'rb'))
top_director_dic = json.load(open("memory_maps/top_director.json", 'r'))
inflation_df = pd.read_csv("inflation_data.csv")



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
    df = handle_budget(df)
    df = handle_genres(df)
    df = handle_homepage(df)
    df = handle_original_languages(df)
    df = handle_release_date(df, stage)
    df = handle_original_title(df)
    df = handle_overview(df)
    df = handle_vote_count(df, stage)
    df = handle_production_companies(df)
    df = handle_production_countries(df)
    df = handle_runtime(df)
    df = handle_spoken_languages(df)
    df = handle_status(df, stage)
    df = handle_tagline(df)
    df = handle_title(df)
    df = handle_keywords(df)
    df = handle_cast(df)
    df = handle_crew(df)
    df = handle_revenue(df)
    # df = handle_inflation(df)
    df.to_csv("parsed_dataset.csv")
    y_revenue = df.revenue
    y_vote_avg = df.vote_average
    df = df.drop(['revenue', 'vote_average'], axis=1)
    return df, y_revenue, y_vote_avg




def handle_first(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage == 'train':
        df = df[df['revenue'].notna()]
        df = df.drop_duplicates()
        df = df[df['revenue'] >= 1000]

    return df


def handle_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("id", 1)
    return df


def handle_belongs_to_collection(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    df[["belongs_to_collection"]] = df[["belongs_to_collection"]].notnull().astype(int)
    return df


def handle_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    leave budget as it is
    :param df:
    :return:
    """
    # todo add log budget
    df['log_budget'] = df['budget'].map(lambda x: 0 if float(x) < 2 else math.log(float(x)))
    # df['budget/runtime'] = df['budget'] / df['runtime']
    # df = df[df['budget/runtime'].notna()]    ##  TODO!!!!!!!!!!!!!!!!!!!
    return df


def handle_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    encode one hot by genres
    :param df:
    :return:
    """
    df['genres'] = df['genres'].fillna("[]")
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
              'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War',
              'Western']
    for g in genres:
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
    # todo check most occurrences of languages and change to 1 just on them
    df['original_language'] = df['original_language'].map(lambda x: 1 if x in top_original_dic else 0)
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


def handle_vote_count(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    leave vote_count as it is
    :param df:
    :param stage:
    :return:
    """
    if stage == 'train':
        df = df[df['runtime'].notna()]

    # df['vote_count/runtime'] = df['vote_count'] / df['runtime']  # todo ensure predict wont get it
    return df


def handle_production_companies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a map from company of production to average vote and average revenue.
    :param df: data.
    :return: data['company'] is dropped and instead df['company_id_vote_batch'] and df['company_id_revenue_batch']
    are added
    """

    def score_by_json_st(json_st, dic: dict):
        company_id_row = re.sub("[^0-9]", "", json_st.split(',')[0])
        if company_id_row == '':
            return 1
        return dic[company_id_row]

    df['company_id_revenue_batch'] = df['production_companies'].map(lambda x: score_by_json_st(x, rev_dic))
    df['company_id_vote_batch'] = df['production_companies'].map(lambda x: score_by_json_st(x, vote_dic))
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

    df['release_date'] = pd.to_datetime(df['release_date'])
    df['quarter'] = df['release_date'].dt.quarter
    df['month'] = pd.DatetimeIndex(df['release_date']).month
    # plot_corr_heatmap(df)
    df['year'] = pd.DatetimeIndex(df['release_date']).year

    # adding decade
    df['decade'] = df['year'].map(lambda x: x - x % 5)

    # adding days passed released
    today = datetime.datetime.now()
    df['days_from_release'] = df['release_date'].map(lambda x: (today - pd.to_datetime(x)).days)

    df = df.drop("release_date", 1)
    # df = df.drop("year", 1)

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
    # df = change_cast_to_actor(df)
    actors = list(top_actor_set)
    df['cast'] = df['cast'].fillna("[]")
    for actor in actors:
        df[actor] = df['cast'].apply(lambda x: 1 if actor in x else 0)
    df = df.drop('cast', axis=1)
    return df


def handle_crew(df: pd.DataFrame) -> pd.DataFrame:
    df['crew'] = df['crew'].fillna("[]")
    for d in top_director_dic.keys():
        df[d] = df['crew'].apply(lambda x: math.ceil(top_director_dic[d]/25) if d in x else 0)
    df = df.drop('crew', axis=1)
    return df


def handle_revenue(df: pd.DataFrame) -> pd.DataFrame:
    return df


# Done Handling

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


def handle_inflation(df: pd.DataFrame) -> pd.DataFrame:
    df_a = inflation_df
    df['temp_year'] = df['year'].map(lambda x: float(x) - 1900)
    df['inflation_rate'] = df['temp_year'].map(lambda x: inflation_df.at[int(x), 'inflation_rate'])


    df['usd_1900'] = df['temp_year'].map(lambda x: inflation_df.at[int(x), 'amount'])
    df['budget_1900'] = df['budget'] / df['usd_1900']

    df['usd_1900'] = df['temp_year'].map(lambda x: inflation_df.at[int(x), 'amount'])
    df = df.drop("temp_year", 1)
    df = df.drop("usd_1900", 1)
    return df


# -------------------------- Crew dummies : director ------------------------

def get_directors(x):
    """classes the directors to classes ,2 is very good 1 is good, 0 is bad"""
    for i in x:
        if i['job'] == 'Director':
            name = i['name']
            if (name in top_director_dic):
                return int(math.ceil(top_director_dic[name] / 25))
    return 0


def change_crew_to_directors(df):
    """change crew to classes of top directors columns with dummy values
    (0 -bad, 1 - good, 2 - very good"""
    df = df[pd.notnull(df['crew'])]  # TODO: ROY!!!!!!
    df['crew'] = df['crew'].apply(literal_eval)
    df['crew'] = df['crew'].apply(get_directors)
    # Change the field ‘crew’ to ‘director’
    df.rename(columns={'crew': 'director'}, inplace=True)

    return df


# -------------------------- Cast dummies : actor ------------------------

def get_top_actor_list(x):
    """change the cast column to list of top actors (from 200 actors) """
    all_actors = ['temp']
    for i in x:
        if i['known_for_department'] == 'Acting':
            name = i['name']
            if (name in top_actor_set):
                all_actors.append(i['name'])
    return all_actors


def encode_one_hot_actor(df, feature: str):
    """change cast list to dummies of 0 and 1 for each top actor
    will have now 193 new columns"""
    genres_df = pd.DataFrame(df[feature].tolist())
    stacked_genres = genres_df.stack()
    raw_dummies = pd.get_dummies(stacked_genres)
    genre_dummies = raw_dummies.sum(level=0)
    df = pd.concat([df, genre_dummies], axis=1)
    return df


def change_cast_to_actor(df):
    """change cast to 200 top actors columns with dummy values"""
    df = df[pd.notnull(df['cast'])]
    df['cast'] = df['cast'].apply(literal_eval)
    df['cast'] = df['cast'].apply(get_top_actor_list)
    df = encode_one_hot_actor(df, 'cast')
    df = df.drop("cast", 1)
    df = df.drop("temp", 1)
    return df
