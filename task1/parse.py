import pandas as pd
import numpy as np
from ast import literal_eval
import re
import json

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


def write_ordinal(df, col, delim=','):
    y = df['revenue']
    bad_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        company_id_row = re.sub("[^0-9]", "", row['production_companies'].split(',')[0])
        if company_id_row == '':
            bad_rows.append(i)
    df = df.drop(bad_rows, axis=0)


    col = df[col]
    company_id_col = col.apply(lambda x: int(re.sub("[^0-9]", "", x.split(',')[0])))



    arr = list(zip(y, company_id_col))
    dic = {}
    for rev, comp in arr:
        if comp not in dic:
            dic[comp] = [rev, 1]
        else:
            dic[comp][0] += rev
            dic[comp][1] += 1
    arr = [(k, v[0]/v[1]) for k, v in dic.items()]
    ids = [t[0] for t in arr]
    revs = [t[1] for t in arr]
    splits = np.linspace(min(revs), max(revs), 10)
    ans = {}
    for item in arr:
        company_id = item[0]
        rev = item[1]
        for i in range(len(splits) - 1):
            if splits[i] <= rev <= splits[i+1]:
                ans[company_id] = i + 1
                break
    # print(ans)
    with open('company_id_map.json', 'w') as fp:
        json.dump(ans, fp)












def clean_data(df: pd.DataFrame):
    # dropping duplicates
    #
    # write_ordinal(df, 'production_companies')
    df = df.drop_duplicates()
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

    # drop original title

    # production companies
    # df = get_dummies_for_uniques(df, 'production_companies') //todo improve

    # y
    y = df.revenue

    return df, y


def handle_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop("id", 1)


def handle_belongs_to_collection(df: pd.DataFrame) -> pd.DataFrame:
    df[["is_belongs_to_collection"]] = df[["belongs_to_collection"]].notnull().astype(int)
    return df


def handle_budget(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_genres(df: pd.DataFrame) -> pd.DataFrame:
    return get_dummies_for_uniques(df, 'genres')


def handle_homepage(df: pd.DataFrame) -> pd.DataFrame:
    df['homepage'] = df['homepage'].map(lambda x: 1 if '.com' in str(x) else 0)
    return df


def handle_original_languages(df: pd.DataFrame) -> pd.DataFrame:
    return encode_one_hot(df, 'original_language')


def handle_original_title(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_overview(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_vote_average(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_vote_count(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_production_companies(df: pd.DataFrame) -> pd.DataFrame:
    # dic = json.load(open("company_id_map.json"))
    # bad_rows = []
    # for i in range(len(df)):
    #     row = df.iloc[i]
    #     company_id_row = re.sub("[^0-9]", "", row['production_companies'].split(',')[0])
    #     if company_id_row == '':
    #         bad_rows.append(i)
    # df = df.drop(bad_rows, axis=0)
    #
    # col = df[col]
    # company_id_col = col.apply(lambda x: int(re.sub("[^0-9]", "", x.split(',')[0])))

    return df


def handle_production_countries(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_release_date(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_runtime(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_spoken_languages(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_status(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_tagline(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_title(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_keywords(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_cast(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_crew(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_release_date(df: pd.DataFrame) -> pd.DataFrame:
    return df


def handle_revenue(df: pd.DataFrame) -> pd.DataFrame:
    return df


###########################

def get_dummies_for_uniques(df, feature: str):
    df['new'] = df[feature].apply(literal_eval)
    df['names'] = df['new'] \
        .apply(lambda x: [e['name'] for e in x] if isinstance(x, list) else [])
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
