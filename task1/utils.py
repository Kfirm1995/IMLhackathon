import matplotlib.pyplot as plt
import numpy as np
import json
import re
import pandas as pd

def plot_corr_heatmap(df):
    corr = df.corr()
    print(corr['revenue'])
    print(corr['vote_average'])

    fig = plt.figure(figsize=(12, 12))
    c = plt.pcolor(corr, cmap='RdBu', vmin=-1, vmax=1)
    plt.xticks(np.arange(0.5, len(corr.columns), 1), corr.columns, rotation=40)
    plt.yticks(np.arange(0.5, len(corr.columns), 1), corr.columns)
    fig.colorbar(c)
    plt.show()
    plt.waitforbuttonpress(-1)


def write_ordinal(df):
    """
    ugly function that writes memory maps.
    """
    bad_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        company_id_row = re.sub("[^0-9]", "", row['production_companies'].split(',')[0])
        if company_id_row == '':
            bad_rows.append(i)
    df = df.drop(bad_rows, axis=0)
    y_rev = df['revenue']
    y_vote_avg = df['vote_average']
    col = df['production_companies']
    company_id_col = col.apply(lambda x: int(re.sub("[^0-9]", "", x.split(',')[0])))
    arr_rev = list(zip(y_rev, company_id_col))
    arr_vote_avg = list(zip(y_vote_avg, company_id_col))
    dic_rev_avg = {}
    dic_vote_avg = {}
    for rev, comp in arr_rev:
        if comp not in dic_rev_avg:
            dic_rev_avg[comp] = [rev, 1]
        else:
            dic_rev_avg[comp][0] += rev
            dic_rev_avg[comp][1] += 1
    arr = [(k, v[0]/v[1]) for k, v in dic_rev_avg.items()]
    ids = [t[0] for t in arr]
    revs = [t[1] for t in arr]
    splits = np.linspace(min(revs), max(revs), 3)
    ans = {}
    for item in arr:
        company_id = item[0]
        rev = item[1]
        for i in range(len(splits) - 1):
            if splits[i] <= rev <= splits[i+1]:
                ans[company_id] = i + 1
                break
    with open('memory_maps/company_id_map_to_rev.json', 'w') as fp:
        json.dump(ans, fp)

    for vote, comp in arr_vote_avg:
        if comp not in dic_vote_avg:
            dic_vote_avg[comp] = [vote, 1]
        else:
            dic_vote_avg[comp][0] += vote
            dic_vote_avg[comp][1] += 1
    arr = [(k, v[0]/v[1]) for k, v in dic_rev_avg.items()]
    ids = [t[0] for t in arr]
    revs = [t[1] for t in arr]
    splits = np.linspace(min(revs), max(revs), 3)
    ans = {}
    for item in arr:
        company_id = item[0]
        rev = item[1]
        for i in range(len(splits) - 1):
            if splits[i] <= rev <= splits[i+1]:
                ans[company_id] = i + 1
                break
    with open('memory_maps/company_id_map_to_vote.json', 'w') as fp:
        json.dump(ans, fp)
    exit(0)


def get_top_n_freq_values(df:pd.DataFrame, n: int, feature: str):
    return df[feature].value_counts()[:n].index.tolist()

def write_best_langs(df):
    dict_rev = {}
    for i in range(len(df)):
        for lang in df.iloc[i]['new']:
            if lang['iso_639_1'] in dict_rev:
                dict_rev[lang['iso_639_1']][0] += df.iloc[i]['revenue']
                dict_rev[lang['iso_639_1']][1] += 1
            else:
                dict_rev[lang['iso_639_1']] = [df.iloc[i]['revenue'], 1]
    dict_rev = {key: val[0]/val[1] for key, val in dict_rev.items()}
    print(sorted(list(dict_rev.items()), key=lambda tup: tup[1])[-10:-1])
    exit(0)


    pass

