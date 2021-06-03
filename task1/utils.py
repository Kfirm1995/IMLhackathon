import matplotlib.pyplot as plt
import numpy as np


def plot_corr_heatmap(df):
    corr = df.corr()
    fig = plt.figure(figsize=(12, 12))
    c = plt.pcolor(corr, cmap='RdBu', vmin=-1, vmax=1)
    plt.xticks(np.arange(0.5, len(corr.columns), 1), df.columns, rotation=40)
    plt.yticks(np.arange(0.5, len(corr.columns), 1), df.columns)
    fig.colorbar(c)
    plt.show()
    plt.waitforbuttonpress(-1)
