import numpy as np
import pandas as pd
from task1.parse import  *
if __name__ == '__main__':
    print("Hello world")
    df = load_data("sample_set.csv")
    X, y = clean_data(df)
    print(X)
    print(y)



