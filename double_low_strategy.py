import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import math
import datetime
import os
import getopt
import sys


def read_csv(file, date_col):
    return pd.read_csv(file, header=0, index_col=0,
                       parse_dates=[date_col],
                       infer_datetime_format=True,
                       keep_date_col=True)


if __name__ == '__main__':
    df = read_csv('data/cbData.csv', 'TRADEDATE')
    print(df)
