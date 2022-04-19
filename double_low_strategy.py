import pandas as pd
import Strategy
import matplotlib.pyplot as plt
import mplfinance as mpf
import math
import datetime
import os
import getopt
import sys


def apply(data_frame: pd.DataFrame, days: int):
    return ''


def read_csv(file, date_col):
    data_frame = pd.read_csv(file, header=0, index_col=0,
                             parse_dates=[date_col],
                             infer_datetime_format=True,
                             keep_date_col=True)
    # groups = data_frame.groupby('SECUCODE')
    # print('Total group: ' + str(groups.ngroups))
    # for code, group in groups:
    #     print('Bode Symbol: ' + str(code))
    #     print(group.first())
    return data_frame


def calculate_premium(df: pd.DataFrame, face_value: int):
    df['Bond Premium'] = df.apply(lambda row: Strategy.calculate_bond_premium(row['CLOSE'], face_value), axis=1)
    df['Convert Premium'] = df.apply(lambda row: Strategy.calculate_convert_premium(row['CLOSE'], row['CONVERTIBLE_PX'],
                                                                                    row['CONVERTIBLE_PX'], face_value), axis=1)
    df['Premium'] = df.apply(lambda row: row['Bond Premium'] + row['Convert Premium'], axis=1)
    return df


def output_results():
    if not os.path.exists('results'):
        os.makedirs('results')
    path = 'results/premium.csv'
    df_premium.to_csv(path, index_label='TRADEDATE', header=True)
    return path


if __name__ == '__main__':
    df = read_csv('data/cbData.csv', 'TRADEDATE')
    df_premium = calculate_premium(df, 100)
    output_results()
    # print(df)
