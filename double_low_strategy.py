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


def read_csv(file, date_col, index_list, data_type_map, col_used=[], clean_function=None):
    """
    Read CSV function
    :param file: absolute file path
    :param date_col: date column
    :param index_list: compound index column list
    :param data_type_map: data type map
    :param col_used: the column list that is needed to avoid huge data import. default None - All columns
    :param clean_function: the data cleansing function, default None - No cleansing needed
    :return: data framework
    """
    data_frame = pd.read_csv(file, header=0,
                             index_col=None,
                             usecols=col_used if col_used else None,
                             dtype=data_type_map,
                             parse_dates=[date_col],
                             infer_datetime_format=True,
                             keep_date_col=True)
    # groups = data_frame.groupby('SECUCODE')
    # print('Total group: ' + str(groups.ngroups))
    # for code, group in groups:
    #     print('Bode Symbol: ' + str(code))
    #     print(group.first())
    if index_list:
        data_frame.reset_index(inplace=True)
        data_frame.set_index(index_list, inplace=True)
    if clean_function:
        clean_function(data_frame)
    return data_frame


def read_excel(file, data_type_map, clean_function=None):
    """
    Read Excel function
    :param file: absolute file path
    :param data_type_map: data type map
    :param clean_function: the data cleansing function, default None - No cleansing needed
    :return:
    """
    data_framework = pd.read_excel(file, dtype=data_type_map, header=0)
    if clean_function:
        clean_function(data_framework)
    return data_framework


def mapping_cleansing(df: pd.DataFrame):
    # drop the row with EQUITY_SECUCODE=0
    df.drop(df[df.EQUITY_SECUCODE == '0'].index, inplace=True)

    # data cleaning - Add market suffix and prefix (leading zeros) to security code
    df['EQUITY_SECUCODE'] = df.apply(lambda row: complete_with_market(row['EQUITY_SECUCODE'], row['SECUMARKET']), axis=1)


def complete_with_market(ori_code, market):
    length = len(ori_code)
    if market == 83:  # Shanghai Exchange
        return ori_code + '.SH'
    else:  # Shenzhen Exchange
        if length < 6:  # need to complete the root code
            if length < 4:  # main board
                while length < 6:
                    ori_code = '0' + ori_code
                    length = len(ori_code)
                return ori_code + '.SZ'
            else:
                return '00' + ori_code + '.SZ'
        else:
            return ori_code + 'SZ'


def get_mapping(df: pd.DataFrame):
    mapping = {}
    for index, row in df.iterrows():
        mapping[row['SECUCODE']] = row['EQUITY_SECUCODE']
    return mapping


def calculate_premium(bond_df: pd.DataFrame, bond_equity_mapping: pd.DataFrame, equity_df: pd.DataFrame, face_value=100):
    """
    Calculate premium of bond and equity
    :param bond_df: bond data with end date (baseline as corresponding equity)
    :param bond_equity_mapping:  bond and equity symbol mapping
    :param equity_df: equity data with start date (baseline as corresponding bond)
    :param face_value: face value of bond - default=100
    :return: consolidated data frame with all calculated premium for all trading days
    """
    # merge two dfs to collect stock symbol and adjusted close price for a particular trading day
    bond_df = pd.merge(bond_df, bond_equity_mapping.loc[:, ['SECUCODE', 'EQUITY_SECUCODE']], how='left', on='SECUCODE')
    bond_df = pd.merge(bond_df, equity_df, how='left', left_on=['EQUITY_SECUCODE', 'TRADEDATE'], right_on=['S_INFO_WINDCODE', 'TRADE_DT'])

    # calculate premiums based on bond/equity close price, convertible strike price and face value
    bond_df['BOND_PREMIUM'] = bond_df.apply(lambda row: Strategy.calculate_bond_premium(row['CLOSE'], face_value), axis=1)
    bond_df['CONVERT_PREMIUM'] = bond_df.apply(lambda row: Strategy.calculate_convert_premium(row['CLOSE'], row['S_DQ_CLOSE'],
                                                                                              row['CONVERTIBLE_PX'], face_value), axis=1)
    bond_df['PREMIUM_INDEX'] = bond_df.apply(lambda row: format(row['BOND_PREMIUM'] + row['CONVERT_PREMIUM'], '.2%'), axis=1)
    return bond_df


def output_results(df: pd.DataFrame):
    if not os.path.exists('results'):
        os.makedirs('results')
    path = 'results/premium.csv'
    df.to_csv(path, header=True)
    return path


def truncate_by_date(filter_function, df: pd.DataFrame):
    return df[(filter_function(df))]


def start_date_filter(date_str: str, date_col: str):
    s_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    return lambda x: x[date_col].dt.date >= s_date


def end_date_filter(date_str: str, date_col: str):
    e_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    return lambda x: x[date_col].dt.date <= e_date


if __name__ == '__main__':
    cb_df = truncate_by_date(end_date_filter('20201009', 'TRADEDATE'), read_csv('data/cbData.csv',  # file path
                                                                                'TRADEDATE',  # date column
                                                                                None,  # compound index column
                                                                                {'SECUCODE': str, 'TRADEDATE': str}))  # column data type
    mapping_df = read_excel('data/announcement.xlsx',
                            {'SECUCODE': str, 'SECUMARKET': int, 'EQUITY_SECUCODE': str, 'EQUITY_SECUMARKET': int},  # column data type
                            mapping_cleansing)  # data cleansing
    stock_df = truncate_by_date(start_date_filter('20121231', 'TRADE_DT'),
                                read_csv('D:/OneDrive/SAIF/2022Spring/IFL-Stock/Wind数据/AShareEODPrices.csv',  # file path
                                         'TRADE_DT',  # date column
                                         None,  # compound index column
                                         {'S_INFO_WINDCODE': str, 'TRADE_DT': str},  # column data type
                                         ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE']))  # column needed
    premium_df = calculate_premium(cb_df, mapping_df, stock_df)
    output_results(premium_df)
    print(mapping_df.head(5))
