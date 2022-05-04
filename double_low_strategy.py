import pandas as pd
import Strategy
import matplotlib.pyplot as plt
import mplfinance as mpf
import math
import datetime
import os
import getopt
import sys
import numpy as np

import warnings

warnings.filterwarnings('ignore')

"""
策略假设：
（1）无视流动性，Volume=0，也可以交易
（2）不留现金，可以买碎股
    
"""


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
    df['EQUITY_SECUCODE'] = df.apply(lambda row: complete_with_market(row['EQUITY_SECUCODE'], row['SECUMARKET']),
                                     axis=1)


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
            return ori_code + '.SZ'


def get_mapping(df: pd.DataFrame):
    mapping = {}
    for index, row in df.iterrows():
        mapping[row['SECUCODE']] = row['EQUITY_SECUCODE']
    return mapping


def calculate_premium(bond_df: pd.DataFrame, bond_equity_mapping: pd.DataFrame, equity_df: pd.DataFrame,
                      face_value=100):
    """
    Calculate premium of bond and equity
    :param bond_df: bond data with end date (baseline as corresponding equity)
    :param bond_equity_mapping:  bond and equity symbol mapping
    :param equity_df: equity data with start date (baseline as corresponding bond)
    :param face_value: face value of bond - default=100
    :return: consolidated data frame with all calculated premium for all trading days
    """
    print("Processing Market Data and calculating CONVERTIBLE BOND PREMIUM_INDEX...")
    # merge two dfs to collect stock symbol and adjusted close price for a particular trading day
    bond_df = pd.merge(bond_df, bond_equity_mapping.loc[:, ['SECUCODE', 'EQUITY_SECUCODE']], how='left', on='SECUCODE')
    bond_df = pd.merge(bond_df, equity_df, how='left', left_on=['EQUITY_SECUCODE', 'TRADEDATE'],
                       right_on=['S_INFO_WINDCODE', 'TRADE_DT'])

    # calculate premiums based on bond/equity close price, convertible strike price and face value
    bond_df['BOND_PREMIUM'] = bond_df.apply(lambda row: Strategy.calculate_bond_premium(row['CLOSE'], face_value),
                                            axis=1)
    bond_df['CONVERT_PREMIUM'] = bond_df.apply(
        lambda row: Strategy.calculate_convert_premium(row['CLOSE'], row['S_DQ_CLOSE'],
                                                       row['CONVERTIBLE_PX'], face_value), axis=1)
    # bond_df['PREMIUM_INDEX'] = bond_df.apply(lambda row: format(row['BOND_PREMIUM'] + row['CONVERT_PREMIUM'],
    # '.2%'), axis=1)
    bond_df['PREMIUM_INDEX'] = bond_df.apply(lambda row: round(row['BOND_PREMIUM'] + row['CONVERT_PREMIUM'], 4), axis=1)

    print("Data processing finished.\n")
    return bond_df


def output_results(df: pd.DataFrame, subpath: str, filename: str):
    if not os.path.exists('results'):
        os.makedirs('results')
    path = 'results/' + subpath + '/' + filename + '.csv'
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


def trade_marked(premium_index_df: pd.DataFrame, number: int, interval_days: int, TRADEMODE: int):
    """
    在premium_index_df基础上，标识每个换仓日交易股票的TradeMark(未考虑Volume = 0情况)
    :param premium_index_df:经过清晰的每日行情数据,dataframe
    :param number:换仓的股票数量,int
    :param interval_days: 换仓日间隔天数,int
    :param TRADEMODE: 1 换仓日增量买入:2 换仓日全量卖出后rebalance买入,bool
    :return: data framework
    """

    print("Picking up daily target stocks from 2012-12-31...")
    print("     Trading interval = %d DAYS" % interval_days)
    print("     Maximum holdings = %d STOCKS" % number)
    print("     TRADEMODE = MODE %d \n" % TRADEMODE)
    # ** 考虑到15-16年的可转债数量很小，从201701开始执行 **
    # premium_index_df = premium_index_df[premium_index_df['TRADE_DT']>'2016-12-31']
    premium_index_df.reset_index()

    # 获取每天前N个premium最低的股票，并根据日期分组 (此处没有考虑Volume = 0情况)
    premium_daily_df = premium_index_df.groupby('TRADE_DT').apply(lambda x: x.sort_values(['PREMIUM_INDEX']))
    daily_trade_group = premium_daily_df.groupby('TRADEDATE')

    # 变量初始化
    daily_full_list = [[] for i in range(2000)]  # 每日全量行情股票列表
    daily_premium_list = [[] for i in range(2000)]  # 每日持仓股票列表
    date_list = [[] for i in range(2000)]  # 每日持仓股票列表
    daily_sell_list = [[] for i in range(2000)]  # 每日卖出股票列表
    daily_buy_list = [[] for i in range(2000)]  # 每日买出股票列表
    daily_holding_list = [[] for i in range(2000)]  # 每日持仓股票列表

    account_balance = [[] for i in range(2000)]  # 每日持仓市值
    buy_stock_df = pd.DataFrame()  # 每日买入Dataframe
    sell_stock_df = pd.DataFrame()  # 每日卖出Dataframe
    holdings_stock_df = pd.DataFrame()  # 每日持仓Dataframe

    # 每日交易列表并统一日期格式
    i = 0
    for name, group in daily_trade_group:
        daily_full_list[i] = list(group['SECUCODE'])  # 每天全量 用于比对是否到期
        daily_premium_list[i] = list(group['SECUCODE'])  # 每天最低的number个
        daily_premium_list[i] = daily_premium_list[i][:number]
        date_list[i] = group['TRADE_DT'].iloc[0]
        i = i + 1

    # 每日对股票进行MARK
    # 换仓日可以买入、卖出股票
    # 其他日只能卖出到期的股票，持有现金一直到下一个换仓日

    for j in range(0, len(daily_trade_group), 1):
        if j == 0:
            daily_sell_list[j] = []
            daily_buy_list[j] = daily_premium_list[j]
            daily_holding_list[j] = daily_premium_list[j]
        elif (j % interval_days) != 0:  # 赎回日当天可以卖出，不可买入
            daily_holding_list[j] = daily_holding_list[j - 1]  # 继承承T-1日的持仓
            daily_sell_list[j] = list(set(daily_holding_list[j]).difference(
                set((daily_full_list[j + 1]))))  # 非换仓日，检查持仓如果不在明天Full list里，则不能再持有下去，则卖出
            daily_buy_list[j] = []  # 非换仓日，不做买入交易
            daily_holding_list[j] = list(set(daily_holding_list[j] + daily_buy_list[j]) - set(daily_sell_list[j]))
        else:
            # TRADEMODE 1 换仓日增量买入
            if TRADEMODE == 1:
                daily_holding_list[j] = daily_holding_list[j - 1]  # 继承承T-1日的持仓
                daily_sell_list[j] = list(set(daily_holding_list[j]).difference(set((daily_premium_list[j])))) + list(
                    set(daily_holding_list[j]).difference(
                        set((daily_full_list[j + 1]))))  # 换仓日，检查持仓，如果今天不在premium list里，或者不在明天Full List里，则卖出
                daily_buy_list[j] = list(
                    set(daily_premium_list[j]).difference(set((daily_holding_list[j]))))  # 换仓日，买入premium list里里没有的
                daily_holding_list[j] = list(set(daily_holding_list[j] + daily_buy_list[j]) - set(daily_sell_list[j]))
            # TRADEMODE 2 换仓日全量卖出后rebalance买入
            else:
                daily_holding_list[j] = daily_holding_list[j - 1]  # 继承承T-1日的持仓
                daily_sell_list[j] = daily_holding_list[j]  # 换仓日，持仓全部卖出
                daily_buy_list[j] = list(set(daily_premium_list[j]))  # 换层日，重新全部买入
                daily_holding_list[j] = daily_buy_list[j]

        # List转为Set,计算holding list
        daily_buy_set = set(daily_buy_list[j])
        daily_sell_set = set(daily_sell_list[j])
        daily_holding_set = set(daily_holding_list[j])
        print("         Marking stocks of %s" % (date_list[j]))
        # 每日更新买入状态（只在换仓日）
        for buy_stock in daily_buy_set:
            temp_buy_df = premium_index_df[
                (premium_index_df["TRADE_DT"] == date_list[j]) & (premium_index_df["SECUCODE"] == buy_stock)]
            if not temp_buy_df.empty:
                temp_buy_df.loc[:, 'BUY_MARK'] = '1'
                buy_stock_df = pd.concat([buy_stock_df, temp_buy_df])

        # 每日更新卖出状态
        for sell_stock in daily_sell_set:
            temp_sell_df = premium_index_df[
                (premium_index_df["TRADE_DT"] == date_list[j]) & (premium_index_df["SECUCODE"] == sell_stock)]
            if not temp_sell_df.empty:
                temp_sell_df.loc[:, 'SELL_MARK'] = '1'
                sell_stock_df = pd.concat([sell_stock_df, temp_sell_df])

        # 每日持仓更新状态
        for holding_stock in daily_holding_set:
            temp_holdings_df = premium_index_df[
                (premium_index_df["TRADE_DT"] == date_list[j]) & (premium_index_df["SECUCODE"] == holding_stock)]
            if not temp_holdings_df.empty:
                temp_holdings_df.loc[:, 'HOLDINGS'] = '1'
                holdings_stock_df = pd.concat([holdings_stock_df, temp_holdings_df])

    # 合并每个交易日的买入、卖出交易，并合并到全量数据表，按照交易日排序
    trade_marked_df = pd.merge(buy_stock_df, sell_stock_df, how='outer')
    trade_marked_df = pd.merge(trade_marked_df, holdings_stock_df, how='outer')
    trade_marked_df = pd.merge(premium_index_df, trade_marked_df, how='left')
    trade_marked_df.sort_values(by="TRADEDATE", inplace=True, ignore_index=True)

    print("Picking up finished.\n")
    return trade_marked_df


def equal_value_trade(trade_marked_df: pd.DataFrame, initial_cash: float):
    # 采用“等价值”的策略执行交易,并计算输出净值 (允许买碎股)
    # :param daily_trade_df:每次换仓日交易列表,dataframe
    # :param number:初始投入金额,float
    # :return: 生成每日资产表

    print("Start Trading...")
    print("     Initial asset = RMB %.2f .\n" % initial_cash)

    pf_nav_df = pd.DataFrame({'TRADEDATE': [], 'CASH_VALUE': [], 'STOCK_VALUE': [], 'TOTAL_ASSET': [], 'NAV': []})
    stock_holdings_df = pd.DataFrame({'SECUCODE': [], 'HLD_AMT': [], 'PRICE': []})  # 持仓表

    cash_value = [[] for i in range(2000)]  # 每日现金
    stock_value = [[] for i in range(2000)]  # 每日股票市值
    total_asset = [[] for i in range(2000)]  # 每日总资产
    pf_nav = [[] for i in range(2000)]  # 每日NAV
    cash_value[0] = initial_cash
    stock_set = pd.DataFrame()  # 临时变量

    # 新增持仓数量HLD_AMT，并按每天对数据进行分组处理
    trade_marked_df['HLD_AMT'] = ''
    trade_marked_gp = trade_marked_df.groupby(['TRADE_DT'], as_index=False)

    # 更新trade_marked_df每天的持股状态
    i = 0
    for name, group in trade_marked_gp:

        print("Executing Trade on %s..." % name)

        # 首日买入
        if i == 0:
            count = int(group["BUY_MARK"].count())
            buy_cash = float(cash_value[i] / count)
            for index, row in group.iterrows():
                if row['BUY_MARK'] == 1 or row['BUY_MARK'] == '1':
                    trade_marked_df['HLD_AMT'][index] = round(buy_cash / row['CLOSE'], 2)
                    stock_holdings_df.loc[index, 'SECUCODE'] = str(trade_marked_df['SECUCODE'][index])
                    stock_holdings_df.loc[index, 'HLD_AMT'] = trade_marked_df['HLD_AMT'][index]
                    stock_holdings_df.loc[index, 'PRICE'] = trade_marked_df['CLOSE'][index]
            cash_value[i] = 0

        # 最后一日保持持股状态
        elif i == len(trade_marked_gp) - 1:
            cash_value[i] = cash_value[i - 1]
            total_asset[i] = total_asset[i - 1]

        else:
            # 前一天现金结余
            cash_value[i] = cash_value[i - 1]

            # 前一天持仓信息同步
            if group["HOLDINGS"].count() != 0:
                for index, row in group.iterrows():
                    if ((row['HOLDINGS'] == 1 or row['HOLDINGS'] == '1') and not (
                            row['BUY_MARK'] == 1 or row['BUY_MARK'] == '1')):
                        trade_marked_df['HLD_AMT'][index] = \
                            stock_holdings_df[stock_holdings_df['SECUCODE'] == str(row['SECUCODE'])]['HLD_AMT'].values
                        stock_holdings_df.loc[
                            stock_holdings_df[stock_holdings_df['SECUCODE'] == str(row['SECUCODE'])].index[
                                0], 'PRICE'] = trade_marked_df['CLOSE'][index]

            # 先执行卖出操作
            if group["SELL_MARK"].count() != 0:
                for index, row in group.iterrows():
                    if row['SELL_MARK'] == 1 or row['SELL_MARK'] == '1':
                        hld_amt = float(
                            stock_holdings_df[stock_holdings_df['SECUCODE'] == str(row['SECUCODE'])]['HLD_AMT'].values)
                        cash_value[i] = round(cash_value[i] + (row['CLOSE'] * hld_amt), 2)
                        # 在两张表里的持仓数据清0
                        trade_marked_df['HLD_AMT'][index] = 0
                        stock_holdings_df.drop(
                            index=stock_holdings_df[stock_holdings_df['SECUCODE'] == str(row['SECUCODE'])].index[0],
                            inplace=True)

            # 再执行买入操作
            if group["BUY_MARK"].count() != 0:
                count = int(group["BUY_MARK"].count())
                buy_cash = float(cash_value[i] / count)
                for index, row in group.iterrows():
                    if row['BUY_MARK'] == 1 or row['BUY_MARK'] == '1':
                        if cash_value[i] == 0:
                            trade_marked_df['HLD_AMT'][index] = 0
                        else:
                            trade_marked_df['HLD_AMT'][index] = round(buy_cash / row['CLOSE'], 2)

                        stock_holdings_df.loc[index, 'SECUCODE'] = str(trade_marked_df['SECUCODE'][index])
                        stock_holdings_df.loc[index, 'HLD_AMT'] = trade_marked_df['HLD_AMT'][index]
                        stock_holdings_df.loc[index, 'PRICE'] = trade_marked_df['CLOSE'][index]
                cash_value[i] = 0

        stock_holdings_df.dropna(how="any", inplace=True)
        stock_holdings_df.reset_index(drop=True, inplace=True)

        print("Stock Holdings List on %s：" % (name))
        print(stock_holdings_df)
        stock_set['subsum'] = stock_holdings_df.apply(lambda row: round(row['PRICE'] * row['HLD_AMT'], 2), axis=1)
        stock_value[i] = round(stock_set['subsum'].sum(), 2)
        total_asset[i] = round(stock_value[i] + cash_value[i], 2)
        pf_nav[i] = round(total_asset[i] / initial_cash, 4)

        # 输出每日净值表pf_nav
        pf_nav_df.loc[i] = [name, cash_value[i], stock_value[i], total_asset[i], pf_nav[i]]

        i = i + 1

        # 输出每日持仓文件trade_df.csv
    print("Trading finished.")
    if TRADEMODE == 1:
        output_results(trade_marked_df, 'mode1', 'daily_trade_records')
    else:
        output_results(trade_marked_df, 'mode2', 'daily_trade_records')
    print("Output daily trade records file: daily_trade_records.csv successfully.")

    return pf_nav_df


if __name__ == '__main__':
    HOLDING_AMOUNT = 20  # 选股个数
    INTERVAL_DAYS = 20  # 换仓日间隔（交易日）
    INITIAL_ASSET = 10000  # 初始资金
    TRADEMODE = 1  # 交易模式

    print("[Start Processing]")
    print("Cleaning Market Data...")

    cb_df = truncate_by_date(end_date_filter('20201009', 'TRADEDATE'), read_csv('data/cbData.csv',  # file path
                                                                                'TRADEDATE',  # date column
                                                                                None,  # compound index column
                                                                                {'SECUCODE': str,
                                                                                 'TRADEDATE': str}))  # column data type
    mapping_df = read_excel('data/announcement.xlsx',
                            {'SECUCODE': str, 'SECUMARKET': int, 'EQUITY_SECUCODE': str, 'EQUITY_SECUMARKET': int},
                            # column data type
                            mapping_cleansing)  # data cleansing

    print(".......")
    stock_df = truncate_by_date(start_date_filter('20121231', 'TRADE_DT'),
                                read_csv('data/AShareEODPrices.csv',  # file path
                                         'TRADE_DT',  # date column
                                         None,  # compound index column
                                         {'S_INFO_WINDCODE': str, 'TRADE_DT': str},  # column data type
                                         ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE']))  # column needed

    premium_df = calculate_premium(cb_df, mapping_df, stock_df)
    # premium_df = pd.read_csv('results/premium.csv')
    output_results(premium_df, '', 'premium')

    if TRADEMODE == 1:
        tradeMarked_df = trade_marked(premium_df, HOLDING_AMOUNT, INTERVAL_DAYS, TRADEMODE)
        # tradeMarked_df = pd.read_csv('results/mode1/trade_marked.csv')

        pfNav_df = equal_value_trade(tradeMarked_df, INITIAL_ASSET)
        output_results(tradeMarked_df, 'mode1', 'trade_marked')
        output_results(pfNav_df, 'mode1', 'pf_nav')
    else:
        tradeMarked_df = trade_marked(premium_df, HOLDING_AMOUNT, INTERVAL_DAYS, TRADEMODE)
        # tradeMarked_df = pd.read_csv('results/mode2/trade_marked.csv')

        pfNav_df = equal_value_trade(tradeMarked_df, INITIAL_ASSET)
        output_results(tradeMarked_df, 'mode2', 'trade_marked')
        output_results(pfNav_df, 'mode2', 'pf_nav')

    print("Output daily NAV file: pf_nav.csv successfully.")
