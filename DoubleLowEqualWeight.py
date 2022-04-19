import Strategy
import pandas as pd


class DoubleLowEqualWeight(Strategy):

    def __init__(self, rolling_days: int):
        self.rolling_days = rolling_days

    def select(self, raw_df: pd.DataFrame):
        pass

