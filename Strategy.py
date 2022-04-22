from abc import ABCMeta, abstractmethod
import pandas as pd


class Strategy(metaclass=ABCMeta):

    @abstractmethod
    def select(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def reallocation(self, current_bsk: pd.DataFrame, new_bsk: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def calculate_pnl(self, df: pd.DataFrame):
        raise NotImplementedError


def calculate_bond_premium(bond_close: float, face_value: int):
    premium = bond_close / face_value - 1
    return float('{:.3f}'.format(premium))


def calculate_convert_premium(bond_close: float, stock_price: float, strike_price: float, face_value: int):
    premium = ((bond_close * strike_price) / (face_value * stock_price)) - 1
    return float('{:.3f}'.format(premium))
