# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable

import fire
import numpy as np
import pandas as pd
from loguru import logger
import baostock as bs

import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
)

_BAOSTOCK_LOGGED_IN = False


class BaostockCollector(BaseCollector):
    retry = 5

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        bs.login()
        super(BaostockCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol)

    @staticmethod
    @deco_retry
    def get_data_from_remote(symbol, interval, start_datetime, end_datetime):
        global _BAOSTOCK_LOGGED_IN
        if not _BAOSTOCK_LOGGED_IN:
            bs.login()
            _BAOSTOCK_LOGGED_IN = True
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_datetime.strftime("%Y-%m-%d"),
            end_date=end_datetime.strftime("%Y-%m-%d"),
            frequency="d" if interval == "1d" else "5",
            adjustflag="2",  # qfq
        )
        if rs.error_code == "0":
            data = rs.get_data()
            if not data.empty:
                data["symbol"] = symbol
                return data
        return pd.DataFrame()

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.get_data_from_remote(symbol, interval, start_datetime, end_datetime)
        return df

class BaostockCollector1d(BaostockCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BaostockNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_baostock(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(BaostockNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan
        df["change"] = BaostockNormalize.calc_change(df, last_close)
        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan
        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.normalize_baostock(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        df = self.adjusted_price(df)
        return df

    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("rewrite adjusted_price")


class BaostockNormalize1d(BaostockNormalize, ABC):
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjustflag" in df:
            # Baostock's adjustflag is what we need for factor
            df["factor"] = pd.to_numeric(df["adjustflag"], errors='coerce')
        else:
            df["factor"] = 1

        # The price from baostock is already adjusted
        # But we need to calculate the factor for future use
        # factor = adjclose / close
        # So, we need to get unadjusted close first.
        # But baostock with adjustflag=2 (qfq) returns adjusted prices.
        # Let's assume the price is adjusted, and we calculate factor from it.
        # When we use it, we should use the adjusted price directly.
        # To be compatible with qlib, we need to provide ohlcv and factor.
        # The ohlc should be forward adjusted. volume should be divided by factor.

        df["factor"] = df["close"] / pd.to_numeric(df["preclose"], errors='coerce')
        df["factor"] = df["factor"].cumprod()

        for _col in self.COLUMNS:
            if _col not in df.columns:
                continue
            if _col == "volume":
                df[_col] = pd.to_numeric(df[_col], errors='coerce') / df["factor"]
            else:
                # prices are already forward-adjusted
                pass
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")


class BaostockNormalize1dExtend(BaostockNormalize1d):
    def __init__(
        self, old_qlib_data_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        super(BaostockNormalize1dExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(BaostockNormalize1dExtend, self).normalize(df)
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]
        if len(df) > 1:
            new_latest_data = df.iloc[0]
            old_latest_data = old_df.loc[latest_date]
            for col in self.column_list[:-1]: # exclude 'change'
                if pd.notna(new_latest_data[col]) and new_latest_data[col] != 0 and pd.notna(old_latest_data[col]):
                    if col == "volume":
                        df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
                    else:
                        df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        return df.drop(df.index[0]).reset_index() if len(df) > 1 else pd.DataFrame()


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"BaostockCollector{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalize{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    def normalize_data_1d_extend(
        self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        _class = globals()[f"{self.normalize_class_name}Extend"]
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        trading_date: str = None,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 0,
        exists_skip: bool = False,
        max_workers: int = 1,
    ):
        # set max_workers for the instance
        self.max_workers = max_workers

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        if trading_date is None:
            calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
            trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0])).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)

        self.normalize_data_1d_extend(qlib_data_1d_dir)

        _dump = DumpDataUpdate(
            data_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        index_list = ["CSI100", "CSI300", "CSI500"]
        try:
            get_instruments = getattr(
                importlib.import_module(f"data_collector.cn_index.collector"), "get_instruments"
            )
            for _index in index_list:
                get_instruments(str(qlib_data_1d_dir), _index, market_index=f"cn_index")
        except Exception as e:
            logger.warning(f"Failed to update index data: {e}")


if __name__ == "__main__":
    fire.Fire(Run)
    bs.logout()
