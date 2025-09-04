from loguru import logger
import os
from typing import Optional, List, Dict, Any
import fire
import pandas as pd
import qlib
from tqdm import tqdm
from qlib.data import D
from joblib import Parallel, delayed

# --- Helper function for parallel execution ---

def check_instrument_data(
    instrument: str,
    freq: str,
    required_fields: List[str],
    large_step_threshold_price: float,
    large_step_threshold_volume: float,
    missing_data_num: int,
) -> Dict[str, Any]:
    """
    Checks a single instrument for data completeness and correctness.
    This function is designed to be called in parallel.
    """
    problems = {"instrument": instrument, "missing_data": {}, "large_steps": [], "missing_columns": [], "missing_factor": None}

    try:
        df = D.features([instrument], required_fields, freq=freq)
        if df.empty:
            problems["missing_data"] = {col: "all" for col in required_fields}
            return problems

        df.rename(
            columns={
                "$open": "open",
                "$close": "close",
                "$low": "low",
                "$high": "high",
                "$volume": "volume",
                "$factor": "factor",
            },
            inplace=True,
        )

        # 1. Check for required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            problems["missing_columns"] = missing_cols

        # 2. Check for missing data in existing columns
        missing_counts = df.isnull().sum()
        missing_data_cols = missing_counts[missing_counts > missing_data_num]
        if not missing_data_cols.empty:
            problems["missing_data"] = missing_data_cols.to_dict()

        # 3. Check for large step changes
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                pct_change = df[col].pct_change(fill_method=None).abs()
                threshold = large_step_threshold_volume if col == "volume" else large_step_threshold_price
                if pct_change.max() > threshold:
                    large_steps = pct_change[pct_change > threshold]
                    problems["large_steps"].append(
                        {
                            "col_name": col,
                            "date": large_steps.index.to_list()[0][1].strftime("%Y-%m-%d"),
                            "pct_change": pct_change.max(),
                        }
                    )

        # 4. Check for missing factor
        if "factor" not in df.columns:
            problems["missing_factor"] = "column_missing"
        elif df["factor"].isnull().all():
            problems["missing_factor"] = "all_nan"

    except Exception as e:
        problems["error"] = str(e)

    return problems


class DataHealthChecker:
    """
    Checks a dataset for data completeness and correctness in parallel.
    """

    def __init__(
        self,
        qlib_dir: str,
        freq: str = "day",
        large_step_threshold_price: float = 0.5,
        large_step_threshold_volume: float = 3.0,
        missing_data_num: int = 0,
    ):
        self.qlib_dir = qlib_dir
        self.freq = freq
        self.large_step_threshold_price = large_step_threshold_price
        self.large_step_threshold_volume = large_step_threshold_volume
        self.missing_data_num = missing_data_num

        qlib.init(provider_uri=self.qlib_dir)

    def check_data(self, n_jobs: int = 1, limit_nums: Optional[int] = None):
        """
        Main method to run the data health check.

        Args:
            n_jobs (int): Number of parallel jobs to run. -1 means use all available cores.
            limit_nums (Optional[int]): Limit the number of instruments to check for debugging.
        """
        logger.info(f"Starting data health check with {n_jobs} parallel jobs...")

        instruments = D.instruments(market="all")
        instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)

        if limit_nums is not None:
            instrument_list = instrument_list[:limit_nums]
            logger.info(f"Checking a limited number of {len(instrument_list)} instruments.")

        required_fields = ["$open", "$close", "$low", "$high", "$volume", "$factor"]

        results = Parallel(n_jobs=n_jobs)(
            delayed(check_instrument_data)(
                instrument=inst,
                freq=self.freq,
                required_fields=required_fields,
                large_step_threshold_price=self.large_step_threshold_price,
                large_step_threshold_volume=self.large_step_threshold_volume,
                missing_data_num=self.missing_data_num,
            )
            for inst in tqdm(instrument_list, desc="Dispatching instrument checks")
        )

        self.summarize_results(results)

    def summarize_results(self, results: List[Dict[str, Any]]):
        """Aggregates and prints the problems found during the check."""
        logger.info("Aggregating results...")

        problems_found = False
        summary = {
            "missing_data": [],
            "large_steps": [],
            "missing_columns": [],
            "missing_factor": [],
            "errors": [],
        }

        for res in results:
            instrument = res["instrument"]
            if res.get("error"):
                summary["errors"].append({"instrument": instrument, "error": res["error"]})
                problems_found = True
            if res.get("missing_data"):
                for col, count in res["missing_data"].items():
                    summary["missing_data"].append({"instrument": instrument, "column": col, "missing_count": count})
                problems_found = True
            if res.get("large_steps"):
                for step_info in res["large_steps"]:
                    summary["large_steps"].append({"instrument": instrument, **step_info})
                problems_found = True
            if res.get("missing_columns"):
                summary["missing_columns"].append({"instrument": instrument, "missing_columns": res["missing_columns"]})
                problems_found = True
            if res.get("missing_factor"):
                summary["missing_factor"].append({"instrument": instrument, "reason": res["missing_factor"]})
                problems_found = True

        print("\n" + "=" * 50)
        print(f"Data Health Check Summary ({len(results)} files checked)")
        print("=" * 50 + "\n")

        if not problems_found:
            logger.success("✅ All checks passed. No issues found.")
            return

        if summary["missing_data"]:
            logger.warning("Found missing data points:")
            print(pd.DataFrame(summary["missing_data"]).to_string(index=False))
            print("-" * 50)

        if summary["large_steps"]:
            logger.warning("Found large step changes:")
            print(pd.DataFrame(summary["large_steps"]).to_string(index=False))
            print("-" * 50)

        if summary["missing_columns"]:
            logger.warning("Found missing required columns (OHLCV):")
            print(pd.DataFrame(summary["missing_columns"]).to_string(index=False))
            print("-" * 50)

        if summary["missing_factor"]:
            logger.warning("Found missing factor data:")
            print(pd.DataFrame(summary["missing_factor"]).to_string(index=False))
            print("-" * 50)

        if summary["errors"]:
            logger.error("Encountered errors during checking:")
            print(pd.DataFrame(summary["errors"]).to_string(index=False))
            print("-" * 50)


if __name__ == "__main__":
    fire.Fire(DataHealthChecker)
