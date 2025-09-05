from pathlib import Path
import yaml
import streamlit as st
import pandas as pd
import pickle
import sys
import subprocess
import copy
import io
import json
import gc
from collections import deque
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
from qlib.model.base import Model


class StreamlitLogHandler(io.StringIO):
    """
    A handler to redirect stdout/stderr to a Streamlit placeholder,
    showing only the last N lines.
    """

    def __init__(self, placeholder, max_lines=500):
        super().__init__()
        self.placeholder = placeholder
        self.buffer = deque(maxlen=max_lines)
        self.partial_line = ""

    def write(self, message):
        # Add new data to the partial line
        self.partial_line += message
        # Split into complete lines
        lines = self.partial_line.split("\n")
        # The last element is the new partial line
        self.partial_line = lines.pop()
        # Add complete lines to the buffer
        for line in lines:
            self.buffer.append(line)

        # Update the display
        log_content = "\n".join(self.buffer)
        if self.partial_line:
            log_content += "\n" + self.partial_line
        self.placeholder.code(log_content, language="log")

    def flush(self):
        pass  # Not needed for real-time updates

# --- Decoupled Model and Factor Configurations ---

MODELS = {
    "LightGBM": {
        "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421, "subsample": 0.8789, "n_estimators": 200, "max_depth": 8}
    },
    "XGBoost": {
        "class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 7}
    },
    "CatBoost": {
        "class": "CatBoostModel", "module_path": "qlib.contrib.model.catboost",
        "kwargs": {"iterations": 200, "learning_rate": 0.05, "depth": 7}
    },
    "ALSTM": {
        "class": "ALSTM", "module_path": "qlib.contrib.model.pytorch_alstm_ts",
        "kwargs": {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.5, "n_epochs": 30, "lr": 1e-4, "early_stop": 5}
    }
}

FACTORS = {
    "Alpha158": {
        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
        "kwargs": {"drop_raw": True} # Time and instruments are set dynamically
    },
    "Alpha360": {
        "class": "Alpha360", "module_path": "qlib.contrib.data.handler",
        "kwargs": {"drop_raw": True} # Time and instruments are set dynamically
    }
}

BASE_DATASET = {
    "class": "DatasetH", "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {}, # To be filled dynamically
        "segments": {} # To be filled dynamically
    }
}

# --- Data Management Functions ---
def get_collector_script_path(source: str):
    """Gets the path to the collector script based on the source."""
    app_dir = Path().resolve()
    script_name = "collector.py"
    if source == "yahoo":
        path = app_dir / "scripts" / "data_collector" / "yahoo" / script_name
    elif source == "baostock":
        path = app_dir / "scripts" / "data_collector" / "baostock_1d" / script_name
    else:
        raise ValueError(f"Unknown data source: {source}")

    if not path.exists():
        raise FileNotFoundError(f"Collector script for source '{source}' not found at {path}")
    return str(path)

def get_script_path(script_name):
    """Gets the path for general scripts."""
    app_dir = Path().resolve()
    path = app_dir / "scripts" / script_name
    if not path.exists():
        raise FileNotFoundError(f"Script '{script_name}' not found at {path}")
    return str(path)

def run_command_with_log(command, placeholder, throttle_lines: int = 1, max_lines: int = 500):
    """
    Runs a command and streams its output to a Streamlit placeholder in real-time,
    showing only the last N lines.
    """
    buffer = deque(maxlen=max_lines)
    buffer.append(f"Running command: {command}\n\n")
    placeholder.code("".join(buffer), language="log")

    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
    )

    line_count = 0
    for line in iter(process.stdout.readline, ""):
        buffer.append(line)
        line_count += 1
        # Update the UI every `throttle_lines` to prevent freezing
        if line_count % throttle_lines == 0:
            placeholder.code("".join(buffer), language="log")

    # Ensure the final, complete log is always displayed regardless of throttling
    final_log = "".join(buffer)
    placeholder.code(final_log, language="log")

    process.stdout.close()

    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=final_log)

def update_daily_data(qlib_dir, start_date, end_date, placeholder, source="yahoo"):
    script_path = get_collector_script_path(source)
    # For both yahoo and baostock, we now pass max_workers to speed up the process.
    # The collector scripts have been refactored or verified to handle this parameter.
    max_workers = max(multiprocessing.cpu_count() - 2, 1)
    command = f'"{sys.executable}" "{script_path}" update_data_to_bin --qlib_data_1d_dir "{qlib_dir}" --trading_date {start_date} --end_date {end_date} --max_workers {max_workers}'
    run_command_with_log(command, placeholder)

def check_data_health(qlib_dir, placeholder, n_jobs=1):
    script_path = get_script_path("check_data_health.py")
    command = f'"{sys.executable}" "{script_path}" check_data --qlib_dir "{qlib_dir}" --n_jobs {n_jobs}'
    # Use throttled logging for this high-volume output task
    run_command_with_log(command, placeholder, throttle_lines=20)

def get_data_summary(qlib_dir_str: str):
    """Scans the Qlib data directory and returns a summary of its contents."""
    summary = {
        "date_range": "N/A",
        "instruments": [],
        "fields": [],
        "error": None
    }
    try:
        qlib_dir = Path(qlib_dir_str)
        if not qlib_dir.exists():
            summary["error"] = "指定的Qlib数据路径不存在。"
            return summary

        # Initialize Qlib to use its data API
        import qlib
        qlib.init(provider_uri=qlib_dir_str, expression_cache=None)
        from qlib.data import D

        # Get date range from calendar
        calendar = D.calendar()
        if calendar is not None and len(calendar) > 0:
            start_date = pd.to_datetime(calendar[0]).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(calendar[-1]).strftime('%Y-%m-%d')
            summary["date_range"] = f"{start_date} to {end_date}"

        # Get instrument list
        instruments_dir = qlib_dir / "instruments"
        if instruments_dir.exists():
            summary["instruments"] = [f.stem for f in instruments_dir.glob("*.txt")]

        # Get fields list from a sample stock
        features_dir = qlib_dir / "features"
        if features_dir.exists():
            # Find the first stock directory that is not a special file
            sample_stock_dir = next((d for d in features_dir.iterdir() if d.is_dir()), None)
            if sample_stock_dir:
                summary["fields"] = [f.stem for f in sample_stock_dir.glob("*.bin")]

        if not summary["date_range"] and not summary["instruments"] and not summary["fields"]:
             summary["error"] = "指定的路径不是一个有效的Qlib数据目录，或者目录为空。"

    except Exception as e:
        summary["error"] = f"扫描数据时发生错误: {e}"

    return summary

# --- Model Training & Evaluation Functions (FIXED) ---
def train_model(
    qlib_dir: str, models_save_dir: str,
    model_name: str, factor_name: str, stock_pool: str,
    segments: dict, model_params: dict = None,
    custom_model_name: str = None, finetune_model_path: str = None, log_placeholder=None
):
    import qlib
    from qlib.utils import init_instance_by_config
    qlib.auto_init(provider_uri=qlib_dir)

    # --- Dynamically Build Config ---
    model_config = copy.deepcopy(MODELS[model_name])
    if model_params:
        model_config["kwargs"].update(model_params)

    handler_config = copy.deepcopy(FACTORS[factor_name])
    handler_config["kwargs"]["start_time"] = segments["train"][0]
    handler_config["kwargs"]["end_time"] = segments["test"][1]
    handler_config["kwargs"]["fit_start_time"] = segments["train"][0]
    handler_config["kwargs"]["fit_end_time"] = segments["train"][1]
    handler_config["kwargs"]["instruments"] = stock_pool

    dataset_config = copy.deepcopy(BASE_DATASET)
    dataset_config["kwargs"]["handler"] = handler_config
    dataset_config["kwargs"]["segments"] = segments

    task_config = {"model": model_config, "dataset": dataset_config}
    # --- End of Config Build ---

    dataset = init_instance_by_config(task_config["dataset"])

    if finetune_model_path:
        initial_model = Model.load(finetune_model_path)
        model_config['kwargs']['init_model'] = initial_model

    model = init_instance_by_config(task_config["model"])

    # Redirect stdout/stderr to the Streamlit placeholder if provided
    log_stream = StreamlitLogHandler(log_placeholder) if log_placeholder else io.StringIO()
    with redirect_stdout(log_stream), redirect_stderr(log_stream):
        print("--- 模型训练开始 ---")
        model.fit(dataset)
        print("--- 模型训练结束 ---")

    training_log = log_stream.buffer if isinstance(log_stream, StreamlitLogHandler) else log_stream.getvalue()

    if custom_model_name:
        model_basename = custom_model_name
    else:
        train_end_date = segments['train'][1]
        model_basename = f"{model_name}_{factor_name}_{stock_pool}_{train_end_date}"

    model_save_path = Path(models_save_dir).expanduser() / f"{model_basename}.pkl"
    config_save_path = Path(models_save_dir).expanduser() / f"{model_basename}.yaml"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    task_config["model"]["kwargs"].pop('init_model', None)

    with open(model_save_path, 'wb') as f: pickle.dump(model, f)
    with open(config_save_path, 'w') as f: yaml.dump(task_config, f)

    # --- Memory Cleanup ---
    # Explicitly delete large objects and run garbage collection
    try:
        del model, dataset
        if 'initial_model' in locals():
            del initial_model
        gc.collect()
        training_log += "\n[INFO] Memory cleanup complete."
    except Exception as e:
        training_log += f"\n[WARNING] Error during memory cleanup: {e}"

    return str(model_save_path), training_log

# --- Settings Persistence ---
CONFIG_FILE = "config.json"

def save_settings(settings: dict):
    """Saves settings to a JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        st.error(f"Error saving settings: {e}")

def load_settings() -> dict:
    """Loads settings from a JSON file."""
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load settings: {e}")
            return {}
    return {}

def predict(model_path_str: str, qlib_dir: str, prediction_date: str):
    import qlib
    from qlib.utils import init_instance_by_config
    qlib.auto_init(provider_uri=qlib_dir)

    model_path = Path(model_path_str)
    config_path = model_path.with_suffix(".yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found for model {model_path.name}")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = Model.load(model_path)
    config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = pd.to_datetime(prediction_date) - pd.DateOffset(years=2)
    config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = prediction_date
    config["dataset"]["kwargs"]["segments"]["test"] = (prediction_date, prediction_date)
    dataset = init_instance_by_config(config["dataset"])
    prediction = model.predict(dataset, segment="test")
    prediction.name = 'score'
    prediction = prediction.reset_index().rename(columns={'instrument': 'StockID', 'datetime': 'Date'})
    return prediction.sort_values(by="score", ascending=False)


def get_historical_prediction(model_path_str: str, qlib_dir: str, stock_id: str, start_date: str, end_date: str, placeholder=None):
    # This can be slow as it predicts day by day
    all_scores = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Business days

    for i, date in enumerate(date_range):
        date_str = date.strftime("%Y-%m-%d")
        if placeholder:
            placeholder.text(f"正在预测第 {i+1}/{len(date_range)} 天: {date_str}")
        try:
            pred_df = predict(model_path_str, qlib_dir, date_str)
            stock_score = pred_df[pred_df['StockID'] == stock_id]
            if not stock_score.empty:
                all_scores.append({'Date': date, 'Score': stock_score.iloc[0]['score']})
        except Exception as e:
            # Skip days where prediction fails (e.g., no data, market closed)
            print(f"Could not predict for {date_str}: {e}")
            continue

    return pd.DataFrame(all_scores)

def get_model_info(model_path_str: str):
    """
    Reads the YAML config file associated with a model and returns info.
    """
    info = {"stock_pool": "N/A", "error": None}
    try:
        model_path = Path(model_path_str)
        config_path = model_path.with_suffix(".yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Safely navigate the dictionary
        stock_pool = config.get("dataset", {}).get("kwargs", {}).get("handler", {}).get("kwargs", {}).get("instruments")
        if stock_pool:
            info["stock_pool"] = stock_pool
        else:
            info["error"] = "Stock pool information not found in config."

    except Exception as e:
        info["error"] = str(e)

    return info

def evaluate_model(model_path_str: str, qlib_dir: str, log_placeholder=None, test_period: tuple = None):
    """
    Evaluates a model using the high-level functions from `qlib.contrib.report`
    as requested by the user. This function now generates Plotly figures directly.
    """
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    import qlib.contrib.report as qcr

    log_stream = StreamlitLogHandler(log_placeholder) if log_placeholder else io.StringIO()
    with redirect_stdout(log_stream), redirect_stderr(log_stream):
        print("--- 模型评估开始 (使用 qlib.contrib.report) ---")
        qlib.auto_init(provider_uri=qlib_dir)

        # --- 1. Load Model, Config, and Dataset ---
        model_path = Path(model_path_str)
        config_path = model_path.with_suffix(".yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found for model {model_path.name}")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        model = Model.load(model_path)

        # If a test_period is provided by the user, use it. Otherwise, use the one from the config file.
        if test_period is None:
            # No override provided, use the test period from the model's original config
            test_period = config["dataset"]["kwargs"]["segments"]["test"]
        else:
            # If the test_period is manually provided, we need to update the dataset's configuration
            # to ensure the data loaded by the handler matches the new evaluation period.
            config["dataset"]["kwargs"]["segments"]["test"] = test_period
            # The handler's start and end time must also be updated to cover the new period.
            # We provide a 2-year buffer before the start date for feature calculation lookback.
            start_date_buffer = pd.to_datetime(test_period[0]) - pd.DateOffset(years=2)
            config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = start_date_buffer.strftime("%Y-%m-%d")
            config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = test_period[1]

        dataset = init_instance_by_config(config["dataset"])
        print(f"模型和配置已加载。测试期: {test_period[0]} to {test_period[1]}")

        # --- 2. Prepare pred_label DataFrame (This is the fix) ---
        print("\n--- [1/3] 准备预测数据 (pred_label) ---")
        # Get predictions (returns a Series with MultiIndex)
        prediction_df = model.predict(dataset, segment="test")
        prediction_df.name = 'score'

        # Get labels (returns a DataFrame with MultiIndex) from the same dataset
        pred_label_df_raw = dataset.prepare("test", col_set="label")

        # Qlib's default label name is not always 'label'. Rename it for compatibility.
        if len(pred_label_df_raw.columns) == 1 and pred_label_df_raw.columns[0] != 'label':
            pred_label_df_raw.rename(columns={pred_label_df_raw.columns[0]: 'label'}, inplace=True)

        # Defensive merging: reset index on both and merge on columns.
        # This is a robust way to avoid errors from incompatible indexes.
        pred_df_for_merge = prediction_df.reset_index()
        label_df_for_merge = pred_label_df_raw.reset_index()

        # Merge on the common columns 'datetime' and 'instrument'
        pred_label_df = pd.merge(
            label_df_for_merge, pred_df_for_merge, on=["datetime", "instrument"], how="inner"
        )

        # Restore the multi-index, which is expected by qlib's reporting functions
        pred_label_df = pred_label_df.set_index(["datetime", "instrument"]).dropna()

        # After merging, if the dataframe is empty, it means the data did not align.
        # This is a critical check to give a user-friendly error message.
        if pred_label_df.empty:
            raise ValueError(
                "无法将模型预测值与标签数据对齐 (After merging, the DataFrame is empty). "
                "这通常意味着测试集的时间范围或股票池配置有误，导致没有重叠的数据。"
            )
        print("预测数据准备完成。")

        # --- 3. Generate Signal Analysis Figures ---
        print("\n--- [2/3] 生成信号分析报告 ---")
        # Use the high-level reporting function as requested
        from qlib.contrib.report.analysis_model import analysis_model_performance
        signal_figs = analysis_model_performance.model_performance_graph(
            pred_label_df, show_notebook=False
        )
        print("信号分析报告生成完毕。")

        # --- 4. End of Evaluation ---
        print("信号分析报告生成完毕。")

    eval_log = log_stream.getvalue()

    # Consolidate results into a dictionary for the frontend
    results = {
        "signal_figures": signal_figs,
    }

    # Clean up memory
    del model, dataset, pred_label_df, prediction_df
    gc.collect()

    return results, eval_log

def run_backtest_and_analysis(model_path_str: str, qlib_dir: str, start_time: str, end_time: str, strategy_kwargs: dict, exchange_kwargs: dict):
    """
    Runs a comprehensive backtest and analysis, generating all necessary data for the UI in one go.
    """
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    import qlib.contrib.report as qcr
    import plotly.express as px

    qlib.auto_init(provider_uri=qlib_dir)

    # --- 1. Load Model, Config, and Dataset ---
    model_path = Path(model_path_str)
    config_path = model_path.with_suffix(".yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found for model {model_path.name}")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = Model.load(model_path)

    config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = start_time
    config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = end_time
    config["dataset"]["kwargs"]["segments"] = {"test": (start_time, end_time)}
    dataset = init_instance_by_config(config["dataset"])

    # --- 2. Instantiate Strategy and Run Backtest ---
    strategy = TopkDropoutStrategy(model=model, dataset=dataset, **strategy_kwargs)
    report_df, positions_df = backtest_daily(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        exchange_kwargs=exchange_kwargs
    )

    # --- 3. Generate Performance Metrics (KPIs) ---
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_df["return"] - report_df["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"])
    analysis["return_with_cost"] = risk_analysis(report_df["return"] - report_df["cost"])
    analysis_df = pd.concat(analysis)

    # --- 4. Generate Visualizations ---
    # Main Equity Curve
    equity_curve_fig = px.line(
        report_df.rename(columns={'account': '策略', 'bench': '基准'}),
        x=report_df.index,
        y=['策略', '基准'],
        title="策略 vs. 基准 (扣除成本后)"
    )

    # Risk Analysis Figures for positions
    risk_analysis_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"])
    risk_figures = qcr.analysis_position.risk_analysis_graph(risk_analysis_df, report_df, show_notebook=False)

    # --- 5. Consolidate and Return Results ---
    results = {
        "report_df": report_df,
        "analysis_df": analysis_df,
        "positions_df": positions_df,
        "equity_curve_fig": equity_curve_fig,
        "risk_figures": risk_figures,
    }

    # --- 6. Memory Cleanup ---
    del model, dataset, report_df, positions_df, analysis_df
    gc.collect()

    return results
