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
    max_workers = max(multiprocessing.cpu_count() - 2, 1)
    command = f'"{sys.executable}" "{script_path}" update_data_to_bin --qlib_data_1d_dir "{qlib_dir}" --trading_date {start_date} --end_date {end_date} --max_workers {max_workers}'
    run_command_with_log(command, placeholder)

def check_data_health(qlib_dir, placeholder, n_jobs=1):
    script_path = get_script_path("check_data_health.py")
    command = f'"{sys.executable}" "{script_path}" check_data --qlib_dir "{qlib_dir}" --n_jobs {n_jobs}'
    run_command_with_log(command, placeholder, throttle_lines=20)

def get_data_summary(qlib_dir_str: str):
    summary = {"date_range": "N/A", "instruments": [], "fields": [], "error": None}
    try:
        qlib_dir = Path(qlib_dir_str)
        if not qlib_dir.exists():
            summary["error"] = "指定的Qlib数据路径不存在。"
            return summary
        import qlib
        qlib.init(provider_uri=qlib_dir_str, expression_cache=None)
        from qlib.data import D
        calendar = D.calendar()
        if calendar is not None and len(calendar) > 0:
            summary["date_range"] = f"{pd.to_datetime(calendar[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(calendar[-1]).strftime('%Y-%m-%d')}"
        instruments_dir = qlib_dir / "instruments"
        if instruments_dir.exists():
            summary["instruments"] = [f.stem for f in instruments_dir.glob("*.txt")]
        features_dir = qlib_dir / "features"
        if features_dir.exists():
            sample_stock_dir = next((d for d in features_dir.iterdir() if d.is_dir()), None)
            if sample_stock_dir:
                summary["fields"] = [f.stem for f in sample_stock_dir.glob("*.bin")]
        if not summary["date_range"] and not summary["instruments"] and not summary["fields"]:
             summary["error"] = "指定的路径不是一个有效的Qlib数据目录，或者目录为空。"
    except Exception as e:
        summary["error"] = f"扫描数据时发生错误: {e}"
    return summary

def train_model(
    qlib_dir: str, models_save_dir: str, model_name: str, factor_name: str, stock_pool: str,
    segments: dict, model_params: dict = None, custom_model_name: str = None,
    finetune_model_path: str = None, log_placeholder=None
):
    import qlib
    from qlib.utils import init_instance_by_config
    qlib.auto_init(provider_uri=qlib_dir)
    model_config = copy.deepcopy(MODELS[model_name])
    if model_params:
        model_config["kwargs"].update(model_params)
    handler_config = copy.deepcopy(FACTORS[factor_name])
    handler_config["kwargs"].update({
        "start_time": segments["train"][0], "end_time": segments["test"][1],
        "fit_start_time": segments["train"][0], "fit_end_time": segments["train"][1],
        "instruments": stock_pool
    })
    dataset_config = copy.deepcopy(BASE_DATASET)
    dataset_config["kwargs"]["handler"] = handler_config
    dataset_config["kwargs"]["segments"] = segments
    task_config = {"model": model_config, "dataset": dataset_config}
    dataset = init_instance_by_config(task_config["dataset"])
    if finetune_model_path:
        initial_model = Model.load(finetune_model_path)
        model_config['kwargs']['init_model'] = initial_model
    model = init_instance_by_config(task_config["model"])
    log_stream = StreamlitLogHandler(log_placeholder) if log_placeholder else io.StringIO()
    with redirect_stdout(log_stream), redirect_stderr(log_stream):
        print("--- 模型训练开始 ---")
        model.fit(dataset)
        print("--- 模型训练结束 ---")
    training_log = log_stream.getvalue()
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
    try:
        del model, dataset
        if 'initial_model' in locals():
            del initial_model
        gc.collect()
        training_log += "\n[INFO] Memory cleanup complete."
    except Exception as e:
        training_log += f"\n[WARNING] Error during memory cleanup: {e}"
    return str(model_save_path), training_log

CONFIG_FILE = "config.json"

def save_settings(settings: dict):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        st.error(f"Error saving settings: {e}")

def load_settings() -> dict:
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load settings: {e}")
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
    config["dataset"]["kwargs"]["segments"] = {"test": (prediction_date, prediction_date)}
    dataset = init_instance_by_config(config["dataset"])
    prediction = model.predict(dataset, segment="test")
    prediction.name = 'score'
    prediction = prediction.reset_index().rename(columns={'instrument': 'StockID', 'datetime': 'Date'})
    return prediction.sort_values(by="score", ascending=False)

def backtest_strategy(model_path_str: str, qlib_dir: str, start_time: str, end_time: str, strategy_kwargs: dict, exchange_kwargs: dict):
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    qlib.auto_init(provider_uri=qlib_dir)
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
    strategy = TopkDropoutStrategy(model=model, dataset=dataset, **strategy_kwargs)
    report_df, _ = backtest_daily(start_time=start_time, end_time=end_time, strategy=strategy, exchange_kwargs=exchange_kwargs)
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_df["return"] - report_df["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"])
    analysis_df = pd.concat(analysis)
    return report_df, analysis_df

def get_historical_prediction(model_path_str: str, qlib_dir: str, stock_id: str, start_date: str, end_date: str, placeholder=None):
    all_scores = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
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
            print(f"Could not predict for {date_str}: {e}")
            continue
    return pd.DataFrame(all_scores)

def get_model_test_period(model_path_str: str):
    model_path = Path(model_path_str)
    config_path = model_path.with_suffix(".yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    test_period = config.get("dataset", {}).get("kwargs", {}).get("segments", {}).get("test")
    if not test_period:
        raise ValueError("Test period not found in model config.")
    return test_period

def get_model_info(model_path_str: str):
    info = {"stock_pool": "N/A", "error": None, "test_period": ("N/A", "N/A")}
    try:
        test_period = get_model_test_period(model_path_str)
        info["test_period"] = test_period
        model_path = Path(model_path_str)
        config_path = model_path.with_suffix(".yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        stock_pool = config.get("dataset", {}).get("kwargs", {}).get("handler", {}).get("kwargs", {}).get("instruments")
        if stock_pool:
            info["stock_pool"] = stock_pool
    except Exception as e:
        info["error"] = str(e)
    return info

def evaluate_model(model_path_str: str, qlib_dir: str, log_placeholder=None):
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    import qlib.contrib.report as qcr
    log_stream = StreamlitLogHandler(log_placeholder) if log_placeholder else io.StringIO()
    with redirect_stdout(log_stream), redirect_stderr(log_stream):
        print("--- 模型评估开始 (使用 qlib.contrib.report) ---")
        qlib.auto_init(provider_uri=qlib_dir)
        model_path = Path(model_path_str)
        config_path = model_path.with_suffix(".yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found for model {model_path.name}")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = Model.load(model_path)
        test_period = config["dataset"]["kwargs"]["segments"]["test"]
        config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = test_period[0]
        config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = test_period[1]
        dataset = init_instance_by_config(config["dataset"])
        print(f"模型和配置已加载。自动使用测试期: {test_period[0]} to {test_period[1]}")
        print("\n--- [1/3] 准备预测数据 (pred_label) ---")
        prediction_s = model.predict(dataset, segment="test")
        prediction_s.name = 'score'
        label_dataset_config = copy.deepcopy(config["dataset"])
        label_dataset_config["kwargs"]["handler"]["kwargs"]["drop_raw"] = False
        label_dataset = init_instance_by_config(label_dataset_config)
        label_df = label_dataset.prepare("test", col_set=["label"])
        label_s = label_df.iloc[:, 0]
        pred_label_df = pd.DataFrame({'score': prediction_s, 'label': label_s})
        pred_label_df.dropna(inplace=True)
        print("预测数据准备完成。")
        print("\n--- [2/3] 生成信号分析报告 ---")
        signal_figs = qcr.analysis_model.model_performance_graph(pred_label_df, show_notebook=False)
        print("信号分析报告生成完毕。")
        print("\n--- [3/3] 生成投资组合分析报告 ---")
        strategy_kwargs = {"topk": 50, "n_drop": 5, "signal": prediction_s}
        strategy_for_eval = TopkDropoutStrategy(**strategy_kwargs)
        report_df, _ = backtest_daily(start_time=test_period[0], end_time=test_period[1], strategy=strategy_for_eval)
        portfolio_figs = qcr.analysis_position.report_graph(report_df, show_notebook=False)
        analysis_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"])
        print("投资组合分析报告生成完毕。")
    eval_log = log_stream.getvalue()
    results = {
        "signal_figures": signal_figs,
        "portfolio_figures": portfolio_figs,
        "risk_analysis_table": pd.DataFrame(analysis_df).T,
        "raw_report_df": report_df
    }
    del model, dataset, label_dataset, pred_label_df, prediction_s, label_s
    gc.collect()
    return results, eval_log

def get_position_analysis(model_path_str: str, qlib_dir: str, start_time: str, end_time: str, strategy_kwargs: dict, exchange_kwargs: dict):
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    import qlib.contrib.report as qcr
    qlib.auto_init(provider_uri=qlib_dir)
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
    strategy = TopkDropoutStrategy(model=model, dataset=dataset, **strategy_kwargs)
    report_df, positions_df = backtest_daily(start_time=start_time, end_time=end_time, strategy=strategy, exchange_kwargs=exchange_kwargs)
    analysis_df = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"])
    risk_figs = qcr.analysis_position.risk_analysis_graph(pd.DataFrame(analysis_df).T, report_df, show_notebook=False)
    results = {
        "positions": positions_df,
        "report": report_df,
        "risk_figures": risk_figs,
        "analysis_df": pd.DataFrame(analysis_df).T
    }
    return results
