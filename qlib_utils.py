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
from contextlib import redirect_stdout, redirect_stderr

class StreamlitLogHandler(io.StringIO):
    """A handler to redirect stdout/stderr to a Streamlit placeholder."""
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        # Use a monospace font that supports Chinese characters well
        self.placeholder.code(self.buffer, language='log')

    def flush(self):
        # This method is called by some libraries, but we handle updates in write.
        pass

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
def get_script_path(script_name):
    app_dir = Path().resolve()
    script_path = app_dir / "scripts" / script_name
    if not script_path.exists():
        script_path = app_dir / "scripts" / "data_collector" / "yahoo" / script_name
        if not script_path.exists():
             raise FileNotFoundError(f"Script '{script_name}' not found. Please ensure the 'scripts' folder from the Qlib GitHub repository is in the same directory as the application.")
    return str(script_path)

def run_command_with_log(command, log_key):
    """Runs a command and streams its output to a streamlit session state variable."""
    st.session_state[log_key] = f"Running command: {command}\n\n"
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    for line in iter(process.stdout.readline, ''):
        st.session_state[log_key] += line
    process.stdout.close()
    if process.wait() != 0:
        # Append the full log to the exception for better debugging
        error_output = st.session_state[log_key]
        raise subprocess.CalledProcessError(process.returncode, command, output=error_output)

def update_daily_data(qlib_dir, start_date, end_date, log_key):
    script_path = get_script_path("collector.py")
    command = f'"{sys.executable}" "{script_path}" update_data_to_bin --qlib_data_1d_dir "{qlib_dir}" --trading_date {start_date} --end_date {end_date}'
    run_command_with_log(command, log_key)

def check_data_health(qlib_dir, log_key):
    script_path = get_script_path("check_data_health.py")
    command = f'"{sys.executable}" "{script_path}" check_data --qlib_dir "{qlib_dir}"'
    run_command_with_log(command, log_key)

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
        with open(finetune_model_path, 'rb') as f:
            initial_model = pickle.load(f)
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
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = pd.to_datetime(prediction_date) - pd.DateOffset(years=2)
    config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = prediction_date
    config["dataset"]["kwargs"]["segments"]["test"] = (prediction_date, prediction_date)
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
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = start_time
    config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = end_time
    config["dataset"]["kwargs"]["segments"]["test"] = (start_time, end_time)
    dataset = init_instance_by_config(config["dataset"])
    strategy = TopkDropoutStrategy(model=model, dataset=dataset, **strategy_kwargs)

    # The backtest_daily function now primarily returns the daily portfolio results
    report_df, _ = backtest_daily(start_time=start_time, end_time=end_time, strategy=strategy, exchange_kwargs=exchange_kwargs)

    # We need to manually calculate the analysis metrics using risk_analysis
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_df["return"] - report_df["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"])
    analysis["return"] = risk_analysis(report_df["return"])

    analysis_df = pd.concat(analysis)  # This will be a DataFrame with metrics

    # Return both the daily report for plotting and the analysis report for metrics
    return report_df, analysis_df

def get_historical_prediction(model_path_str: str, qlib_dir: str, stock_id: str, start_date: str, end_date: str):
    # This can be slow as it predicts day by day
    all_scores = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Business days

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
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

def evaluate_model(model_path_str: str, qlib_dir: str, log_placeholder=None):
    """
    Evaluate a trained model using qlib's standard analysis recorders.
    Returns a dictionary containing signal analysis and portfolio analysis results.
    """
    import qlib
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

    log_stream = StreamlitLogHandler(log_placeholder) if log_placeholder else io.StringIO()
    with redirect_stdout(log_stream), redirect_stderr(log_stream):
        print("--- 模型评估开始 ---")
        qlib.auto_init(provider_uri=qlib_dir)

        model_path = Path(model_path_str)
        config_path = model_path.with_suffix(".yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found for model {model_path.name}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print("为保证评估顺利进行，临时创建`drop_raw=False`的数据集...")
        eval_dataset_config = copy.deepcopy(config["dataset"])
        eval_dataset_config["kwargs"]["handler"]["kwargs"]["drop_raw"] = False
        dataset_for_eval = init_instance_by_config(eval_dataset_config)
        print("数据集创建成功。")

        with R.start(experiment_name="model_evaluation_streamlit", recorder_name="InMemoryRecorder", resume=True):
            recorder = R.get_recorder()

            # 1. Generate prediction and run Signal Analysis
            print("\n--- [1/2] 开始信号分析 (Signal Analysis) ---")
            sr = SignalRecord(model, dataset_for_eval, recorder)
            sr.generate()

            # Load the prediction from the recorder, which is now an artifact
            prediction_df = recorder.load_object("pred.pkl")

            sar = SigAnaRecord(recorder, ana_long_short=False)
            sar.generate()
            signal_report = recorder.load_object("sig_ana/report_normal.pkl")
            print("--- 信号分析完成 ---")

            # 2. Run Portfolio Analysis using the generated prediction
            print("\n--- [2/2] 开始投资组合分析 (Portfolio Analysis) ---")

            # Dynamically set benchmark from config
            instruments = config["dataset"]["kwargs"]["handler"]["kwargs"].get("instruments", "csi300")
            if instruments == "csi300":
                benchmark = "SH000300"
            elif instruments == "csi500":
                benchmark = "SH000905"
            else:
                benchmark = "SH000300"

            test_period = config["dataset"]["kwargs"]["segments"]["test"]

            port_analysis_config = {
                "strategy": {
                    "class": "TopkDropoutStrategy",
                    "module_path": "qlib.contrib.strategy",
                    "kwargs": {
                        "signal": prediction_df, # Pass the loaded prediction DataFrame
                        "topk": 50,
                        "n_drop": 5,
                    },
                },
                "backtest": {
                    "start_time": test_period[0],
                    "end_time": test_period[1],
                    "account": 100000000,
                    "benchmark": benchmark,
                    "exchange_kwargs": {
                        "freq": "day",
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                },
            }

            par = PortAnaRecord(recorder, port_analysis_config, "day")
            par.generate()
            portfolio_report = recorder.load_object("port_ana/report_normal.pkl")
            print("--- 投资组合分析完成 ---")

    eval_log = log_stream.buffer if isinstance(log_stream, StreamlitLogHandler) else log_stream.getvalue()
    return {"signal": signal_report, "portfolio": portfolio_report}, eval_log
