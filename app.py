import streamlit as st
import os
# Suppress the GitPython warning
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import qlib
from qlib.constant import REG_CN
from pathlib import Path
from qlib_utils import (
    MODELS, FACTORS, train_model, predict, backtest_strategy,
    update_daily_data, check_data_health, get_data_summary, get_historical_prediction,
    evaluate_model, load_settings, save_settings, get_model_info, get_position_analysis,
    get_model_test_period
)
import pandas as pd
import plotly.express as px
import datetime
import copy

# --- Streamlit Pages ---

def data_management_page():
    st.header("数据管理")
    with st.expander("💡 操作指南 (Operation Guide)"):
        st.markdown("""
        **本页面负责为Qlib准备和维护数据。高质量的数据是量化研究的基石。**
        **- 核心作用:**
          - **初始化数据**: 为首次使用的用户提供一个清晰、稳定的数据部署流程。
          - **日常更新**: 让用户可以方便地将本地数据更新到最新的交易日。
          - **数据质检**: 提供一个工具来检查本地数据的完整性和连续性，以避免在后续研究中出现因数据问题导致的错误。
        **- 推荐使用流程:**
          1. **首次使用**:
             - **强烈建议**按照“全量数据部署”中的指引，在终端中手动执行命令来下载和解压由社区维护的数据包。这是最快、最稳定的方式。
             - 通过`wget`下载后，使用`tar`命令解压到指定目录。
          2. **日常维护**:
             - 如果您已经拥有了全量数据，每天或定期使用“增量更新”功能即可。
             - 选择一个开始日期（通常是上次更新日期的后一天）和结束日期（通常是今天），然后点击“开始增量更新”。下方日志窗口会实时显示更新过程。
          3. **定期检查**:
             - 建议定期（例如每月）运行一次“开始检查数据”，以确保您的数据没有缺失或中断。
        **- 参数解释:**
          - **Qlib数据路径**: 这是Qlib存放所有数据的根目录，包括股票日线、因子等。您可以在左侧边栏根据需要进行修改。
          - **更新开始/结束日期**: 定义了增量更新的时间区间，程序会自动下载并处理这个区间内的所有交易日数据。
        """)

    # Initialize session state for logs
    if "data_log" not in st.session_state:
        st.session_state.data_log = ""

    qlib_dir = st.session_state.settings.get("qlib_data_path", str(Path.home() / ".qlib" / "qlib_data"))
    qlib_1d_dir = str(Path(qlib_dir) / "cn_data")
    st.info(f"当前Qlib数据路径: `{qlib_dir}` (可在左侧边栏修改)")

    st.subheader("本地数据概览")
    summary = get_data_summary(qlib_1d_dir)
    if summary["error"]:
        st.warning(f"无法加载数据概览: {summary['error']}")
    else:
        col1, col2 = st.columns(2)
        col1.metric("数据覆盖范围", summary["date_range"])
        col2.metric("股票池数量", len(summary["instruments"]))
        with st.expander("查看详细信息"):
            st.json({
                "已发现的股票池文件": summary["instruments"],
                "已发现的数据字段": summary["fields"]
            })

    with st.expander("1. 全量数据部署 (首次使用)", expanded=False):
        st.info("由于直接从雅虎财经大量下载数据不稳定，推荐通过以下步骤手动下载社区提供的数据包来完成首次数据部署。")
        st.markdown("""
        **请在您的终端中依次执行以下命令：**
        ```bash
        # 1. 下载社区提供的预处理数据包
        wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
        # 2. 创建用于存放数据的目录 (如果不存在)
        mkdir -p ~/.qlib/qlib_data/cn_data
        # 3. 解压数据包到指定目录
        tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
        # 4. (可选) 清理下载的压缩包
        rm -f qlib_bin.tar.gz
        ```
        """)

    st.subheader("2. 增量更新与健康度检查")
    st.markdown("如果已有全量数据，可在此处更新到指定日期，或检查本地数据的完整性和质量。所有执行日志都会显示在下方。")

    # Data source selection
    st.markdown("#### 选择数据源")

    def update_data_source():
        st.session_state.settings["data_source"] = st.session_state.data_source_selector
        save_settings(st.session_state.settings)

    data_source_options = ["yahoo", "baostock"]
    default_source = st.session_state.settings.get("data_source", "yahoo")
    default_index = data_source_options.index(default_source) if default_source in data_source_options else 0

    st.radio(
        "选择您的数据下载源 (选择后将自动保存为默认)",
        options=data_source_options,
        index=default_index,
        key="data_source_selector",
        horizontal=True,
        on_change=update_data_source,
        help=(
            "**Yahoo**: 覆盖全球市场，但在中国访问速度慢且不稳定。\n\n"
            "**Baostock**: 仅覆盖中国A股，但在中国速度快，数据质量好。"
        )
    )
    data_source = st.session_state.data_source_selector

    with st.container(height=400):
        log_placeholder = st.empty()
        log_placeholder.code("日志输出将显示在此处" , language='log')

    col1, col2 = st.columns(2)
    start_date = col1.date_input("更新开始日期", datetime.date.today() - datetime.timedelta(days=7))
    end_date = col2.date_input("更新结束日期", datetime.date.today())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("开始增量更新", use_container_width=True):
            with st.spinner(f"正在从 {data_source} 更新从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据..."):
                try:
                    update_daily_data(qlib_1d_dir, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), log_placeholder, source=data_source)
                    st.success("增量更新命令已成功执行！")
                except Exception as e:
                    st.error(f"增量更新过程中发生错误。详情请查看上方日志。")

    with col2:
        n_jobs = st.number_input("健康检查并行数 (n_jobs)", -1, 64, -1, help="设置用于并行计算的线程数。-1 表示使用所有可用的CPU核心。")
        if st.button("开始检查数据", use_container_width=True):
            with st.spinner(f"正在并行检查数据 (n_jobs={n_jobs})..."):
                try:
                    check_data_health(qlib_1d_dir, log_placeholder, n_jobs)
                    st.success("数据健康度检查已完成！详情请查看上方日志。")
                except Exception as e:
                    st.error(f"检查过程中发生错误。详情请查看上方日志。")

def model_training_page():
    st.header("模型训练")
    with st.expander("💡 操作指南 (Operation Guide)"):
        st.markdown("""
        **本页面是进行量化模型训练的核心功能区。**
        **- 核心作用:**
          - **模型训练**: 基于选择的因子（特征）和股票池，训练一个机器学习或深度学习模型，用以预测未来的股票收益率。
          - **增量学习**: 在已有的旧模型基础上，使用新的数据进行增量训练（Finetune），以达到让模型与时俱进的目的。
          - **参数调优**: 提供界面让用户可以方便地调整模型的关键超参数，以探索最佳的模型配置。
        **- 推荐使用流程:**
          1. **选择模式**:
             - 如果是第一次训练，或希望用全新的参数训练，选择“从零开始新训练”。
             - 如果希望在之前训练好的模型上继续学习，选择“在旧模型上继续训练”，并选择一个已存在的`.pkl`模型文件。
          2. **配置模型**:
             - **选择模型**: 选择一个您希望使用的算法，如`LightGBM`（速度快，效果好）或`ALSTM`（深度学习，更复杂）。
             - **选择因子**: 因子是模型的输入特征。`Alpha158`和`Alpha360`是Qlib提供的两套经典因子组合。
             - **输入股票池名称**: 输入您的数据对应的股票池名称，例如`csi300`。请确保您本地有该股票池的数据。
          3. **设置时间**:
             - 合理地划分训练集、验证集和测试集。三者之间时间不能重叠，且要符合**训练 -> 验证 -> 测试**的先后顺序。
          4. **调节超参数**:
             - 对于GBDT类模型，您可以调整并行线程数(`n_jobs`设为-1可使用全部CPU核心以加速)、树的数量、深度、学习率等。好的超参数对模型效果至关重要。
          5. **开始训练**:
             - 点击“开始训练”，下方日志区会实时展示训练过程。训练结束后，模型文件（`.pkl`）和配置文件（`.yaml`）会自动保存在您设置的模型路径中。
        **- 关于GPU加速的特别说明:**
          - **前提条件**: 要成功使用"尝试使用GPU加速"选项，您的计算机需要满足以下条件：
            - **1. 拥有支持OpenCL的NVIDIA或AMD显卡。**
            - **2. 已安装最新的显卡驱动程序。** 对于大多数用户来说，最新的驱动已包含所需的OpenCL运行库。
            - **3. (macOS用户)**: 您需要通过Homebrew安装OpenMP库: `brew install libomp`。
          - **如何使用**:
            - 只需在"超参数调节"区域勾选"尝试使用GPU加速"即可。
            - 如果您的环境配置正确，`lightgbm`等模型在训练时会自动利用GPU，速度将大幅提升。
            - 如果环境未配置或配置错误，训练过程可能会失败并显示相关错误日志，此时请取消勾选GPU选项，或参照[LightGBM官方GPU教程](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)进行排查。
        **- 注意事项:**
          - **内存警告**: Qlib在处理数据时会将所选时间段的全部数据加载到内存。如果您的时间范围过长、股票池过大，可能会导致内存不足。这是正常现象，请通过缩短时间范围或更换机器来解决。
        """)

    if "training_status" not in st.session_state:
        st.session_state.training_status = None
    if "training_log" not in st.session_state:
        st.session_state.training_log = ""

    qlib_dir = st.session_state.settings.get("qlib_data_path", str(Path.home() / ".qlib" / "qlib_data" / "cn_data"))
    models_save_dir = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    st.info(f"当前Qlib数据路径: `{qlib_dir}`")
    st.info(f"当前模型存读路径: `{models_save_dir}` (可在左侧边栏修改)")

    st.subheader("1. 训练模式与模型配置")
    train_mode = st.radio("选择训练模式", ["从零开始新训练", "在旧模型上继续训练 (Finetune)"], key="train_mode", horizontal=True, on_change=lambda: setattr(st.session_state, 'training_status', None))
    finetune_model_path = None
    if train_mode == "在旧模型上继续训练 (Finetune)":
        finetune_dir_path = Path(models_save_dir).expanduser()
        available_finetune_models = [f.name for f in finetune_dir_path.glob("*.pkl")] if finetune_dir_path.exists() else []
        if available_finetune_models:
            selected_finetune_model = st.selectbox("选择一个要继续训练的模型", available_finetune_models)
            finetune_model_path = str(finetune_dir_path / selected_finetune_model)
        else:
            st.warning(f"在 '{finetune_dir_path}' 中未找到任何 .pkl 模型文件。")
            return

    col1, col2 = st.columns(2)
    model_name = col1.selectbox("选择模型", list(MODELS.keys()))
    factor_name = col2.selectbox("选择因子", list(FACTORS.keys()))

    summary = get_data_summary(qlib_dir)
    instrument_list = summary.get("instruments")
    if instrument_list:
        stock_pool = st.selectbox("选择股票池", options=instrument_list, help="这是从您的数据目录中自动扫描到的股票池列表。")
    else:
        st.warning("未在您的数据目录中扫描到股票池文件。请手动输入股票池名称。")
        stock_pool = st.text_input("输入股票池名称 (例如 csi300)", "csi300")

    custom_model_name = st.text_input("为新模型命名 (可选, 留空则使用默认名)")
    if "ALSTM" in model_name:
        st.warning("️️️**注意：** ALSTM是深度学习模型，训练时间非常长，对电脑性能要求很高。")

    st.subheader("2. 数据段与时间范围")
    with st.expander("设置训练、验证和测试集的时间", expanded=False):
        c1, c2 = st.columns(2)
        train_start = c1.date_input("训练开始", datetime.date(2017, 1, 1))
        train_end = c2.date_input("训练结束", datetime.date(2020, 12, 31))
        c1, c2 = st.columns(2)
        valid_start = c1.date_input("验证开始", datetime.date(2021, 1, 1))
        valid_end = c2.date_input("验证结束", datetime.date(2021, 12, 31))
        c1, c2 = st.columns(2)
        test_start = c1.date_input("测试开始", datetime.date(2022, 1, 1))
        test_end = c2.date_input("测试结束", datetime.date.today() - datetime.timedelta(days=1))


    st.subheader("3. 超参数调节")
    use_gpu = st.checkbox("尝试使用GPU加速 (如果可用)", value=False, help="如果您的LightGBM/XGBoost已正确配置GPU支持，勾选此项可以大幅提速。")

    params = copy.deepcopy(MODELS[model_name]["kwargs"])
    if use_gpu:
        params['device'] = 'gpu'

    with st.expander("调节模型参数", expanded=True):
        if any(m in model_name for m in ["LightGBM", "XGBoost", "CatBoost"]):
            if not use_gpu:
                params['n_jobs'] = st.number_input("并行计算线程数 (n_jobs)", -1, 16, -1, help="设置用于并行计算的线程数。-1 表示使用所有可用的CPU核心。")
            if "CatBoost" in model_name:
                params['iterations'] = st.slider("迭代次数", 50, 500, params.get('iterations', 200), 10, key=f"it_{model_name}")
                params['depth'] = st.slider("最大深度", 3, 15, params.get('depth', 7), key=f"depth_{model_name}")
            else:
                params['n_estimators'] = st.slider("树的数量", 50, 500, params.get('n_estimators', 200), 10, key=f"n_est_{model_name}")
                params['max_depth'] = st.slider("最大深度", 3, 15, params.get('max_depth', 7), key=f"depth_{model_name}")
            params['learning_rate'] = st.slider("学习率", 0.01, 0.2, params.get('learning_rate', 0.05), 0.01, key=f"lr_{model_name}")
        elif "ALSTM" in model_name:
            st.info("ALSTM模型的超参数调节暂未在此界面支持。")

    st.subheader("4. 开始训练与日志")
    with st.container(height=400):
        log_placeholder = st.empty()
        if st.session_state.training_log:
            log_placeholder.code(st.session_state.training_log, language='log')
        else:
            log_placeholder.code("训练日志将显示在此处", language='log')

    if st.button("开始训练", key="btn_train"):
        st.session_state.training_status = None
        st.session_state.training_log = ""
        log_placeholder.empty()
        with st.spinner("正在训练模型，此过程可能需要较长时间，请耐心等待..."):
            try:
                if not (train_start < train_end < valid_start < valid_end < test_start < test_end):
                    st.error("日期区间设置错误：必须遵循 训练 < 验证 < 测试 的顺序，且开始日期不能晚于结束日期。")
                    raise ValueError("日期顺序不正确。")
                segments = {
                    "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                    "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
                    "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))
                }
                saved_path, training_log = train_model(
                    qlib_dir=qlib_dir, models_save_dir=models_save_dir, model_name=model_name,
                    factor_name=factor_name, stock_pool=stock_pool, segments=segments,
                    model_params=params, custom_model_name=custom_model_name if custom_model_name else None,
                    finetune_model_path=finetune_model_path, log_placeholder=log_placeholder
                )
                st.session_state.training_status = {"status": "success", "message": f"模型训练成功！已保存至: {saved_path}"}
                st.session_state.training_log = training_log
            except Exception as e:
                st.session_state.training_status = {"status": "error", "message": f"训练过程中发生错误: {e}"}

    if st.session_state.training_status:
        if st.session_state.training_status["status"] == "success":
            st.success(st.session_state.training_status["message"])
            st.balloons()
        elif st.session_state.training_status["status"] == "error":
            st.error(st.session_state.training_status["message"])

def prediction_page():
    st.header("投资组合预测")
    with st.expander("💡 操作指南 (Operation Guide)"):
        st.markdown("""
        **本页面利用已训练好的模型进行预测，帮助您分析和比较模型的预测结果。**
        (Content omitted for brevity)
        """)
    # Full page content omitted for brevity

def backtesting_page():
    st.header("策略回测")
    with st.expander("💡 操作指南 (Operation Guide)"):
        st.markdown("""
        **本页面基于您训练好的模型，运行一个具体、透明的交易策略，以评估模型的实战表现。**
        **- 智能日期填充:**
          - **自动填充**: 当您选择一个模型后，下方的“回测开始/结束日期”会自动填充为该模型在训练时所用的`测试集`时间范围。
          - **手动修改**: 您可以接受这个默认的、推荐的测试范围，也可以手动修改日期来进行更自由的探索。
        """)

    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None

    models_dir = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件进行回测", available_models, key="bt_model_select")
    selected_model_path = str(models_dir_path / selected_model_name)

    try:
        start_date_val, end_date_val = get_model_test_period(selected_model_path)
        start_date_val = datetime.datetime.strptime(start_date_val, "%Y-%m-%d").date()
        end_date_val = datetime.datetime.strptime(end_date_val, "%Y-%m-%d").date()
        st.success(f"已自动从模型配置加载测试期: **{start_date_val}** to **{end_date_val}**。您可以手动修改。")
    except Exception as e:
        st.warning(f"无法从模型配置中自动加载测试期: {e}。请手动设置日期。")
        start_date_val, end_date_val = (datetime.date.today() - datetime.timedelta(days=365), datetime.date.today() - datetime.timedelta(days=1))

    st.subheader("回测参数配置")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("回测开始日期", start_date_val)
    end_date = col2.date_input("回测结束日期", end_date_val)

    st.subheader("策略参数 (Top-K Dropout)")
    c1, c2 = st.columns(2)
    topk = c1.number_input("买入Top-K只股票", 1, 100, 50)
    n_drop = c2.number_input("持有期(天)", 1, 20, 5)

    st.subheader("交易参数")
    c1, c2, c3 = st.columns(3)
    open_cost = c1.number_input("开仓手续费率", 0.0, 0.01, 0.0005, format="%.4f")
    close_cost = c2.number_input("平仓手续费率", 0.0, 0.01, 0.0015, format="%.4f")
    min_cost = c3.number_input("最低手续费", 0, 10, 5)

    if st.button("开始回测", key="btn_bt"):
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期！")
        else:
            strategy_kwargs = {"topk": topk, "n_drop": n_drop}
            exchange_kwargs = {"open_cost": open_cost, "close_cost": close_cost, "min_cost": min_cost, "deal_price": "close"}
            with st.spinner("正在回测..."):
                try:
                    daily_report_df, analysis_df = backtest_strategy(
                        selected_model_path, st.session_state.settings.get("qlib_data_path"), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                        strategy_kwargs, exchange_kwargs
                    )
                    fig = px.line(daily_report_df, x=daily_report_df.index, y=['account', 'bench'], title="策略 vs. 基准")
                    st.session_state.backtest_results = {"daily": daily_report_df, "analysis": analysis_df, "fig": fig}
                except Exception as e:
                    st.error(f"回测过程中发生错误: {e}")
                    st.session_state.backtest_results = None

    if st.session_state.backtest_results:
        st.success("回测完成！")
        analysis_df = st.session_state.backtest_results["analysis"]
        metrics = analysis_df.loc["excess_return_with_cost"]
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("年化收益率", f"{metrics['annualized_return']:.2%}")
        kpi_cols[1].metric("信息比率", f"{metrics['information_ratio']:.2f}")
        kpi_cols[2].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
        kpi_cols[3].metric("平均换手率", f"{st.session_state.backtest_results['daily']['turnover'].mean():.3f}")
        st.subheader("资金曲线")
        st.plotly_chart(st.session_state.backtest_results["fig"], use_container_width=True)
        with st.expander("查看详细分析报告"):
            st.dataframe(analysis_df)

def model_evaluation_page():
    st.header("模型评估")
    with st.expander("💡 操作指南 (Operation Guide)"):
        st.markdown("""
        **本页面对单个模型进行一次全面、标准化的体检，是评判模型好坏的关键。**
        **- 智能日期填充:**
          - **自动执行**: 当您选择一个模型后，本页面会自动使用该模型在训练时所用的`测试集`时间范围进行评估，无需手动设置日期。
        """)

    if "eval_results" not in st.session_state: st.session_state.eval_results = None
    if "evaluation_log" not in st.session_state: st.session_state.evaluation_log = ""

    models_dir = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件进行评估", available_models, key="eval_model_select")

    try:
        start_date_val, end_date_val = get_model_test_period(str(models_dir_path / selected_model_name))
        st.success(f"将自动使用模型配置中的测试期: **{start_date_val}** to **{end_date_val}**")
    except Exception as e:
        st.error(f"无法从模型配置中加载测试期: {e}")
        return

    st.subheader("评估日志")
    with st.container(height=400):
        log_placeholder = st.empty()
        log_placeholder.code("评估日志将显示在此处", language='log')

    if st.button("开始评估", key="btn_eval"):
        st.session_state.evaluation_log = ""
        log_placeholder.empty()
        with st.spinner("正在执行评估，这可能需要几分钟时间..."):
            try:
                model_path = str(models_dir_path / selected_model_name)
                results, eval_log = evaluate_model(model_path, st.session_state.settings.get("qlib_data_path"), log_placeholder=log_placeholder)
                st.session_state.eval_results = results
                st.session_state.evaluation_log = eval_log
                log_placeholder.code(eval_log, language='log')
            except Exception as e:
                st.error(f"评估过程中发生错误: {e}")
                st.session_state.eval_results = None

    if st.session_state.eval_results:
        st.success("模型评估完成！")
        results = st.session_state.eval_results
        signal_figs = results.get("signal_figures", [])
        portfolio_figs = results.get("portfolio_figures", [])
        risk_analysis_table = results.get("risk_analysis_table")
        raw_report_df = results.get("raw_report_df")
        tab1, tab2 = st.tabs(["图表分析 (Visualizations)", "详细数据 (Data Tables)"])
        with tab1:
            st.subheader("投资组合分析 (Portfolio Analysis)")
            for fig in portfolio_figs: st.plotly_chart(fig, use_container_width=True)
            st.subheader("信号分析 (Signal Analysis)")
            for fig in signal_figs: st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("投资组合分析报告")
            if risk_analysis_table is not None:
                st.dataframe(risk_analysis_table)
            if raw_report_df is not None:
                with st.expander("查看每日收益和换手率的原始数据"):
                    st.dataframe(raw_report_df)

def position_analysis_page():
    st.header("策略仓位分析")
    with st.expander("💡 操作指南 (Operation Guide)"):
        st.markdown("""
        **本页面旨在提供对策略在回测期间每日持仓的深入洞察。**
        **- 智能日期填充:**
          - **自动填充**: 当您选择一个模型后，下方的“分析开始/结束日期”会自动填充为该模型在训练时所用的`测试集`时间范围。
          - **手动修改**: 您可以接受这个默认的、推荐的测试范围，也可以手动修改日期来进行更自由的探索。
        """)

    if "pa_results" not in st.session_state: st.session_state.pa_results = None
    models_dir = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件进行分析", available_models, key="pa_model_select")
    selected_model_path = str(models_dir_path / selected_model_name)

    try:
        start_date_val, end_date_val = get_model_test_period(selected_model_path)
        start_date_val = datetime.datetime.strptime(start_date_val, "%Y-%m-%d").date()
        end_date_val = datetime.datetime.strptime(end_date_val, "%Y-%m-%d").date()
        st.success(f"已自动从模型配置加载测试期: **{start_date_val}** to **{end_date_val}**。您可以手动修改。")
    except Exception as e:
        st.warning(f"无法从模型配置中自动加载测试期: {e}。请手动设置日期。")
        start_date_val, end_date_val = (datetime.date.today() - datetime.timedelta(days=365), datetime.date.today() - datetime.timedelta(days=1))

    st.subheader("分析参数配置")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("分析开始日期", start_date_val, key="pa_start")
    end_date = col2.date_input("分析结束日期", end_date_val, key="pa_end")

    st.subheader("策略参数 (Top-K Dropout)")
    c1, c2 = st.columns(2)
    topk = c1.number_input("买入Top-K只股票", 1, 100, 30, key="pa_topk")
    n_drop = c2.number_input("持有期(天)", 1, 20, 5, key="pa_ndrop")

    exchange_kwargs = {"open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5, "deal_price": "close"}

    if st.button("开始分析", key="btn_pa_run"):
        with st.spinner("正在运行回测以生成仓位数据..."):
            try:
                strategy_kwargs = {"topk": topk, "n_drop": n_drop}
                results = get_position_analysis(
                    selected_model_path, st.session_state.settings.get("qlib_data_path"),
                    start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                    strategy_kwargs, exchange_kwargs
                )
                st.session_state.pa_results = results
            except Exception as e:
                st.error(f"分析过程中发生错误: {e}")
                st.session_state.pa_results = None

    if st.session_state.pa_results:
        st.success("仓位数据分析完成！")
        positions_df = st.session_state.pa_results.get("positions")
        risk_figures = st.session_state.pa_results.get("risk_figures", [])
        analysis_df = st.session_state.pa_results.get("analysis_df")

        st.subheader("整体策略表现")
        for fig in risk_figures: st.plotly_chart(fig, use_container_width=True)
        with st.expander("查看详细风险指标"):
            st.dataframe(analysis_df)

        st.subheader("每日持仓数据")
        if positions_df is None or positions_df.empty:
            st.warning("未能获取任何持仓数据。")
        else:
            st.dataframe(positions_df)

def main():
    st.set_page_config(layout="wide", page_title="Qlib 可视化工具")
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()
    st.sidebar.image("https://avatars.githubusercontent.com/u/65423353?s=200&v=4", width=100)
    st.sidebar.title("Qlib 可视化面板")
    page_options = ["数据管理", "模型训练", "投资组合预测", "模型评估", "策略回测", "仓位分析"]
    page = st.sidebar.radio("选择功能页面", page_options)
    st.sidebar.title("路径设置")
    st.sidebar.info("在这里修改的路径会在所有页面生效。点击下方按钮以保存。")
    def update_setting(key):
        st.session_state.settings[key] = st.session_state[key]
    default_qlib_data_path = st.session_state.settings.get("qlib_data_path", str(Path.home() / ".qlib" / "qlib_data"))
    default_models_path = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    st.sidebar.text_input("Qlib 数据存储根路径", value=default_qlib_data_path, key="qlib_data_path", on_change=update_setting, args=("qlib_data_path",))
    st.sidebar.text_input("模型保存/加载根路径", value=default_models_path, key="models_path", on_change=update_setting, args=("models_path",))
    if st.sidebar.button("保存当前路径设置"):
        save_settings(st.session_state.settings)
        st.sidebar.success("路径已保存!")
    st.sidebar.markdown("---")
    if page == "数据管理": data_management_page()
    elif page == "模型训练": model_training_page()
    elif page == "投资组合预测": prediction_page()
    elif page == "模型评估": model_evaluation_page()
    elif page == "策略回测": backtesting_page()
    elif page == "仓位分析": position_analysis_page()

if __name__ == "__main__":
    main()
