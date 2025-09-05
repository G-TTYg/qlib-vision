import streamlit as st
import os
# Suppress the GitPython warning
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
import yaml
import qlib
from qlib.constant import REG_CN
from pathlib import Path
from qlib_utils import (
    MODELS, FACTORS, train_model, predict, run_backtest_and_analysis,
    update_daily_data, check_data_health, get_data_summary, get_historical_prediction,
    evaluate_model, load_settings, save_settings, get_model_info
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
                    # Pass the placeholder directly for real-time updates
                    check_data_health(qlib_1d_dir, log_placeholder, n_jobs)
                    st.success("数据健康度检查已完成！详情请查看上方日志。")
                except Exception as e:
                    st.error(f"检查过程中发生错误。详情请查看上方日志。")
                    # The error details are already in the placeholder via the logger

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

    # Dynamically create stock pool selection
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
            # Add n_jobs here for parallel processing
            if not use_gpu: # n_jobs is for CPU parallelism
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
    st.warning("""
    **重要：关于内存使用的说明**

    Qlib在处理数据时，会一次性将所选**时间范围**和**股票池**的全部数据加载到内存中。这是一个设计特性，旨在最大化计算速度。

    因此，如果您遇到 `Unable to allocate...` 或类似的内存不足错误，这是**正常现象**，表明您选择的数据量超过了您计算机的可用RAM。

    **解决方案**:
    - **缩短时间范围**: 这是最有效的解决方法。请尝试将训练、验证和测试集的总时间跨度减小。
    - **切换为小盘股**: `csi500`比`csi300`需要更多的内存。
    - **硬件升级**: 如果需要处理大规模数据，请在具有更大内存（RAM）的机器上运行。
    """)

    with st.container(height=400):
        log_placeholder = st.empty()
        if st.session_state.training_log:
            log_placeholder.code(st.session_state.training_log, language='log')
        else:
            log_placeholder.code("训练日志将显示在此处", language='log')

    if st.button("开始训练", key="btn_train"):
        st.session_state.training_status = None # Reset status on new run
        st.session_state.training_log = "" # Clear log from session state
        log_placeholder.empty() # Clear previous logs from the placeholder

        with st.spinner("正在训练模型，此过程可能需要较长时间，请耐心等待..."):
            try:
                # --- Config modification for time ranges ---
                all_dates = [train_start, train_end, valid_start, valid_end, test_start, test_end]
                if any(d is None for d in all_dates):
                    raise ValueError("所有日期都必须设置。")
                if not (train_start < train_end < valid_start < valid_end < test_start < test_end):
                    st.error("日期区间设置错误：必须遵循 训练 < 验证 < 测试 的顺序，且开始日期不能晚于结束日期。")
                    raise ValueError("日期顺序不正确。")

                # Build segments and params for new train_model signature
                train_start_str, train_end_str = train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")
                valid_start_str, valid_end_str = valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")
                test_start_str, test_end_str = test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")

                segments = {
                    "train": (train_start_str, train_end_str),
                    "valid": (valid_start_str, valid_end_str),
                    "test": (test_start_str, test_end_str)
                }

                model_params = params # `params` is already updated by the sliders

                saved_path, training_log = train_model(
                    qlib_dir=qlib_dir,
                    models_save_dir=models_save_dir,
                    model_name=model_name,
                    factor_name=factor_name,
                    stock_pool=stock_pool,
                    segments=segments,
                    model_params=model_params,
                    custom_model_name=custom_model_name if custom_model_name else None,
                    finetune_model_path=finetune_model_path,
                    log_placeholder=log_placeholder
                )
                st.session_state.training_status = {"status": "success", "message": f"模型训练成功！已保存至: {saved_path}"}
                st.session_state.training_log = training_log # Save for persistence if needed
            except Exception as e:
                st.session_state.training_status = {"status": "error", "message": f"训练过程中发生错误: {e}"}
                # The log placeholder already contains the error details from the redirected stderr

    if st.session_state.training_status:
        status = st.session_state.training_status
        if status["status"] == "success":
            st.success(status["message"])
            st.balloons()
        elif status["status"] == "error":
            st.error(status["message"])

def prediction_page():
    st.header("投资组合预测")
    with st.expander("💡 操作指南 (Operation Guide)"):
        st.markdown("""
        **本页面利用已训练好的模型进行预测，帮助您分析和比较模型的预测结果。**

        **- 核心作用:**
          - **横向对比**: 在同一天，用多个不同的模型对全市场或特定股票池的股票进行打分，直观地比较哪个模型表现更好。
          - **纵向分析**: 追踪单个模型对某一只特定股票在一段时间内的评分变化，以判断模型对该股票的看法是否稳定、是否存在趋势。

        **- 功能解释:**
          - **1. 多模型对比预测 (单日)**:
            - **用途**: 用于模型“选美”。例如，您用不同参数训练了三个LightGBM模型，您想知道在`2023-01-05`这一天，哪个模型选出的股票表现最好。
            - **操作**: 选择一个或多个您想要对比的模型，选择一个预测日期，然后点击“执行对比预测”。
            - **结果**: 会生成一个包含所有模型打分的数据表，并绘制一张条形图，展示综合评分最高的10只股票以及每个模型对它们的具体打分。
          - **2. 单一股票历史分数追踪**:
            - **用途**: 用于深度分析单个模型对某只股票的“偏见”或“看法”。例如，您想知道您训练的模型是否长期看好贵州茅台（SH600519）。
            - **操作**: 选择一个模型，输入您关心的股票代码（如`SH600519`），选择一个历史时间段，然后点击“开始追踪”。
            - **结果**: 会生成一张折线图，展示在该时间段内，模型每天对这只股票的评分。如果分数持续走高，说明模型近期看好该股票。

        **- 注意事项:**
          - 历史分数追踪功能需要对时间范围内的每一天都进行一次预测，因此如果时间跨度太长，可能会比较耗时。
        """)

    # Initialize session state
    if "pred_results" not in st.session_state:
        st.session_state.pred_results = None
    if "hist_results" not in st.session_state:
        st.session_state.hist_results = None

    qlib_dir = st.session_state.settings.get("qlib_data_path", str(Path.home() / ".qlib" / "qlib_data" / "cn_data"))
    models_dir = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    st.info(f"当前Qlib数据路径: `{qlib_dir}`")
    st.info(f"当前模型加载路径: `{models_dir}` (可在左侧边栏修改)")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    st.subheader("1. 多模型对比预测 (单日)")
    selected_models = st.multiselect("选择一个或多个模型进行对比预测", available_models)

    if selected_models:
        with st.container(border=True):
            st.markdown("**已选模型信息:**")
            for model_name in selected_models:
                model_path = str(models_dir_path / model_name)
                info = get_model_info(model_path)
                stock_pool = info.get('stock_pool', '未知')
                if info.get("error"):
                    st.warning(f"- **{model_name}**: 无法加载信息 ({info['error']})")
                else:
                    st.markdown(f"- **{model_name}**: 预测股票池 `{stock_pool}`")

    prediction_date = st.date_input("选择预测日期", datetime.date.today() - datetime.timedelta(days=1))

    progress_placeholder = st.empty()

    if st.button("执行对比预测", key="btn_pred") and selected_models:
        with st.spinner("正在执行预测..."):
            try:
                all_preds = []
                for i, model_name in enumerate(selected_models):
                    progress_placeholder.text(f"正在预测第 {i+1}/{len(selected_models)} 个模型: {model_name}...")
                    model_path = str(models_dir_path / model_name)
                    pred_df = predict(model_path, qlib_dir, prediction_date.strftime("%Y-%m-%d"))
                    pred_df = pred_df.rename(columns={"score": f"score_{model_name.replace('.pkl', '')}"})
                    all_preds.append(pred_df.set_index('StockID')[f"score_{model_name.replace('.pkl', '')}"])
                combined_df = pd.concat(all_preds, axis=1).reset_index()

                if combined_df.empty:
                    st.session_state.pred_results = {"status": "empty"}
                else:
                    score_cols = [col for col in combined_df.columns if 'score' in col]
                    combined_df['average_score'] = combined_df[score_cols].mean(axis=1)
                    top_10_stocks = combined_df.nlargest(10, 'average_score')
                    plot_df = top_10_stocks.melt(id_vars=['StockID'], value_vars=score_cols, var_name='Model', value_name='Score')
                    plot_df['Model'] = plot_df['Model'].str.replace('score_', '')
                    fig = px.bar(plot_df, x="StockID", y="Score", color="Model", barmode='group', title="Top-10 股票多模型分数对比")
                    st.session_state.pred_results = {"df": combined_df, "fig": fig, "status": "ok"}

            except Exception as e:
                st.error(f"预测过程中发生错误: {e}")
                st.session_state.pred_results = None

    if st.session_state.pred_results:
        if st.session_state.pred_results.get("status") == "ok":
            st.success("预测完成！")
            st.dataframe(st.session_state.pred_results["df"])
            st.plotly_chart(st.session_state.pred_results["fig"], use_container_width=True)
        elif st.session_state.pred_results.get("status") == "empty":
            st.warning("预测结果为空。可能原因：您选择的日期是非交易日、数据缺失、或所选股票池当天全部停牌。")

    st.subheader("2. 单一股票历史分数追踪")
    col1, col2 = st.columns(2)
    single_model_name = col1.selectbox("选择用于追踪的模型", available_models, key="single_model_select")

    if single_model_name:
        model_path = str(models_dir_path / single_model_name)
        info = get_model_info(model_path)
        stock_pool = info.get('stock_pool', '未知')
        st.info(f"已选模型 **{single_model_name}** 在股票池 `{stock_pool}` 上进行训练。")

    stock_id_input = col2.text_input("输入股票代码 (例如 SH600519)", "SH600519")
    col3, col4 = st.columns(2)
    hist_start_date = col3.date_input("追踪开始日期", datetime.date.today() - datetime.timedelta(days=90))
    hist_end_date = col4.date_input("追踪结束日期", datetime.date.today() - datetime.timedelta(days=1))

    hist_progress_placeholder = st.empty()

    if st.button("开始追踪", key="btn_hist"):
        if not single_model_name or not stock_id_input:
            st.warning("请选择一个模型并输入股票代码。")
            st.session_state.hist_results = None
        else:
            with st.spinner(f"正在为股票 {stock_id_input} 获取历史分数..."):
                try:
                    single_model_path = str(models_dir_path / single_model_name)
                    hist_df = get_historical_prediction(single_model_path, qlib_dir, stock_id_input.upper(), str(hist_start_date), str(hist_end_date), placeholder=hist_progress_placeholder)
                    if hist_df.empty:
                        st.session_state.hist_results = {"status": "empty"}
                    else:
                        fig = px.line(hist_df, x="Date", y="Score", title=f"模型 {single_model_name} 对 {stock_id_input} 的历史评分")
                        st.session_state.hist_results = {"df": hist_df, "fig": fig, "status": "ok"}
                except Exception as e:
                    st.error(f"历史分数追踪过程中发生错误: {e}")
                    st.session_state.hist_results = None

    if st.session_state.hist_results:
        if st.session_state.hist_results["status"] == "ok":
            st.success("历史分数追踪完成！")
            st.plotly_chart(st.session_state.hist_results["fig"], use_container_width=True)
        elif st.session_state.hist_results["status"] == "empty":
            st.warning("在指定时间段内未能获取到该股票的有效预测分数。")

def backtesting_and_analysis_page():
    st.header("策略回测与分析")
    with st.expander("💡 操作指南 (Operation Guide)", expanded=True):
        st.markdown("""
        **本页面将模型预测与实际交易策略相结合，提供一个从策略表现到每日持仓的全面、深入的分析报告。**

        #### 一、核心功能
        - **实战模拟**: 将模型的预测分数转化为透明的买卖操作，并在历史数据上进行模拟交易，以检验模型的真实盈利能力和风险。
        - **风险洞察**: 不仅展示收益，更深入分析策略的风险来源，如持股集中度、交易频率等。
        - **持仓追溯**: 可以查看回测期间任何一天的详细持仓记录，让策略的每一个决策都清晰可见。

        #### 二、参数解释
        - **回测参数**:
          - `开始/结束日期`: 定义了进行模拟交易的历史时间段。
        - **策略参数: Top-K Dropout**
          - 这是一个经典的选股策略。在每个交易日，买入模型评分最高的 `K` 只股票，持有 `N` 天后卖出。
          - `买入Top-K只股票`: 每天买入的股票数量。K值越小，策略越集中，潜在风险和收益都可能更高。
          - `持有期(天)`: 每只股票的持有天数。持有期越短，交易越频繁，换手率和交易成本也越高。
        - **交易参数**:
          - `手续费率/最低收费`: 用于模拟真实交易成本，这会直接影响最终的净收益。

        #### 三、报告解读
        报告现在由三个核心部分组成，提供了一个完整、多维度的策略评估：

        1.  **综合回测报告 (Overall Report)**
            - `(由 report_graph 生成)`
            - 这是最核心的回测图表，展示了策略的**累计收益**、**与基准的超额收益**、**每日换手率**以及重要的**回撤区间**。通过这张图，您可以对策略的整体表现有一个宏观的认识。

        2.  **IC 分析 (IC Analysis)**
            - `(由 score_ic_graph 生成)`
            - 这部分图表评估的是模型**预测能力**的本身，即模型的预测分数（Score）与未来真实收益（Label）的相关性。
            - `IC` 和 `Rank IC` 的时间序列图和均值，是判断模型好坏的核心依据。一个好的模型，其IC值应该持续、稳定地大于0。

        3.  **风险分析 (Risk Analysis)**
            - `(由 risk_analysis_graph 生成)`
            - 这部分图表将策略的各项风险指标（如年化收益、信息比率、最大回撤）按**年度和月度**进行了分解。
            - 通过它，您可以观察到策略在不同市场环境下的表现是否稳定，例如，策略在哪一年表现最好，在哪一月回撤最大。

        4.  **每日详细持仓 (Daily Holdings)**
            - 这是一个详细的数据表，记录了回测期间每一天的详细持仓股票、成本、价格、权重等信息，提供了最终极的透明度。
        """)

    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None

    # --- Setup and Model Selection ---
    qlib_dir = st.session_state.settings.get("qlib_data_path", str(Path.home() / ".qlib" / "qlib_data" / "cn_data"))
    models_dir = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    st.info(f"当前Qlib数据路径: `{qlib_dir}`")
    st.info(f"当前模型加载路径: `{models_dir}` (可在左侧边栏修改)")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return
    selected_model_name = st.selectbox("选择一个模型文件进行回测", available_models, key="bt_model_select")

    # --- Date Configuration ---
    start_date_val, end_date_val = datetime.date(2022, 1, 1), datetime.date.today() - datetime.timedelta(days=1)
    if selected_model_name:
        config_path = (models_dir_path / selected_model_name).with_suffix(".yaml")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                test_period = config.get("dataset", {}).get("kwargs", {}).get("segments", {}).get("test")
                if test_period and len(test_period) == 2:
                    start_date_val, end_date_val = pd.to_datetime(test_period[0]).date(), pd.to_datetime(test_period[1]).date()
            except Exception as e:
                st.warning(f"无法加载模型配置文件 {config_path.name} 中的默认日期: {e}")

    selected_model_path = str(models_dir_path / selected_model_name) if selected_model_name else None

    # --- Parameters UI ---
    st.subheader("回测参数配置")
    st.info("默认加载模型训练时使用的测试集时间范围，可手动修改。")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("开始日期", value=start_date_val, key="bt_start")
    end_date = col2.date_input("结束日期", value=end_date_val, key="bt_end")
    benchmark = st.text_input("基准代码 (Benchmark Ticker)", "SH000300", help="用于对比的基准指数代码，例如沪深300是 'SH000300'。")

    st.subheader("策略参数 (Top-K Dropout)")
    c1, c2 = st.columns(2)
    topk = c1.number_input("买入Top-K只股票", 1, 100, 50)
    n_drop = c2.number_input("持有期(天)", 1, 20, 5)

    st.subheader("交易参数")
    c1, c2, c3 = st.columns(3)
    open_cost = c1.number_input("开仓手续费率", 0.0, 0.01, 0.0005, format="%.4f")
    close_cost = c2.number_input("平仓手续费率", 0.0, 0.01, 0.0015, format="%.4f")
    min_cost = c3.number_input("最低手续费", 0, 10, 5)

    # --- Execution Button ---
    if st.button("开始回测与分析", key="btn_bt_run"):
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期！")
            st.session_state.backtest_results = None
        else:
            strategy_kwargs = {"topk": topk, "n_drop": n_drop}
            exchange_kwargs = {"open_cost": open_cost, "close_cost": close_cost, "min_cost": min_cost, "deal_price": "close"}
            with st.spinner("正在执行回测与分析，此过程可能需要一些时间..."):
                try:
                    results = run_backtest_and_analysis(
                        model_path=selected_model_path,
                        qlib_dir=qlib_dir,
                        start_time=start_date.strftime("%Y-%m-%d"),
                        end_time=end_date.strftime("%Y-%m-%d"),
                        strategy_kwargs=strategy_kwargs,
                        exchange_kwargs=exchange_kwargs,
                        benchmark=benchmark
                    )
                    st.session_state.backtest_results = results
                except Exception as e:
                    st.error(f"回测分析过程中发生错误: {e}")
                    st.session_state.backtest_results = None

    # --- Results Display ---
    if st.session_state.backtest_results:
        st.success("回测与分析完成！")
        results = st.session_state.backtest_results

        # Unpack all the results from the backend
        positions_df = results.get("positions_df")
        report_figures = results.get("report_figures", [])
        risk_figures = results.get("risk_figures", [])
        ic_figures = results.get("ic_figures", [])

        # Display section for the main report graphs
        st.subheader("综合回测报告 (Overall Report)")
        if report_figures:
            for fig in report_figures:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("未能生成综合回测报告图。")

        # Display section for IC analysis graphs
        st.subheader("IC 分析 (IC Analysis)")
        if ic_figures:
            for fig in ic_figures:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("未能生成IC分析图。")

        # Display section for risk analysis graphs
        st.subheader("风险分析 (Risk Analysis)")
        if risk_figures:
            for fig in risk_figures:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("未能生成风险分析图。")

        # Display section for detailed daily holdings
        with st.expander("查看每日详细持仓 (Daily Holdings)"):
            if positions_df is not None and not positions_df.empty:
                st.info("下表记录了回测期间每一天的详细持仓情况，包括每只股票的代码、持仓成本、当前价格和持仓权重。")
                # Robustly convert all object-type columns to strings to prevent pyarrow serialization errors.
                for col in positions_df.columns:
                    if positions_df[col].dtype == 'object':
                        positions_df[col] = positions_df[col].astype(str)
                st.dataframe(positions_df)
            else:
                st.info("没有可显示的每日持仓数据。")

def model_evaluation_page():
    st.header("模型评估")
    with st.expander("💡 操作指南 (Operation Guide)", expanded=True):
        st.markdown("""
        **本页面对单个模型进行一次全面、标准化的体检，是评判模型好坏的关键。**

        #### 一、核心作用
        - **综合评估**: 从“预测准确度”和“模拟实战潜力”两个维度，对模型进行深度分析，避免单一指标带来的误判。
        - **标准化流程**: 所有模型都走同一套评估流程，确保了不同模型之间性能的可比性。

        #### 二、报告解读
        评估报告主要包含**信号分析 (Signal Analysis)**，它评估的是模型预测的“分数”（Signal）本身的质量，即预测的有多准，这与任何具体的交易策略无关。

        ##### **关键指标详解:**
        - **`IC (Information Coefficient)` - 信息系数**:
          - **定义**: 计算模型预测的`分数(score)`与`未来真实收益(label)`之间的**皮尔逊相关系数**。
          - **解读**: IC的范围是 `[-1, 1]`。
            - `IC > 0`: 预测方向正确（预测分高的股票未来收益也高）。
            - `IC < 0`: 预测方向相反。
            - `IC 绝对值` 越高，代表预测能力越强。在量化领域，**IC的绝对值通常很低**，一般认为 `|IC| > 0.02` 就有一定预测价值，`|IC| > 0.05` 就算相当不错的模型了。

        - **`Rank IC (Rank Information Coefficient)` - 等级信息系数**:
          - **定义**: 计算`分数(score)`的**排序**与`未来真实收益(label)`的**排序**之间的**斯皮尔曼相关系数**。
          - **解读**: 对于选股模型来说，我们更关心的是**相对排序**（哪些股票比其他股票更好），而不是分数的绝对值。因此，**Rank IC 是比 IC 更重要的核心指标**。它的解读方式与IC类似，`|Rank IC| > 0.03` 通常被认为是有效信号。

        - **`ICIR (Information Coefficient Information Ratio)` - 信息系数比率**:
          - **定义**: `ICIR = mean(IC) / std(IC)`，即 **IC均值** 除以 **IC标准差**。
          - **解读**: ICIR衡量了模型预测能力的**稳定性和强度**。一个模型可能某些天IC很高，某些天很低甚至为负。ICIR越高，说明模型在整个回测期间内能持续、稳定地产生正向的IC。通常认为 `|ICIR| > 0.5` 是一个比较强的信号。

        - **`Rank ICIR (Rank Information Coefficient Information Ratio)` - 等级信息系数比率**:
          - **定义**: `Rank ICIR = mean(Rank IC) / std(Rank IC)`。
          - **解读**: 同理，这是衡量模型**排序能力稳定性**的核心指标。一个高且稳定的Rank ICIR，是优秀选股模型的标志。

        ##### **图表解读:**
        - **IC Series / Rank IC Series**: 展示了在回测期内，每一天的IC/Rank IC的值。可以直观地看出模型表现的波动性。
        - **IC Distribution / Rank IC Distribution**: IC/Rank IC的直方图，展示了IC值的分布情况。
        - **Group Return**: 将股票按模型分数从高到低分为几组（Quantile），计算每组的平均收益率。一个好的模型，其`Group 1`（得分最高组）的收益率应显著高于其他组，且收益率应大致随组号递减。

        #### 三、操作流程
          1. 从下拉框中选择一个您已经训练好的模型。
          2. （可选）修改评估的起止日期，默认为模型训练时设定的测试集周期。
          3. 点击“开始评估”，等待片刻，下方即会生成完整的信号分析报告。
        """)

    # Initialize session state
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    if "evaluation_log" not in st.session_state:
        st.session_state.evaluation_log = ""

    qlib_dir = st.session_state.settings.get("qlib_data_path", str(Path.home() / ".qlib" / "qlib_data" / "cn_data"))
    models_dir = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))
    st.info(f"当前Qlib数据路径: `{qlib_dir}`")
    st.info(f"当前模型加载路径: `{models_dir}` (可在左侧边栏修改)")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件进行评估", available_models, key="eval_model_select")

    # --- Date Override UI ---
    # Set default dates first
    start_date_val = datetime.date(2022, 1, 1)
    end_date_val = datetime.date.today() - datetime.timedelta(days=1)

    # If a model is selected, try to load its config to set better default dates
    if selected_model_name:
        config_path = (models_dir_path / selected_model_name).with_suffix(".yaml")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                # Safely get the test period from the config
                default_test_period = config.get("dataset", {}).get("kwargs", {}).get("segments", {}).get("test")
                if default_test_period and len(default_test_period) == 2:
                    start_date_val = pd.to_datetime(default_test_period[0]).date()
                    end_date_val = pd.to_datetime(default_test_period[1]).date()
            except Exception as e:
                st.warning(f"无法加载模型配置文件 {config_path.name} 中的默认日期: {e}")

    st.subheader("评估周期配置 (可手动修改)")
    st.info("默认加载模型训练时使用的测试集时间范围。")
    col1, col2 = st.columns(2)
    eval_start_date = col1.date_input("开始日期", value=start_date_val, key="eval_start")
    eval_end_date = col2.date_input("结束日期", value=end_date_val, key="eval_end")
    # --- End of Date Override UI ---

    st.subheader("评估日志")
    with st.container(height=400):
        log_placeholder = st.empty()
        log_placeholder.code("评估日志将显示在此处", language='log')


    if st.button("开始评估", key="btn_eval"):
        if not selected_model_name:
            st.warning("请选择一个模型。")
            st.session_state.eval_results = None
        elif eval_start_date >= eval_end_date:
            st.error("开始日期必须早于结束日期！")
            st.session_state.eval_results = None
        else:
            st.session_state.evaluation_log = "" # Clear previous logs
            log_placeholder.empty()
            with st.spinner("正在执行评估，这可能需要几分钟时间..."):
                try:
                    model_path = str(models_dir_path / selected_model_name)
                    # Pass the potentially overridden test period to the backend
                    test_period_override = (eval_start_date.strftime("%Y-%m-%d"), eval_end_date.strftime("%Y-%m-%d"))
                    results, eval_log = evaluate_model(
                        model_path,
                        qlib_dir,
                        log_placeholder=log_placeholder,
                        test_period=test_period_override
                    )
                    st.session_state.eval_results = results
                    st.session_state.evaluation_log = eval_log
                    log_placeholder.code(eval_log, language='log') # Display final log
                except Exception as e:
                    st.error(f"评估过程中发生错误: {e}")
                    st.session_state.eval_results = None

    if st.session_state.eval_results:
        st.success("模型评估完成！")

        results = st.session_state.eval_results
        signal_figs = results.get("signal_figures", [])

        st.header("评估报告：信号分析")
        if not signal_figs:
            st.warning("未能生成任何信号分析图表。请检查日志输出。")
        else:
            st.info("信号分析（Signal Analysis）评估模型预测的“分数”本身的质量，即预测的有多准，与具体买卖的交易策略无关。")
            for fig in signal_figs:
                st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide", page_title="Qlib 可视化工具")

    # --- Settings Initialization ---
    if 'settings' not in st.session_state:
        st.session_state.settings = load_settings()

    st.sidebar.image("https://avatars.githubusercontent.com/u/65423353?s=200&v=4", width=100)
    st.sidebar.title("Qlib 可视化面板")

    # --- Page Selection ---
    page_options = ["数据管理", "模型训练", "投资组合预测", "模型评估", "策略回测与分析"]
    page = st.sidebar.radio("选择功能页面", page_options)

    # --- Settings Persistence ---
    st.sidebar.title("路径设置")
    st.sidebar.info("在这里修改的路径会在所有页面生效。点击下方按钮以保存。")

    # We use a trick here: the text_input's key is the same as the settings key.
    # The on_change callback updates the session_state.settings dict.
    # This makes the code cleaner as we don't need to handle each input individually.
    def update_setting(key):
        st.session_state.settings[key] = st.session_state[key]

    # Get default paths
    default_qlib_data_path = st.session_state.settings.get("qlib_data_path", str(Path.home() / ".qlib" / "qlib_data"))
    default_models_path = st.session_state.settings.get("models_path", str(Path.home() / "qlib_models"))

    st.sidebar.text_input("Qlib 数据存储根路径", value=default_qlib_data_path, key="qlib_data_path", on_change=update_setting, args=("qlib_data_path",))
    st.sidebar.text_input("模型保存/加载根路径", value=default_models_path, key="models_path", on_change=update_setting, args=("models_path",))

    if st.sidebar.button("保存当前路径设置"):
        save_settings(st.session_state.settings)
        st.sidebar.success("路径已保存!")

    st.sidebar.markdown("---") # Separator

    # --- Page Display ---
    if page == "数据管理": data_management_page()
    elif page == "模型训练": model_training_page()
    elif page == "投资组合预测": prediction_page()
    elif page == "模型评估": model_evaluation_page()
    elif page == "策略回测与分析": backtesting_and_analysis_page()

if __name__ == "__main__":
    main()
