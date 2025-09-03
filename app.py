import streamlit as st
import qlib
from qlib.constant import REG_CN
from pathlib import Path
from qlib_utils import (
    SUPPORTED_MODELS, train_model, predict, backtest_strategy,
    update_daily_data, check_data_health, get_historical_prediction,
    evaluate_model
)
import pandas as pd
import plotly.express as px
import datetime
import copy

# --- Streamlit Pages ---

def data_management_page():
    st.header("数据管理")
    st.markdown("""
    本页面提供Qlib所需数据的管理功能。请遵循以下步骤：
    - **首次使用者**: 请先按照“全量数据部署”中的指引，通过命令行手动下载并解压数据。这是最稳定、最推荐的初始化方式。
    - **日常使用者**: 如果您已经部署了全量数据，可以使用“增量更新”功能来获取最新数据。
    - **数据检查**: 您可以使用“健康度检查”来验证本地数据的完整性。
    """)

    # Initialize session state for logs
    if "data_log" not in st.session_state:
        st.session_state.data_log = ""

    default_path = str(Path.home() / ".qlib" / "qlib_data")
    qlib_dir = st.text_input("Qlib 数据存储根路径", default_path, key="data_dir_dm")
    qlib_1d_dir = str(Path(qlib_dir) / "cn_data")

    with st.expander("1. 全量数据部署 (首次使用)", expanded=True):
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

    log_placeholder = st.empty()
    log_placeholder.code(st.session_state.data_log, language='log')

    col1, col2 = st.columns(2)
    start_date = col1.date_input("更新开始日期", datetime.date.today() - datetime.timedelta(days=7))
    end_date = col2.date_input("更新结束日期", datetime.date.today())

    c1, c2, c3 = st.columns([1, 1, 5])
    if c1.button("开始增量更新"):
        st.session_state.data_log = "" # Clear previous logs
        log_placeholder.code(st.session_state.data_log, language='log') # Clear display
        with st.spinner(f"正在更新从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据..."):
            try:
                update_daily_data(qlib_1d_dir, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), "data_log")
                st.success("增量更新命令已成功执行！")
            except Exception as e:
                st.error(f"增量更新过程中发生错误: {e}")
        log_placeholder.code(st.session_state.data_log, language='log') # Update display with final logs

    if c2.button("开始检查数据"):
        st.session_state.data_log = "" # Clear previous logs
        log_placeholder.code(st.session_state.data_log, language='log') # Clear display
        with st.spinner("正在检查数据..."):
            try:
                check_data_health(qlib_1d_dir, "data_log")
                st.success("数据健康度检查已完成！详情请查看上方日志。")
            except Exception as e:
                st.error(f"检查过程中发生错误: {e}")
        log_placeholder.code(st.session_state.data_log, language='log') # Update display with final logs

def model_training_page():
    st.header("模型训练")
    st.markdown("""
    在这里，您可以训练自己的量化模型。
    - **模型与因子**: Qlib提供了多种内置模型（如LightGBM, XGBoost等）和因子（如Alpha158, Alpha360）。
    - **股票池**: 您可以选择在不同的股票池（如沪深300, 中证500）上进行训练。
    - **训练模式**: 您可以从零开始训练一个全新的模型，或者在已有的模型基础上进行增量训练（Finetune）。
    - **超参数**: 对于GBDT类的模型，您可以方便地调节树的数量、深度、学习率等关键超参数。
    """)

    if "training_status" not in st.session_state:
        st.session_state.training_status = None

    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_train")
    models_save_dir = st.text_input("训练后模型的保存路径", default_models_path)
    st.subheader("1. 训练模式与模型配置")
    train_mode = st.radio("选择训练模式", ["从零开始新训练", "在旧模型上继续训练 (Finetune)"], key="train_mode", horizontal=True, on_change=lambda: setattr(st.session_state, 'training_status', None))
    finetune_model_path = None
    if train_mode == "在旧模型上继续训练 (Finetune)":
        finetune_model_dir = st.text_input("要加载的旧模型所在目录", default_models_path, key="finetune_dir")
        finetune_dir_path = Path(finetune_model_dir).expanduser()
        available_finetune_models = [f.name for f in finetune_dir_path.glob("*.pkl")] if finetune_dir_path.exists() else []
        if available_finetune_models:
            selected_finetune_model = st.selectbox("选择一个要继续训练的模型", available_finetune_models)
            finetune_model_path = str(finetune_dir_path / selected_finetune_model)
        else:
            st.warning(f"在 '{finetune_dir_path}' 中未找到任何 .pkl 模型文件。")
            return
    col1, col2 = st.columns(2)
    model_name_key = col1.selectbox("选择模型和因子", list(SUPPORTED_MODELS.keys()))
    stock_pool = col2.selectbox("选择股票池", ["csi300", "csi500"], index=0)
    custom_model_name = st.text_input("为新模型命名 (可选, 留空则使用默认名)")
    if "ALSTM" in model_name_key:
        st.warning("️️️**注意：** ALSTM是深度学习模型，训练时间非常长，对电脑性能要求很高。")
    st.subheader("2. 超参数调节")
    config = copy.deepcopy(SUPPORTED_MODELS[model_name_key])
    params = config['task']['model']['kwargs']
    with st.expander("调节模型参数", expanded=True):
        if any(m in model_name_key for m in ["LightGBM", "XGBoost", "CatBoost"]):
            if "CatBoost" in model_name_key:
                params['iterations'] = st.slider("迭代次数", 50, 500, params.get('iterations', 200), 10, key=f"it_{model_name_key}")
                params['depth'] = st.slider("最大深度", 3, 15, params.get('depth', 7), key=f"depth_{model_name_key}")
            else:
                params['n_estimators'] = st.slider("树的数量", 50, 500, params.get('n_estimators', 200), 10, key=f"n_est_{model_name_key}")
                params['max_depth'] = st.slider("最大深度", 3, 15, params.get('max_depth', 7), key=f"depth_{model_name_key}")
            params['learning_rate'] = st.slider("学习率", 0.01, 0.2, params.get('learning_rate', 0.05), 0.01, key=f"lr_{model_name_key}")
        elif "ALSTM" in model_name_key:
            st.info("ALSTM模型的超参数调节暂未在此界面支持。")

    if st.button("开始训练", key="btn_train"):
        st.session_state.training_status = None # Reset status on new run
        with st.spinner("正在训练模型，此过程可能需要较长时间，请耐心等待..."):
            try:
                saved_path = train_model(model_name_key, qlib_dir, models_save_dir, config, custom_model_name if custom_model_name else None, stock_pool, finetune_model_path)
                st.session_state.training_status = {"status": "success", "message": f"模型训练成功！已保存至: {saved_path}"}
            except Exception as e:
                st.session_state.training_status = {"status": "error", "message": f"训练过程中发生错误: {e}"}

    if st.session_state.training_status:
        status = st.session_state.training_status
        if status["status"] == "success":
            st.success(status["message"])
            st.balloons()
        elif status["status"] == "error":
            st.error(status["message"])

def prediction_page():
    st.header("投资组合预测")
    st.markdown("""
    本页面提供两种预测模式：
    - **多模型对比预测**: 选择多个模型，对单一日期的所有股票进行打分。您可以查看分数的数据表，以及Top-10股票的分数对比图。这有助于横向比较不同模型在同一时间点上的优劣。
    - **单一股票历史分数追踪**: 选择一个模型和一只股票，查看该模型在过去一段时间内对这只股票的评分变化。这有助于分析模型对特定股票的判断是否具有时间上的一致性。
    """)

    # Initialize session state
    if "pred_results" not in st.session_state:
        st.session_state.pred_results = None
    if "hist_results" not in st.session_state:
        st.session_state.hist_results = None

    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_pred")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_pred")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    st.subheader("1. 多模型对比预测 (单日)")
    selected_models = st.multiselect("选择一个或多个模型进行对比预测", available_models)
    prediction_date = st.date_input("选择预测日期", datetime.date.today() - datetime.timedelta(days=1))

    if st.button("执行对比预测", key="btn_pred") and selected_models:
        with st.spinner("正在执行预测..."):
            try:
                all_preds = []
                for model_name in selected_models:
                    model_path = str(models_dir_path / model_name)
                    pred_df = predict(model_path, qlib_dir, prediction_date.strftime("%Y-%m-%d"))
                    pred_df = pred_df.rename(columns={"score": f"score_{model_name.replace('.pkl', '')}"})
                    all_preds.append(pred_df.set_index('StockID')[f"score_{model_name.replace('.pkl', '')}"])
                combined_df = pd.concat(all_preds, axis=1).reset_index()
                score_cols = [col for col in combined_df.columns if 'score' in col]
                combined_df['average_score'] = combined_df[score_cols].mean(axis=1)
                top_10_stocks = combined_df.nlargest(10, 'average_score')
                plot_df = top_10_stocks.melt(id_vars=['StockID'], value_vars=score_cols, var_name='Model', value_name='Score')
                plot_df['Model'] = plot_df['Model'].str.replace('score_', '')
                fig = px.bar(plot_df, x="StockID", y="Score", color="Model", barmode='group', title="Top-10 股票多模型分数对比")
                st.session_state.pred_results = {"df": combined_df, "fig": fig}
            except Exception as e:
                st.error(f"预测过程中发生错误: {e}")
                st.session_state.pred_results = None

    if st.session_state.pred_results:
        st.success("预测完成！")
        st.dataframe(st.session_state.pred_results["df"])
        st.plotly_chart(st.session_state.pred_results["fig"], use_container_width=True)

    st.subheader("2. 单一股票历史分数追踪")
    col1, col2 = st.columns(2)
    single_model_name = col1.selectbox("选择用于追踪的模型", available_models, key="single_model_select")
    stock_id_input = col2.text_input("输入股票代码 (例如 SH600519)", "SH600519")
    col3, col4 = st.columns(2)
    hist_start_date = col3.date_input("追踪开始日期", datetime.date.today() - datetime.timedelta(days=90))
    hist_end_date = col4.date_input("追踪结束日期", datetime.date.today() - datetime.timedelta(days=1))

    if st.button("开始追踪", key="btn_hist"):
        if not single_model_name or not stock_id_input:
            st.warning("请选择一个模型并输入股票代码。")
            st.session_state.hist_results = None
        else:
            with st.spinner(f"正在为股票 {stock_id_input} 获取历史分数..."):
                try:
                    single_model_path = str(models_dir_path / single_model_name)
                    hist_df = get_historical_prediction(single_model_path, qlib_dir, stock_id_input.upper(), str(hist_start_date), str(hist_end_date))
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

def backtesting_page():
    st.header("策略回测")
    st.markdown("""
    本页面基于您训练好的模型，进行Top-K选股并持有N天的交易策略，在指定的历史时间段内进行模拟交易，以评估模型的实战表现。
    - **回测参数**: 设置回测的开始和结束日期。
    - **策略参数**: `Top-K`指每日买入模型评分最高的K只股票，`持有期`指每只股票买入后持有N天再卖出。
    - **交易参数**: 您可以设置交易的手续费率和最低费用，以更真实地模拟交易成本。
    最终会生成策略的详细绩效指标（年化收益、夏普比率、最大回撤等）和资金曲线图。
    """)

    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None

    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_bt")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_bt")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return
    selected_model_name = st.selectbox("选择一个模型文件进行回测", available_models)
    selected_model_path = str(models_dir_path / selected_model_name)
    st.subheader("回测参数配置")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("开始日期", datetime.date.today() - datetime.timedelta(days=365))
    end_date = col2.date_input("结束日期", datetime.date.today() - datetime.timedelta(days=1))
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
            st.session_state.backtest_results = None
        else:
            strategy_kwargs = {"topk": topk, "n_drop": n_drop}
            exchange_kwargs = {"open_cost": open_cost, "close_cost": close_cost, "min_cost": min_cost, "deal_price": "close"}
            with st.spinner("正在回测..."):
                try:
                    report_df = backtest_strategy(selected_model_path, qlib_dir, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), strategy_kwargs, exchange_kwargs)
                    fig = px.line(report_df, x=report_df.index, y=report_df.columns, title="策略 vs. 基准")
                    st.session_state.backtest_results = {"report": report_df, "fig": fig}
                except Exception as e:
                    st.error(f"回测过程中发生错误: {e}")
                    st.session_state.backtest_results = None

    if st.session_state.backtest_results:
        st.success("回测完成！")
        st.subheader("绩效指标")
        report_df = st.session_state.backtest_results["report"]
        metrics = report_df.loc["excess_return_with_cost"]
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("年化收益率", f"{metrics['annualized_return']:.2%}")
        kpi_cols[1].metric("夏普比率", f"{metrics['information_ratio']:.2f}")
        kpi_cols[2].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
        kpi_cols[3].metric("换手率", f"{metrics['turnover_rate']:.2f}")
        st.subheader("资金曲线")
        st.plotly_chart(st.session_state.backtest_results["fig"], use_container_width=True)

def model_evaluation_page():
    st.header("模型评估")
    st.markdown("""
    本页面对单个模型进行全面的性能评估，包含两个核心部分：
    - **信号分析 (Signal Analysis)**: 评估模型预测信号（即股票分数）自身的质量，如IC、Rank IC等。这反映了模型的预测能力，与具体交易策略无关。
    - **组合分析 (Portfolio Analysis)**: 基于模型信号，运行一个标准的Top-K策略，并分析该策略的绩效。这反映了模型在模拟实战中的表现，包含年化收益、夏普比率、最大回撤等指标。
    """)

    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None

    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_eval")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_eval")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件进行评估", available_models, key="eval_model_select")

    if st.button("开始评估", key="btn_eval"):
        if not selected_model_name:
            st.warning("请选择一个模型。")
            st.session_state.eval_results = None
        else:
            with st.spinner("正在执行评估，这可能需要几分钟时间..."):
                try:
                    model_path = str(models_dir_path / selected_model_name)
                    results = evaluate_model(model_path, qlib_dir)
                    st.session_state.eval_results = results
                except Exception as e:
                    st.error(f"评估过程中发生错误: {e}")
                    st.session_state.eval_results = None

    if st.session_state.eval_results:
        st.success("模型评估完成！")

        st.subheader("1. 信号分析 (Signal Analysis)")
        signal_report = st.session_state.eval_results["signal"]
        st.dataframe(signal_report)

        st.subheader("2. 组合分析 (Portfolio Analysis)")
        portfolio_report = st.session_state.eval_results["portfolio"]

        st.markdown("**关键绩效指标 (KPIs)**")
        metrics = portfolio_report.loc["excess_return_with_cost"]
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("年化收益率", f"{metrics['annualized_return']:.2%}")
        kpi_cols[1].metric("夏普比率", f"{metrics['information_ratio']:.2f}")
        kpi_cols[2].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
        kpi_cols[3].metric("换手率", f"{metrics['turnover_rate']:.2f}")

        st.markdown("**详细回测报告**")
        st.dataframe(portfolio_report)

def main():
    st.set_page_config(layout="wide", page_title="Qlib 可视化工具")
    st.sidebar.image("https://avatars.githubusercontent.com/u/65423353?s=200&v=4", width=100)
    st.sidebar.title("Qlib 可视化面板")
    page_options = ["数据管理", "模型训练", "投资组合预测", "模型评估", "策略回测"]
    page = st.sidebar.radio("选择功能页面", page_options) # horizontal=True
    if page == "数据管理": data_management_page()
    elif page == "模型训练": model_training_page()
    elif page == "投资组合预测": prediction_page()
    elif page == "模型评估": model_evaluation_page()
    elif page == "策略回测": backtesting_page()

if __name__ == "__main__":
    main()
