# -*- coding: utf-8 -*-
"""
Streamlit交互式可视化看板
功能：整合所有分析模块，提供交互式数据可视化界面
"""

import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_clean import clean_data, load_raw_data
from analysis import (
    rfm_analysis, funnel_analysis, kmeans_clustering,
    brand_analysis, category_analysis, time_analysis,
    price_analysis, generate_analysis_summary
)
from ai_service import create_ai_service


# 页面配置
st.set_page_config(
    page_title="电商用户行为分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_and_clean_data():
    """缓存加载和清洗数据，避免重复计算"""
    with st.spinner("正在加载和清洗数据..."):
        df = clean_data()
    return df


def plot_funnel_chart(funnel_data):
    """绘制漏斗图"""
    # 过滤掉值为0的阶段
    stages = ["浏览(pv)", "收藏(fav)", "加购(cart)", "购买(buy)"]
    values = [funnel_data.get(s, 0) for s in stages]
    
    # 只保留有数据的阶段
    valid_stages = [s for s, v in zip(stages, values) if v > 0]
    valid_values = [v for v in values if v > 0]
    
    if not valid_stages:
        return go.Figure().update_layout(title="无数据")
    
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"][:len(valid_stages)]
    
    fig = go.Figure(go.Funnel(
        y=valid_stages,
        x=valid_values,
        textposition="inside",
        textinfo="value+percent initial",
        marker={
            "color": colors,
            "line": {"width": [4] + [2] * (len(valid_stages) - 1), "color": ["white"] * len(valid_stages)}
        },
        connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
    ))
    
    fig.update_layout(
        title="用户行为转化漏斗",
        font={"size": 14},
        height=400
    )
    return fig


def plot_rfm_distribution(rfm_df):
    """绘制RFM用户层级分布饼图"""
    user_type_counts = rfm_df["用户层级"].value_counts().reset_index()
    user_type_counts.columns = ["用户层级", "用户数"]
    
    fig = px.pie(
        user_type_counts,
        values="用户数",
        names="用户层级",
        title="RFM用户层级分布",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=450)
    return fig


def plot_rfm_bar(rfm_df):
    """绘制RFM用户层级柱状图"""
    user_type_counts = rfm_df["用户层级"].value_counts().reset_index()
    user_type_counts.columns = ["用户层级", "用户数"]
    
    fig = px.bar(
        user_type_counts,
        x="用户层级",
        y="用户数",
        title="RFM用户层级用户数对比",
        color="用户数",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=400)
    return fig


def plot_hourly_heatmap(hourly_data):
    """绘制小时行为热力图"""
    # 重置索引以便绘图
    df_plot = hourly_data.reset_index()
    df_melted = df_plot.melt(id_vars=["小时"], var_name="行为类型", value_name="次数")
    
    fig = px.density_heatmap(
        df_melted,
        x="小时",
        y="行为类型",
        z="次数",
        title="24小时用户行为分布热力图",
        color_continuous_scale="YlOrRd"
    )
    fig.update_layout(height=350)
    return fig


def plot_weekly_behavior(weekday_data):
    """绘制星期行为分布图"""
    df_plot = weekday_data.reset_index()
    df_melted = df_plot.melt(id_vars=["星期中文"], var_name="行为类型", value_name="次数")
    
    fig = px.bar(
        df_melted,
        x="星期中文",
        y="次数",
        color="行为类型",
        title="星期用户行为分布",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"星期中文": "星期"}
    )
    fig.update_layout(height=400)
    return fig


def plot_daily_trend(daily_data):
    """绘制每日购买趋势图"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily_data["日期"],
            y=daily_data["购买用户数"],
            name="购买用户数",
            line={"color": "#636EFA", "width": 2}
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_data["日期"],
            y=daily_data["销售额"],
            name="销售额",
            marker_color="#EF553B",
            opacity=0.6
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="每日购买趋势",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="日期")
    fig.update_yaxes(title_text="购买用户数", secondary_y=False)
    fig.update_yaxes(title_text="销售额(元)", secondary_y=True)
    
    return fig


def plot_cluster_scatter(cluster_df):
    """绘制聚类散点图"""
    fig = px.scatter(
        cluster_df,
        x="Frequency",
        y="Monetary_Sum",
        color="Cluster",
        size="Monetary_Mean",
        hover_data=["用户ID"],
        title="用户聚类分布（购买频次 vs 消费总额）",
        color_continuous_scale="Viridis",
        labels={
            "Frequency": "购买频次",
            "Monetary_Sum": "消费总额(元)",
            "Monetary_Mean": "平均客单价(元)"
        }
    )
    fig.update_layout(height=450)
    return fig


def plot_brand_top10(brand_df):
    """绘制品牌TOP10柱状图"""
    top10 = brand_df.head(10)
    
    fig = px.bar(
        top10,
        x="品牌",
        y="总销售额",
        title="品牌销售额TOP10",
        color="总销售额",
        color_continuous_scale="Sunset"
    )
    fig.update_layout(height=400)
    return fig


def plot_category_pie(category_df):
    """绘制商品类目销售占比饼图"""
    top_categories = category_df.head(8)
    
    fig = px.pie(
        top_categories,
        values="总销售额",
        names="商品类别",
        title="商品类目销售占比TOP8",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=450)
    return fig


def plot_price_distribution(price_df):
    """绘制价格区间分布图"""
    fig = px.bar(
        price_df,
        x="价格区间",
        y="销量",
        title="价格区间销量分布",
        color="总销售额",
        color_continuous_scale="Tealgrn"
    )
    fig.update_layout(height=400)
    return fig


def render_dashboard():
    """主看板渲染函数"""
    # 侧边栏
    st.sidebar.title("📊 电商用户行为分析系统")
    st.sidebar.markdown("---")
    
    # 页面选择
    page = st.sidebar.radio(
        "选择分析页面",
        ["数据概览", "RFM用户分层", "转化漏斗", "用户聚类", "品牌与类目", "时间分析", "价格分析", "AI智能分析"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("基于淘宝精简用户行为数据集\n使用Ollama+Qwen3本地模型")
    
    # 加载数据
    df = load_and_clean_data()
    
    # 数据概览页
    if page == "数据概览":
        st.title("📈 数据概览")
        
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        total_users = df["用户ID"].nunique()
        total_products = df["商品ID"].nunique()
        total_behaviors = len(df)
        buy_count = len(df[df["行为类型"] == "buy"])
        
        col1.metric("总用户数", f"{total_users:,}")
        col2.metric("商品总数", f"{total_products:,}")
        col3.metric("行为记录数", f"{total_behaviors:,}")
        col4.metric("购买次数", f"{buy_count:,}")
        
        st.markdown("---")
        
        # 行为类型分布
        col1, col2 = st.columns(2)
        
        with col1:
            behavior_counts = df["行为类型"].value_counts().reset_index()
            behavior_counts.columns = ["行为类型", "次数"]
            fig = px.pie(
                behavior_counts,
                values="次数",
                names="行为类型",
                title="行为类型分布"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("数据字段说明")
            st.dataframe(df[["用户ID", "商品ID", "品牌", "行为类型", "时间", "售价"]].head(10), use_container_width=True)
        
        # 行为类型统计
        st.subheader("各行为类型统计")
        behavior_stats = df.groupby("行为类型").agg({
            "用户ID": "nunique",
            "商品ID": "count",
            "售价": ["sum", "mean"]
        }).reset_index()
        behavior_stats.columns = ["行为类型", "用户数", "次数", "总金额", "平均金额"]
        st.dataframe(behavior_stats, use_container_width=True)
    
    # RFM用户分层页
    elif page == "RFM用户分层":
        st.title("🎯 RFM用户分层分析")
        
        with st.spinner("正在计算RFM指标..."):
            rfm_df = rfm_analysis(df)
        
        # 用户层级分布
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_rfm_distribution(rfm_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_rfm_bar(rfm_df)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # RFM详细数据
        st.subheader("RFM用户分层详细数据")
        
        # 按用户层级汇总
        rfm_summary = rfm_df.groupby("用户层级").agg({
            "用户ID": "count",
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "RFM_Score": "mean"
        }).reset_index()
        rfm_summary.columns = ["用户层级", "用户数", "平均R值", "平均F值", "平均M值", "平均RFM得分"]
        st.dataframe(rfm_summary, use_container_width=True)
        
        # 筛选查看
        st.subheader("查看特定层级用户")
        selected_level = st.selectbox(
            "选择用户层级",
            rfm_df["用户层级"].unique()
        )
        
        filtered_users = rfm_df[rfm_df["用户层级"] == selected_level]
        st.dataframe(filtered_users.head(50), use_container_width=True)
        st.write(f"共 {len(filtered_users)} 个用户")
    
    # 转化漏斗页
    elif page == "转化漏斗":
        st.title("🔻 用户行为转化漏斗")
        
        with st.spinner("正在计算漏斗数据..."):
            funnel_data = funnel_analysis(df)
        
        # 漏斗图
        fig = plot_funnel_chart(funnel_data)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 转化率指标
        st.subheader("关键转化率指标")
        
        # 说明：由于数据中购买用户数远大于浏览用户数，使用总用户数计算整体转化率
        st.info(f"说明：总用户数 {funnel_data.get('总用户数', 0)}，浏览用户 {funnel_data.get('浏览(pv)', 0)}，购买用户 {funnel_data.get('购买(buy)', 0)}。部分用户直接购买未产生浏览记录。")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("浏览到收藏", f"{funnel_data.get('浏览到收藏转化率', 0)}%")
        col2.metric("浏览到购买", f"{funnel_data.get('浏览到购买转化率', 0)}%")
        col3.metric("整体购买转化率", f"{funnel_data.get('整体购买转化率', 0)}%")
        
        if funnel_data.get('收藏(fav)', 0) > 0:
            col1, col2 = st.columns(2)
            col1.metric("收藏到加购", f"{funnel_data.get('收藏到加购转化率', 0)}%")
            if funnel_data.get('加购(cart)', 0) > 0:
                col2.metric("加购到购买", f"{funnel_data.get('加购到购买转化率', 0)}%")
            else:
                st.info("数据中暂无加购(cart)行为记录")
        
        st.markdown("---")
        
        # 各行为用户数
        st.subheader("各行为独立用户数")
        st.info("注：数据中购买用户数(47095)远大于浏览用户数(14011)，说明大量用户直接购买而未产生浏览记录")
        behavior_users = df.groupby("行为类型")["用户ID"].nunique().reset_index()
        behavior_users.columns = ["行为类型", "独立用户数"]
        behavior_users = behavior_users.sort_values("独立用户数", ascending=False)
        fig = px.bar(
            behavior_users,
            x="行为类型",
            y="独立用户数",
            title="各行为独立用户数",
            color="独立用户数",
            color_continuous_scale="Blues",
            text="独立用户数"
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # 用户聚类页
    elif page == "用户聚类":
        st.title("🔬 K-Means用户聚类分析")
        
        # 聚类数量选择
        n_clusters = st.sidebar.slider("选择聚类数量", 2, 8, 4)
        
        with st.spinner("正在进行聚类分析..."):
            cluster_df, cluster_summary = kmeans_clustering(df, n_clusters)
        
        # 聚类散点图
        fig = plot_cluster_scatter(cluster_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 聚类摘要
        st.subheader("聚类特征摘要")
        cluster_summary_display = cluster_summary.copy()
        cluster_summary_display.columns = ["用户数", "平均R值", "平均F值", "平均消费总额", "平均客单价", "聚类名称"]
        cluster_summary_display.insert(0, "聚类ID", cluster_summary_display.index)
        st.dataframe(cluster_summary_display, use_container_width=True)
        
        # 聚类分布饼图
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                cluster_summary,
                values="用户数",
                names="聚类名称",
                title="聚类用户分布"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 品牌与类目页
    elif page == "品牌与类目":
        st.title("🏷️ 品牌与商品类目分析")
        
        with st.spinner("正在分析品牌和类目数据..."):
            brand_df = brand_analysis(df)
            category_df = category_analysis(df)
        
        # 品牌分析
        st.subheader("品牌销售分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_brand_top10(brand_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 品牌数量统计
            top_brands = brand_df.head(10)
            fig = px.bar(
                top_brands,
                x="品牌",
                y="销量",
                title="品牌销量TOP10",
                color="销量",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 类目分析
        st.subheader("商品类目分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_category_pie(category_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_cats = category_df.head(10)
            fig = px.bar(
                top_cats,
                x="商品类别",
                y="总销售额",
                title="类目销售额TOP10",
                color="总销售额",
                color_continuous_scale="Sunset"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 详细数据表格
        st.subheader("品牌详细数据")
        st.dataframe(brand_df.head(20), use_container_width=True)
        
        st.subheader("类目详细数据")
        st.dataframe(category_df, use_container_width=True)
    
    # 时间分析页
    elif page == "时间分析":
        st.title("⏰ 时间维度分析")
        
        with st.spinner("正在分析时间数据..."):
            time_data = time_analysis(df)
        
        # 小时分布
        st.subheader("24小时行为分布")
        fig = plot_hourly_heatmap(time_data["hourly"])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 星期分布
        st.subheader("星期行为分布")
        fig = plot_weekly_behavior(time_data["weekday"])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 每日趋势
        st.subheader("每日购买趋势")
        fig = plot_daily_trend(time_data["daily"])
        st.plotly_chart(fig, use_container_width=True)
    
    # 价格分析页
    elif page == "价格分析":
        st.title("💰 价格区间分析")
        
        with st.spinner("正在分析价格数据..."):
            price_df = price_analysis(df)
        
        # 价格分布图
        fig = plot_price_distribution(price_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 价格区间详细数据
        st.subheader("价格区间详细数据")
        st.dataframe(price_df, use_container_width=True)
        
        # 价格区间占比
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                price_df,
                values="销量",
                names="价格区间",
                title="各价格区间销量占比"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                price_df,
                values="总销售额",
                names="价格区间",
                title="各价格区间销售额占比"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # AI智能分析页
    elif page == "AI智能分析":
        st.title("🤖 AI智能分析助手")
        
        # 初始化AI服务
        try:
            ai_service = create_ai_service()
            ai_available = True
        except Exception as e:
            st.error(f"AI服务初始化失败: {e}")
            st.info("请确保Ollama服务已启动，并已下载qwen3模型")
            ai_available = False
        
        if ai_available:
            # 生成运营策略
            st.subheader("AI运营策略生成")
            
            if st.button("生成运营策略", type="primary"):
                with st.spinner("AI正在分析数据并生成策略..."):
                    # 准备分析数据
                    rfm_df = rfm_analysis(df)
                    funnel_data = funnel_analysis(df)
                    
                    # 转换为字典格式
                    rfm_dist = rfm_df["用户层级"].value_counts().to_dict()
                    rfm_data = {
                        "用户层级分布": rfm_dist,
                        "总用户数": len(rfm_df)
                    }
                    
                    cluster_df, cluster_summary = kmeans_clustering(df)
                    cluster_data = cluster_summary.to_dict()
                    
                    # 调用AI生成策略
                    strategy = ai_service.generate_operation_strategy(rfm_data, funnel_data, cluster_data)
                    st.markdown(strategy)
            
            st.markdown("---")
            
            # 生成分析报告
            st.subheader("AI数据分析报告")
            
            if st.button("生成分析报告"):
                with st.spinner("AI正在生成分析报告..."):
                    rfm_df = rfm_analysis(df)
                    funnel_data = funnel_analysis(df)
                    brand_df = brand_analysis(df)
                    category_df = category_analysis(df)
                    
                    summary = generate_analysis_summary(rfm_df, funnel_data, None)
                    
                    brand_data = brand_df.head(10).to_dict()
                    category_data = category_df.head(10).to_dict()
                    
                    report = ai_service.generate_analysis_report(summary, brand_data, category_data)
                    st.markdown(report)
            
            st.markdown("---")
            
            # 自由问答
            st.subheader("数据问答")
            question = st.text_input("输入你的问题，AI将基于数据回答：")
            
            if question and st.button("提问"):
                with st.spinner("AI正在思考..."):
                    # 准备上下文数据
                    context = {
                        "总用户数": df["用户ID"].nunique(),
                        "总商品数": df["商品ID"].nunique(),
                        "行为记录数": len(df),
                        "购买用户数": df[df["行为类型"] == "buy"]["用户ID"].nunique(),
                        "平均售价": round(df["售价"].mean(), 2),
                        "品牌数量": df["品牌"].nunique(),
                        "类目数量": df["商品类别"].nunique()
                    }
                    
                    answer = ai_service.answer_question(question, context)
                    st.markdown(answer)


if __name__ == "__main__":
    render_dashboard()
