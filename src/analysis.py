# -*- coding: utf-8 -*-
"""
数据分析模块
功能：RFM用户分层、漏斗分析、K-Means用户聚类、商品分析、时间分析
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def rfm_analysis(df):
    """
    RFM用户价值分析
    R (Recency): 最近一次购买时间距离分析日期的天数
    F (Frequency): 统计周期内购买次数
    M (Monetary): 统计周期内消费总金额
    :param df: 清洗后的DataFrame
    :return: RFM分析结果DataFrame
    """
    print("开始RFM分析...")
    
    # 只保留购买行为
    buy_df = df[df["行为类型"] == "buy"].copy()
    
    # 计算分析基准日期（数据中最大日期）
    reference_date = buy_df["时间"].max()
    
    # 按用户ID分组计算RFM指标
    rfm = buy_df.groupby("用户ID").agg({
        "时间": lambda x: (reference_date - x.max()).days,  # R: 最近购买距今天数
        "商品ID": "count",  # F: 购买次数
        "售价": "sum"  # M: 消费总金额
    }).rename(columns={
        "时间": "Recency",
        "商品ID": "Frequency",
        "售价": "Monetary"
    })
    
    # 对R、F、M分别打分（1-5分）
    # Recency越小越好，所以用ascending=False
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop")
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    
    # 转换为数值类型
    rfm["R_Score"] = rfm["R_Score"].astype(int)
    rfm["F_Score"] = rfm["F_Score"].astype(int)
    rfm["M_Score"] = rfm["M_Score"].astype(int)
    
    # 计算总分
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    
    # 用户分层
    def classify_user(row):
        if row["R_Score"] >= 4 and row["F_Score"] >= 4 and row["M_Score"] >= 4:
            return "重要价值用户"
        elif row["R_Score"] < 4 and row["F_Score"] >= 4 and row["M_Score"] >= 4:
            return "重要保持用户"
        elif row["R_Score"] >= 4 and row["F_Score"] < 4 and row["M_Score"] >= 4:
            return "重要发展用户"
        elif row["R_Score"] < 4 and row["F_Score"] < 4 and row["M_Score"] >= 4:
            return "重要挽留用户"
        elif row["R_Score"] >= 4 and row["F_Score"] >= 4 and row["M_Score"] < 4:
            return "一般价值用户"
        elif row["R_Score"] < 4 and row["F_Score"] >= 4 and row["M_Score"] < 4:
            return "一般保持用户"
        elif row["R_Score"] >= 4 and row["F_Score"] < 4 and row["M_Score"] < 4:
            return "一般发展用户"
        else:
            return "一般挽留用户"
    
    rfm["用户层级"] = rfm.apply(classify_user, axis=1)
    rfm = rfm.reset_index()
    
    print(f"RFM分析完成，共分析 {len(rfm)} 个用户")
    return rfm


def funnel_analysis(df):
    """
    用户行为漏斗分析
    典型电商漏斗：浏览(pv) -> 收藏(fav) -> 加购(cart) -> 购买(buy)
    :param df: 清洗后的DataFrame
    :return: 漏斗数据字典
    """
    print("开始漏斗分析...")
    
    # 统计各行为类型的用户数
    behavior_counts = df.groupby("行为类型")["用户ID"].nunique()
    
    # 构建漏斗数据（根据实际存在的行为类型）
    funnel = {
        "浏览(pv)": int(behavior_counts.get("pv", 0)),
        "收藏(fav)": int(behavior_counts.get("fav", 0)),
        "加购(cart)": int(behavior_counts.get("cart", 0)),
        "购买(buy)": int(behavior_counts.get("buy", 0))
    }
    
    # 计算转化率（以总用户数为基准，因为购买用户可能没有浏览记录）
    total_users = df["用户ID"].nunique()  # 使用总用户数作为分母
    pv_users = funnel["浏览(pv)"]
    
    if total_users > 0:
        funnel["浏览到收藏转化率"] = round(funnel["收藏(fav)"] / pv_users * 100, 2) if pv_users > 0 else 0
        funnel["浏览到加购转化率"] = round(funnel["加购(cart)"] / pv_users * 100, 2) if pv_users > 0 else 0
        funnel["浏览到购买转化率"] = round(funnel["购买(buy)"] / pv_users * 100, 2) if pv_users > 0 else 0
        funnel["收藏到加购转化率"] = round(funnel["加购(cart)"] / funnel["收藏(fav)"] * 100, 2) if funnel["收藏(fav)"] > 0 else 0
        funnel["加购到购买转化率"] = round(funnel["购买(buy)"] / funnel["加购(cart)"] * 100, 2) if funnel["加购(cart)"] > 0 else 0
        funnel["整体购买转化率"] = round(funnel["购买(buy)"] / total_users * 100, 2)
        funnel["总用户数"] = total_users
    
    print("漏斗分析完成")
    return funnel


def kmeans_clustering(df, n_clusters=4):
    """
    K-Means用户聚类分析
    基于RFM特征进行用户分群
    :param df: 清洗后的DataFrame
    :param n_clusters: 聚类数量
    :return: 聚类结果DataFrame
    """
    print(f"开始K-Means聚类分析 (n={n_clusters})...")
    
    # 只保留购买行为
    buy_df = df[df["行为类型"] == "buy"].copy()
    
    # 计算RFM特征
    reference_date = buy_df["时间"].max()
    
    user_features = buy_df.groupby("用户ID").agg({
        "时间": lambda x: (reference_date - x.max()).days,
        "商品ID": "count",
        "售价": ["sum", "mean"]
    }).reset_index()
    
    user_features.columns = ["用户ID", "Recency", "Frequency", "Monetary_Sum", "Monetary_Mean"]
    
    # 特征标准化
    scaler = StandardScaler()
    features = user_features[["Recency", "Frequency", "Monetary_Sum", "Monetary_Mean"]].values
    features_scaled = scaler.fit_transform(features)
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_features["Cluster"] = kmeans.fit_predict(features_scaled)
    
    # 计算每个聚类的特征均值
    cluster_summary = user_features.groupby("Cluster").agg({
        "用户ID": "count",
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary_Sum": "mean",
        "Monetary_Mean": "mean"
    }).rename(columns={"用户ID": "用户数"})
    
    # 为聚类命名（基于分位数而非均值，因为数据分布不均匀）
    freq_q75 = cluster_summary["Frequency"].quantile(0.75)
    freq_q25 = cluster_summary["Frequency"].quantile(0.25)
    monetary_q75 = cluster_summary["Monetary_Sum"].quantile(0.75)
    monetary_q25 = cluster_summary["Monetary_Sum"].quantile(0.25)
    recency_q25 = cluster_summary["Recency"].quantile(0.25)  # R值越小越好
    recency_q75 = cluster_summary["Recency"].quantile(0.75)
    
    def name_cluster(row):
        # R值越小越好（最近购买），F和M越大越好
        is_very_recent = row["Recency"] < recency_q25
        is_recent = row["Recency"] < recency_q75
        is_very_frequent = row["Frequency"] > freq_q75
        is_frequent = row["Frequency"] > freq_q25
        is_high_value = row["Monetary_Sum"] > monetary_q75
        is_medium_value = row["Monetary_Sum"] > monetary_q25
        
        if is_very_recent and is_very_frequent and is_high_value:
            return "高价值活跃用户"
        elif is_very_frequent and is_high_value:
            return "高价值用户"
        elif is_very_recent and is_very_frequent:
            return "高频活跃用户"
        elif is_high_value:
            return "高消费用户"
        elif is_very_frequent:
            return "高频用户"
        elif is_very_recent:
            return "活跃用户"
        elif is_medium_value:
            return "中等价值用户"
        else:
            return "低价值用户"
    
    cluster_summary["聚类名称"] = cluster_summary.apply(name_cluster, axis=1)
    
    print("K-Means聚类分析完成")
    return user_features, cluster_summary


def brand_analysis(df):
    """
    品牌销售分析
    :param df: 清洗后的DataFrame
    :return: 品牌分析结果DataFrame
    """
    print("开始品牌分析...")
    
    buy_df = df[df["行为类型"] == "buy"]
    
    # 按品牌统计
    brand_stats = buy_df.groupby("品牌").agg({
        "用户ID": "nunique",  # 购买用户数
        "商品ID": "count",  # 销量
        "售价": ["sum", "mean"]  # 总销售额、平均售价
    }).reset_index()
    
    brand_stats.columns = ["品牌", "购买用户数", "销量", "总销售额", "平均售价"]
    brand_stats = brand_stats.sort_values("总销售额", ascending=False)
    
    print("品牌分析完成")
    return brand_stats


def category_analysis(df):
    """
    商品类目分析
    :param df: 清洗后的DataFrame
    :return: 类目分析结果DataFrame
    """
    print("开始商品类目分析...")
    
    buy_df = df[df["行为类型"] == "buy"]
    
    # 按类目统计
    category_stats = buy_df.groupby("商品类别").agg({
        "用户ID": "nunique",
        "商品ID": "count",
        "售价": ["sum", "mean"]
    }).reset_index()
    
    category_stats.columns = ["商品类别", "购买用户数", "销量", "总销售额", "平均售价"]
    category_stats = category_stats.sort_values("总销售额", ascending=False)
    
    print("商品类目分析完成")
    return category_stats


def time_analysis(df):
    """
    时间维度分析（小时分布、星期分布、月度趋势）
    :param df: 清洗后的DataFrame
    :return: 时间分析结果字典
    """
    print("开始时间分析...")
    
    # 按小时统计行为分布
    hourly_behavior = df.groupby(["小时", "行为类型"]).size().unstack(fill_value=0)
    
    # 按星期统计
    weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    df["星期中文"] = df["星期"].map(weekday_map)
    weekday_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    
    weekday_behavior = df.groupby(["星期中文", "行为类型"]).size().unstack(fill_value=0)
    weekday_behavior = weekday_behavior.reindex(weekday_order)
    
    # 按日期统计购买趋势
    daily_buy = df[df["行为类型"] == "buy"].groupby("日期").agg({
        "用户ID": "nunique",
        "售价": "sum"
    }).reset_index()
    daily_buy.columns = ["日期", "购买用户数", "销售额"]
    
    print("时间分析完成")
    return {
        "hourly": hourly_behavior,
        "weekday": weekday_behavior,
        "daily": daily_buy
    }


def price_analysis(df):
    """
    价格区间分析
    :param df: 清洗后的DataFrame
    :return: 价格区间分析结果DataFrame
    """
    print("开始价格区间分析...")
    
    buy_df = df[df["行为类型"] == "buy"].copy()
    
    # 划分价格区间
    bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 100000]
    labels = ["0-50元", "50-100元", "100-200元", "200-500元", "500-1000元", "1000-2000元", "2000-5000元", "5000元以上"]
    
    buy_df["价格区间"] = pd.cut(buy_df["售价"], bins=bins, labels=labels, right=False)
    
    price_stats = buy_df.groupby("价格区间", observed=False).agg({
        "用户ID": "nunique",
        "商品ID": "count",
        "售价": "sum"
    }).reset_index()
    
    price_stats.columns = ["价格区间", "购买用户数", "销量", "总销售额"]
    
    print("价格区间分析完成")
    return price_stats


def generate_analysis_summary(rfm_result, funnel_result, cluster_summary):
    """
    生成分析结论摘要
    :param rfm_result: RFM分析结果
    :param funnel_result: 漏斗分析结果
    :param cluster_summary: 聚类摘要
    :return: 分析结论字符串
    """
    # RFM用户分布
    rfm_dist = rfm_result["用户层级"].value_counts()
    top_user_type = rfm_dist.index[0]
    
    # 漏斗转化率
    buy_rate = funnel_result.get("浏览到购买转化率", 0)
    
    # 聚类信息
    if cluster_summary is not None:
        cluster_count = len(cluster_summary)
        high_value_cluster = cluster_summary[cluster_summary["聚类名称"] == "高价值用户"]
        if len(high_value_cluster) > 0:
            high_value_users = int(high_value_cluster["用户数"].values[0])
        else:
            high_value_users = 0
    else:
        cluster_count = 0
        high_value_users = 0
    
    summary = f"""
【电商用户行为分析结论】

1. 用户价值分层：
   - 共分析 {len(rfm_result)} 个购买用户
   - 最多的用户类型是：{top_user_type}（{rfm_dist.values[0]}人）
   - 用户层级分布：{dict(rfm_dist)}

2. 转化漏斗：
   - 浏览用户数：{funnel_result.get('浏览(pv)', 0)}
   - 最终购买用户数：{funnel_result.get('购买(buy)', 0)}
   - 整体转化率：{buy_rate}%
   - 收藏到加购转化率：{funnel_result.get('收藏到加购转化率', 0)}%
   - 加购到购买转化率：{funnel_result.get('加购到购买转化率', 0)}%

3. 用户聚类：
   - 共分为 {cluster_count} 个用户群体
   - 高价值用户数：{high_value_users}人

4. 运营建议方向：
   - 针对{top_user_type}制定专项运营策略
   - 优化转化漏斗中流失率最高的环节
   - 对高价值用户提供专属权益和个性化推荐
"""
    return summary


if __name__ == "__main__":
    # 测试分析模块
    from data_clean import clean_data
    
    # 加载清洗后的数据
    df = clean_data()
    
    # 执行各项分析
    rfm_result = rfm_analysis(df)
    funnel_result = funnel_analysis(df)
    cluster_result, cluster_summary = kmeans_clustering(df)
    brand_result = brand_analysis(df)
    category_result = category_analysis(df)
    time_result = time_analysis(df)
    price_result = price_analysis(df)
    
    # 打印分析摘要
    summary = generate_analysis_summary(rfm_result, funnel_result, cluster_summary)
    print(summary)
