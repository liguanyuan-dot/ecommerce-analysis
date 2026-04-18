# -*- coding: utf-8 -*-
"""
数据清洗模块
功能：读取原始数据、处理缺失值、过滤异常值、时间格式转换、保存清洗后数据
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 加载.env配置文件
load_dotenv()


def load_raw_data(file_path=None):
    """
    读取原始CSV数据文件
    :param file_path: 数据文件路径，默认从.env读取
    :return: 原始DataFrame
    """
    if file_path is None:
        file_path = os.getenv("RAW_DATA_PATH", "data/raw/UserBehavior_2025.csv")
    
    print(f"正在读取数据: {file_path}")
    df = pd.read_csv(file_path)
    print(f"原始数据量: {len(df)} 行, {len(df.columns)} 列")
    return df


def handle_missing_values(df):
    """
    处理缺失值
    - 删除关键字段（用户ID、商品ID、行为类型、时间戳）为空的行
    - 数值字段（售价）用中位数填充
    - 文本字段（品牌、商品名称、商品类别）用"未知"填充
    :param df: 原始DataFrame
    :return: 处理后的DataFrame
    """
    print("开始处理缺失值...")
    
    # 记录原始数据量
    original_count = len(df)
    
    # 删除关键字段为空的行
    key_columns = ["用户ID", "商品ID", "行为类型", "时间戳"]
    df = df.dropna(subset=key_columns)
    
    # 售价用中位数填充
    if "售价" in df.columns:
        median_price = df["售价"].median()
        df["售价"] = df["售价"].fillna(median_price)
    
    # 文本字段用"未知"填充
    text_columns = ["品牌", "商品名称", "商品类别"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("未知")
    
    # 删除品牌ID、商品类目ID为空的行
    id_columns = ["品牌ID", "商品类目ID"]
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    removed_count = original_count - len(df)
    print(f"缺失值处理完成，删除 {removed_count} 行")
    return df


def filter_outliers(df):
    """
    过滤异常值
    - 售价：保留0-100000之间的数据
    - 用户ID：必须为正整数
    - 时间戳：保留合理范围内的数据（2025年）
    :param df: 处理后的DataFrame
    :return: 过滤后的DataFrame
    """
    print("开始过滤异常值...")
    
    original_count = len(df)
    
    # 售价异常值过滤（0到10万之间）
    if "售价" in df.columns:
        df = df[(df["售价"] >= 0) & (df["售价"] <= 100000)]
    
    # 用户ID必须为正整数
    df = df[df["用户ID"] > 0]
    
    # 时间戳过滤（2025年1月1日到2025年12月31日）
    # 2025-01-01 00:00:00 = 1735689600
    # 2025-12-31 23:59:59 = 1767225599
    df = df[(df["时间戳"] >= 1735689600) & (df["时间戳"] <= 1767225599)]
    
    removed_count = original_count - len(df)
    print(f"异常值过滤完成，删除 {removed_count} 行")
    return df


def convert_timestamp(df):
    """
    将Unix时间戳转换为标准datetime格式
    并提取小时、日期、月份、星期等时间特征
    :param df: 过滤后的DataFrame
    :return: 添加时间特征后的DataFrame
    """
    print("开始转换时间格式...")
    
    # 将时间戳转换为datetime
    df["时间"] = pd.to_datetime(df["时间戳"], unit="s")
    
    # 提取时间特征
    df["日期"] = df["时间"].dt.date
    df["小时"] = df["时间"].dt.hour
    df["月份"] = df["时间"].dt.month
    df["星期"] = df["时间"].dt.dayofweek  # 0=周一, 6=周日
    df["星期名称"] = df["时间"].dt.day_name()
    
    print("时间格式转换完成")
    return df


def clean_data(df=None):
    """
    完整的数据清洗流程
    :param df: 原始DataFrame，如果为None则自动读取
    :return: 清洗后的DataFrame
    """
    # 1. 读取数据
    if df is None:
        df = load_raw_data()
    
    # 2. 处理缺失值
    df = handle_missing_values(df)
    
    # 3. 过滤异常值
    df = filter_outliers(df)
    
    # 4. 转换时间格式
    df = convert_timestamp(df)
    
    print(f"\n数据清洗完成！最终数据量: {len(df)} 行")
    return df


def save_cleaned_data(df, output_path=None):
    """
    保存清洗后的数据到CSV文件
    :param df: 清洗后的DataFrame
    :param output_path: 输出文件路径，默认从.env读取
    """
    if output_path is None:
        output_path = os.getenv("PROCESSED_DATA_PATH", "data/processed/user_behavior_cleaned.csv")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"清洗后数据已保存至: {output_path}")


if __name__ == "__main__":
    # 运行数据清洗流程
    cleaned_df = clean_data()
    save_cleaned_data(cleaned_df)
    
    # 打印清洗后数据信息
    print("\n清洗后数据概览：")
    print(cleaned_df.info())
    print("\n行为类型分布：")
    print(cleaned_df["行为类型"].value_counts())
