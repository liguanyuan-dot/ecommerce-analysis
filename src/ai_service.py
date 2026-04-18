# -*- coding: utf-8 -*-
"""
AI服务模块
功能：调用本地Ollama Qwen3模型，基于分析数据生成运营策略和分析结论
"""

import os
import json
import ollama
from dotenv import load_dotenv

# 加载.env配置文件
load_dotenv()


class AIService:
    """AI服务类，封装Ollama模型调用逻辑"""
    
    def __init__(self, model=None, base_url=None):
        """
        初始化AI服务
        :param model: 模型名称，默认从.env读取
        :param base_url: Ollama服务地址，默认从.env读取
        """
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # 初始化Ollama客户端
        self.client = ollama.Client(host=self.base_url)
        
        print(f"AI服务已初始化，模型: {self.model}, 地址: {self.base_url}")
    
    def chat(self, prompt, system_prompt=None, temperature=0.7):
        """
        调用大模型进行对话
        :param prompt: 用户输入
        :param system_prompt: 系统提示词
        :param temperature: 温度参数，控制输出随机性
        :return: 模型回复文本
        """
        messages = []
        
        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加用户输入
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature}
            )
            return response["message"]["content"]
        except Exception as e:
            error_msg = f"调用AI模型失败: {str(e)}\n\n请检查：\n1. Ollama服务是否已启动（运行 'ollama serve'）\n2. 模型 '{self.model}' 是否已下载（运行 'ollama pull {self.model}'）\n3. 模型地址 '{self.base_url}' 是否正确"
            print(error_msg)
            return error_msg
    
    def generate_operation_strategy(self, rfm_data, funnel_data, cluster_data=None):
        """
        基于分析数据生成运营策略
        :param rfm_data: RFM分析结果字典
        :param funnel_data: 漏斗分析结果字典
        :param cluster_data: 聚类分析结果字典（可选）
        :return: 运营策略文本
        """
        # 构建系统提示词
        system_prompt = """你是一位资深的电商运营专家和数据分析师。
请根据提供的用户行为分析数据，生成专业、可执行的运营策略建议。
要求：
1. 策略要具体、可量化、可执行
2. 针对不同用户群体给出差异化策略
3. 包含短期和长期策略
4. 使用中文回复
5. 格式清晰，分点说明"""
        
        # 构建用户提示词
        prompt = f"""请根据以下电商用户行为分析数据，生成详细的运营策略：

【RFM用户分层数据】
{json.dumps(rfm_data, ensure_ascii=False, indent=2)}

【转化漏斗数据】
{json.dumps(funnel_data, ensure_ascii=False, indent=2)}

【用户聚类数据】
{json.dumps(cluster_data, ensure_ascii=False, indent=2) if cluster_data else '暂无聚类数据'}

请从以下维度给出运营策略：
1. 用户分层运营策略（针对不同RFM层级用户）
2. 转化漏斗优化策略（提升各环节转化率）
3. 用户留存与激活策略
4. 精准营销策略
5. 产品优化建议"""
        
        return self.chat(prompt, system_prompt)
    
    def generate_analysis_report(self, analysis_summary, brand_data=None, category_data=None, time_data=None):
        """
        生成数据分析报告
        :param analysis_summary: 分析摘要
        :param brand_data: 品牌分析数据
        :param category_data: 类目分析数据
        :param time_data: 时间分析数据
        :return: 分析报告文本
        """
        system_prompt = """你是一位资深的电商数据分析师。
请根据提供的数据分析结果，生成一份专业的数据分析报告。
要求：
1. 报告结构清晰，包含数据概览、核心发现、趋势分析、结论建议
2. 用数据说话，引用具体数值
3. 给出可落地的业务建议
4. 使用中文回复
5. 格式规范，适合向管理层汇报"""
        
        prompt = f"""请根据以下数据分析结果，生成一份完整的电商用户行为分析报告：

【分析摘要】
{analysis_summary}

【品牌销售数据】
{json.dumps(brand_data, ensure_ascii=False, indent=2) if brand_data else '暂无品牌数据'}

【商品类目数据】
{json.dumps(category_data, ensure_ascii=False, indent=2) if category_data else '暂无类目数据'}

【时间趋势数据】
{json.dumps(time_data, ensure_ascii=False, indent=2) if time_data else '暂无时间数据'}

请生成包含以下结构的报告：
1. 数据概览（样本量、时间范围等）
2. 核心发现（3-5个关键洞察）
3. 用户行为特征分析
4. 销售趋势分析
5. 业务建议与行动计划"""
        
        return self.chat(prompt, system_prompt)
    
    def generate_product_recommendation(self, user_behavior, top_products=None):
        """
        基于用户行为生成个性化推荐建议
        :param user_behavior: 用户行为数据
        :param top_products: 热门商品列表
        :return: 推荐建议文本
        """
        system_prompt = """你是一位电商推荐系统专家。
请根据用户行为数据和热门商品，生成个性化推荐策略。
要求：
1. 推荐逻辑清晰
2. 说明推荐理由
3. 考虑用户偏好和购买力
4. 使用中文回复"""
        
        prompt = f"""请根据以下数据生成个性化推荐策略：

【用户行为特征】
{json.dumps(user_behavior, ensure_ascii=False, indent=2)}

【热门商品】
{json.dumps(top_products, ensure_ascii=False, indent=2) if top_products else '暂无热门商品数据'}

请给出：
1. 推荐算法策略说明
2. 针对不同用户群体的推荐方案
3. 推荐位展示策略
4. 推荐效果评估指标"""
        
        return self.chat(prompt, system_prompt)
    
    def answer_question(self, question, context_data=None):
        """
        回答关于数据的自然语言问题
        :param question: 用户问题
        :param context_data: 相关数据上下文
        :return: 回答文本
        """
        system_prompt = """你是一位电商数据分析助手。
请根据提供的数据上下文，用通俗易懂的语言回答用户的问题。
要求：
1. 回答准确、简洁
2. 用数据支撑结论
3. 适当给出建议
4. 使用中文回复"""
        
        prompt = f"""用户问题：{question}

【相关数据上下文】
{json.dumps(context_data, ensure_ascii=False, indent=2) if context_data else '暂无数据上下文'}

请回答用户的问题。"""
        
        return self.chat(prompt, system_prompt)


def create_ai_service():
    """
    工厂函数：创建AI服务实例
    :return: AIService实例
    """
    return AIService()


if __name__ == "__main__":
    # 测试AI服务
    ai = create_ai_service()
    
    # 测试对话
    test_data = {
        "用户总数": 1000,
        "购买转化率": "5.2%",
        "平均客单价": "256元"
    }
    
    response = ai.answer_question(
        "如何提高电商平台的购买转化率？",
        test_data
    )
    print(response)
