电商用户行为分析与智能运营系统

项目介绍
本项目基于电商用户行为数据集完成全流程分析与可视化，实现用户行为挖掘、价值分层、转化链路监控，并结合本地大模型生成智能化运营策略。项目可直接用于学习、业务复盘和数据分析展示。

技术栈
Python
Pandas 数据处理
Scikit-learn 聚类建模
Streamlit 交互式可视化
Pyecharts 图表生成
Ollama + Qwen3 本地AI服务

快速运行
1. 安装项目依赖
pip install -r requirements.txt
2. 执行数据清洗
python src/data_clean.py
3. 启动可视化看板
streamlit run src/streamlit_app.py

核心功能
数据预处理与异常值过滤
用户行为漏斗转化分析
RFM 模型用户价值分层
K-Means 用户聚类
交互式数据看板
本地AI生成运营策略建议

文件结构
src/ 核心代码目录
data/ 数据存储目录
requirements.txt 依赖文件
.env 环境配置文件

注意事项
1. 数据集未上传至仓库，需自行下载并放置至指定目录
2. 运行AI相关功能需提前启动 Ollama 服务并加载 Qwen3 模型
3. 项目仅用于学习与实践使用