import os
import logging
from crewai import Agent, Task, Crew, Process, LLM

# 设置环境变量避免OpenAI验证
os.environ["OPENAI_API_KEY"] = "dummy-key"

# 抑制错误日志
logging.getLogger().setLevel(logging.CRITICAL)
os.environ["PYTHONWARNINGS"] = "ignore"


llm = LLM(
    model="openai/Qwen/Qwen3-8B",
    provider="siliconflow",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-sbdwwnjulncostljtkbapfltvfwdzannjhpumoqbglgjewea",
    temperature=0,
)

embeddings = {
    "provider": "siliconflow",
    "config": {
        "model": 'BAAI/bge-large-zh-v1.5',
        "base_url": 'https://api.siliconflow.cn/v1',
        "api_key": 'sk-sbdwwnjulncostljtkbapfltvfwdzannjhpumoqbglgjewea'
    }
}


# 定义智能体
researcher = Agent(
    role='市场研究员',
    goal='收集电动汽车市场数据',
    backstory="专注于新能源汽车领域的分析师。",
    memory=True,
    verbose=True,
    llm=llm,
    embeddings=embeddings
)

analyst = Agent(
    role='数据分析师',
    goal='分析市场趋势',
    backstory="擅长数据处理和分析的专家。",
    memory=True,
    verbose=True,
    llm=llm,
    embeddings=embeddings
)

# 简化任务以避免长时间运行
research_task = Task(
    description="收集2024年电动汽车市场的基本信息，包括主要厂商和技术发展。",
    agent=researcher,
    expected_output='简要的市场概述'
)

analysis_task = Task(
    description="分析电动汽车市场的一个主要趋势。",
    agent=analyst,
    expected_output='简要的趋势分析'
)

# 创建团队
ev_crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True,
    memory=True
)

# 运行团队
if __name__ == "__main__":
    print("🚀 开始电动汽车市场调研任务...")
    print("📋 使用模型: openai/Qwen/Qwen3-8B via siliconflow")
    print("🔤 嵌入模型: BAAI/bge-large-zh-v1.5")
    
    try:
        result = ev_crew.kickoff()
        print("\n" + "="*50)
        print("✅ 任务完成！")
        print("="*50)
        print(result)
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
