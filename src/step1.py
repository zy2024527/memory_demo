import os
import logging
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 禁用所有错误日志
logging.getLogger().setLevel(logging.CRITICAL)
os.environ["PYTHONWARNINGS"] = "ignore"


# 设置环境变量，使用OpenAI兼容的API
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"

# 初始化中文嵌入模型
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    device = 'cpu'

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# 使用OpenAI兼容的接口连接Ollama
llm = ChatOpenAI(
    model="ollama/qwen3:4b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.3
)



# 定义智能体
researcher = Agent(
    role='市场研究员',
    goal='收集电动汽车市场数据',
    backstory="专注于新能源汽车领域的分析师。",
    memory=True,
    verbose=True,
    llm=llm,  # 使用明确初始化的 Ollama LLM
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
    print("📋 使用模型: qwen3:4b via Ollama")
    print("🔤 嵌入模型: BAAI/bge-small-zh-v1.5")
    
    # 检查 Ollama 服务是否运行
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama 服务连接正常")
            models = response.json().get('models', [])
            if models:
                print("📦 已安装的模型:")
                for model in models:
                    print(f"   - {model.get('name')}")
            else:
                print("❌ 没有找到已安装的模型")
                print("请运行: ollama pull qwen3:4b")
        else:
            print("❌ Ollama 服务异常")
    except requests.ConnectionError:
        print("❌ 无法连接到 Ollama 服务")
        print("请确保 Ollama 正在运行: ollama serve")
        exit(1)
    
    try:
        result = ev_crew.kickoff()
        print("\n" + "="*50)
        print("✅ 任务完成！")
        print("="*50)
        print(result)
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("\n🔧 故障排除步骤:")
        print("1. 确保 Ollama 服务运行: ollama serve")
        print("2. 确认模型已下载: ollama pull qwen3:4b")
        print("3. 检查模型列表: ollama list")
        print("4. 测试模型: ollama run qwen3:4b '你好'")