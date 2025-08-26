import os
import logging
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 抑制错误日志
logging.getLogger().setLevel(logging.CRITICAL)
os.environ["PYTHONWARNINGS"] = "ignore"

# 设置环境变量
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"

# 初始化嵌入模型
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

# 初始化LLM
llm = ChatOpenAI(
    model="ollama/qwen3:4b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.3
)

# 创建简单的智能体
agent = Agent(
    role='测试员',
    goal='测试内存功能',
    backstory="负责测试系统功能的专家。",
    memory=True,
    verbose=True,
    llm=llm,
    embeddings=embeddings
)

# 简单任务
task = Task(
    description="说一句关于电动汽车的话。",
    agent=agent,
    expected_output='一句简短的话'
)

# 创建团队
crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
    memory=True
)

if __name__ == "__main__":
    print("🧪 测试内存功能...")
    try:
        result = crew.kickoff()
        print(f"✅ 成功！结果: {result}")
    except Exception as e:
        print(f"❌ 错误: {e}")