import os
from crewai import Agent, Task, Crew, Process, LLM
# from openai import APIStatusError
# import sys
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from typing import List, Optional

# 很好！APIStatusError问题已经解决了。现在显示的是真正的错误：403错误，表示"不支持的国家/地区"。这是SiliconFlow API的地理限制问题，不是代码问题。

# 设置环境变量避免OpenAI验证
os.environ["OPENAI_API_KEY"] = "dummy-key"

# Configure storage path using environment variable
storage_path = os.getenv("CREWAI_STORAGE_DIR", "./storage")

# # 修复APIStatusError初始化问题
# original_init = APIStatusError.__init__

# def patched_init(self, message, *, response=None, body=None):
#     if response is None:
#         # 创建一个完整的模拟的response对象
#         class MockRequest:
#             method = "POST"
#             url = "https://api.mock.com/v1/chat/completions"
            
#         class MockResponse:
#             status_code = 500
#             headers = {}
#             request = MockRequest()
            
#             def json(self):
#                 return {"error": {"message": str(message)}}
                
#             @property
#             def text(self):
#                 return str({"error": {"message": str(message)}})
                
#         response = MockResponse()
    
#     if body is None:
#         body = {"error": {"message": str(message)}}
    
#     original_init(self, message, response=response, body=body)

# APIStatusError.__init__ = patched_init


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
    verbose=True,
    llm=llm
)

analyst = Agent(
    role='数据分析师',
    goal='分析市场趋势',
    backstory="擅长数据处理和分析的专家。",
    verbose=True,
    llm=llm
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
    memory=True,
    long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(
            db_path="{storage_path}/ltm_memory.db".format(storage_path=storage_path)
        )
    ),
    # Short-term memory for current context using RAG
    short_term_memory = ShortTermMemory(
        storage = RAGStorage(
                embedder_config={
                    "provider": "ollama",
                    "config": {
                        "model": 'nomic-embed-text'
                    }
                },
                type="short_term",
                path=f"{storage_path}"
        )
    ),

    # Entity memory for tracking key information about entities
    entity_memory = EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "ollama",
                "config": {
                    "model": 'nomic-embed-text'
                }
            },
            type="short_term",
            path=f"{storage_path}"
        )
    ),
)

# 运行团队
if __name__ == "__main__":
    print("🚀 开始电动汽车市场调研任务...")
    print("📋 使用模型: openai/Qwen/Qwen3-8B via siliconflow")
    print("🔤 嵌入模型: BAAI/bge-large-zh-v1.5")
    print("🔧 APIStatusError已修复")
    
    try:
        result = ev_crew.kickoff()
        print("\n" + "="*50)
        print("✅ 任务完成！")
        print("="*50)
        print(result)
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
