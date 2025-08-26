import os
from crewai import Agent, Task, Crew, Process, LLM
from openai import APIStatusError

# 设置环境变量避免OpenAI验证
os.environ["OPENAI_API_KEY"] = "dummy-key"

# 修复APIStatusError初始化问题
original_init = APIStatusError.__init__

def patched_init(self, message, *, response=None, body=None):
    if response is None:
        # 创建一个完整的模拟的response对象
        class MockRequest:
            method = "POST"
            url = "https://api.mock.com/v1/chat/completions"
            
        class MockResponse:
            status_code = 500
            headers = {}
            request = MockRequest()
            
            def json(self):
                return {"error": {"message": str(message)}}
                
            @property
            def text(self):
                return str({"error": {"message": str(message)}})
                
        response = MockResponse()
    
    if body is None:
        body = {"error": {"message": str(message)}}
    
    original_init(self, message, response=response, body=body)

APIStatusError.__init__ = patched_init

# 测试修复是否有效
try:
    # 尝试创建一个APIStatusError来测试修复
    error = APIStatusError("Test error message")
    print("✅ APIStatusError修复成功！")
    print(f"错误信息: {error}")
except Exception as e:
    print(f"❌ APIStatusError修复失败: {e}")

# 简单的LLM配置测试
llm = LLM(
    model="openai/Qwen/Qwen3-8B",
    provider="siliconflow",
    base_url="https://api.siliconflow.cn/v1",
    api_key="test-key",
    temperature=0,
)

print("✅ LLM配置成功")
print("🔧 修复验证完成")