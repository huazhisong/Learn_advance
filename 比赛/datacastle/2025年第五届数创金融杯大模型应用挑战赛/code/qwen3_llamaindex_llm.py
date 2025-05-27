import openai
import os
from llama_index.llms.openai_like import OpenAILike

os.environ["OPENAI_API_KEY"] = "860f7bb0-8efc-4b48-8498-88ec0ef8a60e"
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api-inference.modelscope.cn/v1/"
openai.api_key = os.environ["OPENAI_API_BASE"]


# 所调用的模型
llm = OpenAILike(
    model="Qwen/Qwen3-8B",
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    is_chat_model=True,
    temperature=0.1,
    additional_kwargs={
        "extra_body": {"enable_thinking": True},
    },
)

response = llm.stream_complete("帮我推荐一下江浙沪5天的旅游攻略。")
print(response)
