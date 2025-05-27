# pip install llama-index-llms-modelscopepip install llama-index-llms-modelscope
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

os.environ["OPENAI_API_KEY"] = "860f7bb0-8efc-4b48-8498-88ec0ef8a60e"
os.environ["OPENAI_API_BASE"] = "https://api-inference.modelscope.cn/v1/"


# 所调用的模型
Settings.llm = OpenAILike(
    model="Qwen/Qwen3-8B",
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    is_chat_model=True,
    temperature=0.1,
    additional_kwargs={
        "extra_body": {"enable_thinking": False},
    },
)
# bge-m3 embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")


documents = SimpleDirectoryReader(
    "/Users/hyjiang/Downloads/tmp_downloads/2025xib_厦门国际银行数创金融杯/初赛/制度文档demo"
).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("如何支持小微企业和“三农”发展？")
print(response)
