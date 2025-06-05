from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# global default
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
)
Settings.llm = OpenAILike(
    model="Qwen/Qwen3-8B",
    api_base="https://api-inference.modelscope.cn/v1/",
    api_key="860f7bb0-8efc-4b48-8498-88ec0ef8a60e",
    is_chat_model=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    additional_kwargs={
        "extra_body": {"enable_thinking": False},
    },
)

documents = SimpleDirectoryReader(
    "/Users/hyjiang/Downloads/tmp_downloads/2025xib_厦门国际银行数创金融杯/初赛/制度文档demo"
).load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query(
    "如何区分国有商业银行分行与城市商业银行分行在高管任职资格上的差异？"
)
print(response)
