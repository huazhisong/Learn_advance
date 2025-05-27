from jinja2 import Template
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    get_response_synthesizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI


client = OpenAI(
    api_key="860f7bb0-8efc-4b48-8498-88ec0ef8a60e",
    base_url="https://api-inference.modelscope.cn/v1/",
)


class choice(BaseModel):
    """正确的选项"""

    correct_chice: str


prompt_choice = """
{{question}}
{{choice}}
"""
choice_template = Template(prompt_choice)

prompt_jianda = """
{{question}}
"""
jianda_template = Template(prompt_jianda)


extract_answer = """
你是一个答案提取助手，请直接从我给出的包含答案选项和解析的文本中提取出正确答案选项,只返回选项对应的字母，不返回任何其他内容。

文本陈述:
{{chenshu}}
"""

jian_template = Template(extract_answer)


Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
)

documents = SimpleDirectoryReader(
    "/Users/hyjiang/Downloads/tmp_downloads/2025xib_厦门国际银行数创金融杯/初赛/制度文档demo",
    recursive=True,
).load_data(show_progress=True)

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

index = VectorStoreIndex.from_documents(documents)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize"
    # output_cls=choice
)

# 修改text_qa_template
new_choice_tmpl_str = (
    "你是一个兼容监管制度知识专家。下面是上下文信息。\n"
    "{context_str}\n"
    "你将会再查询接受到一个金融监管制度方面不定项选择题和多个选项。请根据上下文信息和非先验知识, 在给出的选项中选出正确的答案（可能是单个或多个选项)"
    "查询：{query_str}\n"
    "答案:"
)
new_choice_tmpl = PromptTemplate(new_choice_tmpl_str)

new_jianda_tmpl_str = (
    "你是一个兼容监管制度知识专家。下面是上下文信息。\n"
    "{context_str}\n"
    "你将会再查询接受到一个金融监管制度方面简答题。请根据上下文信息和非先验知识, 对问题做出正确的解答"
    "查询：{query_str}\n"
    "答案:"
)
new_jianda_tmpl = PromptTemplate(new_jianda_tmpl_str)

# query_engine = index.as_query_engine(
#         response_mode="tree_summarize",
#         output_cls=choice
# )

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
)

data = pd.read_json(
    "/Users/hyjiang/Downloads/tmp_downloads/2025xib_厦门国际银行数创金融杯/初赛/数据集demo/demo_train.json",
    lines=True,
)
results = {"id": [], "answer": []}
for idx, row in data.iterrows():
    print("=====================================")
    if row["category"] == "选择题":
        query_engine.update_prompts(
            {"response_synthesizer:summary_template": new_choice_tmpl}
        )
        prompt = choice_template.render(question=row["question"], choice=row["content"])
        response = query_engine.query(prompt)
        print(response)
        chenshu = jian_template.render(chenshu=response)
        message = [
            # {'role':"system","content":sys},
            {"role": "user", "content": chenshu}
        ]
        result = client.chat.completions.create(
            messages=message,
            model="Qwen/Qwen3-8B",
            temperature=0.1,
            top_p=0.8,
            max_tokens=1000,
            extra_body={"repetition_penalty": 1.05, "enable_thinking": False},
        )
        print(result.choices[0].message.content)
        result = [result.choices[0].message.content.strip()]
        results["id"].append(row["id"])
        results["answer"].append(result)
    else:
        print("============================================")
        query_engine.update_prompts(
            {"response_synthesizer:summary_template": new_jianda_tmpl}
        )
        prompt = jianda_template.render(question=row["question"])
        response = query_engine.query(prompt)
        print(response)
        results["id"].append(row["id"])
        results["answer"].append(result)
