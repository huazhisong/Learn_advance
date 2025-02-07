from llama_index.core.objects import SQLTableNodeMapping
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core.llms import OpenAI
from llama_index.core.query_pipeline import QueryPipeline

# 创建数据库连接
engine = create_engine("sqlite:///test.db")
metadata = MetaData()

# 创建表结构
table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("age", Integer)
)
metadata.create_all(engine)

# 创建数据库对象
sql_database = SQLDatabase(engine)

# 创建对象索引
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [SQLTableSchema(table_name="users", context_str="A table of users and their ages")]

obj_index = ObjectIndex.from(table_schema_objs, table_node_mapping, VectorStoreIndex)

# 创建查询管道
llm = OpenAI(model="gpt-3.5-turbo")
text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query:query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str,)

# 创建查询管道，不执行 SQL 查询
qp = QueryPipeline(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_index.as_retrie(similarity_top_k=3),
        "table_output_parser": FnComponent(fn=lambda x: "\n".join([f"Table '{table_name}' has columns: {', '.join(columns)}" for table_name, columns in x])),
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": FnComponent(fn=lambda x: x        "sql_retriever": SQLRetriever(sql_database),
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
    execute_sql=False  # 设置此选项以避免执行 SQL 查询
)

# 运行查询
response = qp.run(query="Show all users' names and ages.")
print(response)