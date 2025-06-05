from llama_index.core import SimpleDirectoryReader, Document


def read_data(
    input_dir: str = "data",
    recursive: bool = True,
    required_exts: list[str] = [".docx", ".pdf", ".txt"],
) -> list[Document]:
    # 创建一个SimpleDirectoryReader对象，用于读取指定目录下的文件
    reader = SimpleDirectoryReader(
        input_dir=input_dir, recursive=recursive, required_exts=required_exts
    )
    # 返回读取到的数据
    return reader.load_data()


if __name__ == "__main__":
    # 调用read_data函数，读取指定目录下的文件
    input_dir = "/Users/hyjiang/Downloads/tmp_downloads/2025xib_厦门国际银行数创金融杯/初赛/制度文档demo"
    documents = read_data(
        input_dir=input_dir, recursive=True, required_exts=[".docx", ".pdf", ".txt"]
    )
    # 打印读取到的文档数量
    print(f"读取到 {len(documents)} 个文档")
