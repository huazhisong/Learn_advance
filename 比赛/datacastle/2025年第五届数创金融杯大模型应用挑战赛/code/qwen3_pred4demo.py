from openai import OpenAI
import pandas as pd

base_url = "https://api-inference.modelscope.cn/v1/"
api_key = "860f7bb0-8efc-4b48-8498-88ec0ef8a60e"
model = "Qwen/Qwen3-32B"  # ModelScope Model-Id


def ask_model_question(
    question: str, enable_thinking: bool = False, stream: bool = False
):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    # set extra_body for thinking control
    extra_body = {
        # enable thinking, set to False to disable
        "enable_thinking": enable_thinking,
        # use thinking_budget to contorl num of tokens used for thinking
        # "thinking_budget": 4096
    }
    response = client.chat.completions.create(
        model=model,  # ModelScope Model-Id
        messages=[{"role": "user", "content": question}],
        stream=stream,
        extra_body=extra_body,
    )

    thinking_process = ""
    final_answer = ""

    if enable_thinking or stream:
        done_thinking = False
        for chunk in response:
            thinking_chunk = chunk.choices[0].delta.reasoning_content
            answer_chunk = chunk.choices[0].delta.content
            if thinking_chunk != "":
                thinking_process += thinking_chunk
                # print(thinking_chunk, end="", flush=True)
            elif answer_chunk != "":
                if not done_thinking:
                    # print("\n\n === Final Answer ===\n")
                    done_thinking = True
                final_answer += answer_chunk
                # print(answer_chunk, end="", flush=True)
    else:
        thinking_process = response.choices[0].message.reasoning_content
        # final answer
        final_answer = response.choices[0].message.content

    return thinking_process, final_answer


def test_demo():
    question = "请帮我写一句笑话"
    thinking_process, final_answer = ask_model_question(
        question=question,
        enable_thinking=False,
        stream=False,
    )
    print("\n\n === Reasoning think ===\n")
    print(thinking_process)
    print("\n\n === Final Answer ===\n")
    print(final_answer)


def main():
    filepath = "/Users/hyjiang/Downloads/tmp_downloads/2025xib_厦门国际银行数创金融杯/初赛/数据集demo/demo_train.json"
    train_df = pd.read_json(filepath, lines=True)
    filepath = "/Users/hyjiang/Downloads/tmp_downloads/2025xib_厦门国际银行数创金融杯/初赛/数据集demo/demo_train_answer.json"
    answer_df = pd.read_json(filepath, lines=True)
    for index, row in train_df.iterrows():
        category = row["category"]
        question = row["question"]
        content = row["content"]
        question = f"""
        你是一个金融监管制度智能问答助手，请你根据问题进行思考，给出合理的答案。
        # 背景
        基于大模型的文档问答，根据输入的问题（如“个人理财产品的销售需满足哪些监管要求？”），基于给定的金融文档库，生成准确、合规的答案。题型包含不定项选择题和问答题。
        # 输出
          - 如果问题是选择题，输出格式为：["A", "B", "C"]，其中A、B、C为选项内容。
          - 如果问题是问答题，直接输出问题的答案即可。
          - 请只输出答案，不要输出其他内容。
        # 问题如下：
            问题分类: {category}
            问题: {question}
            {"答案选项：" + content if category == "选择题" else ""}
        """

        print(f"Question: {question}")
        # 1. 提问
        _, final_answer = ask_model_question(
            question=question,
            enable_thinking=True,
            stream=True,
        )
        # 2. 打印答案
        print("\n\n === Final Answer ===\n")
        print(final_answer)
        # 2. 对比答案
        answer = answer_df.iloc[index]["answer"]
        print(f"答案: {answer}")
        print("===" * 20)


if __name__ == "__main__":
    # test_demo()
    main()
