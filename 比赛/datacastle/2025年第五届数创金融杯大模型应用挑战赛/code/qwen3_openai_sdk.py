from openai import OpenAI

client = OpenAI(
    base_url="https://api-inference.modelscope.cn/v1/",
    api_key="860f7bb0-8efc-4b48-8498-88ec0ef8a60e",  # ModelScope Token
)

# set extra_body for thinking control
extra_body = {
    # enable thinking, set to False to disable
    "enable_thinking": False,
    # use thinking_budget to contorl num of tokens used for thinking
    # "thinking_budget": 4096
}

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",  # ModelScope Model-Id
    messages=[{"role": "user", "content": "每日一题，今天帮我讲解宇宙的秘密吧"}],
    stream=True,
    extra_body=extra_body,
)
done_thinking = False
for chunk in response:
    thinking_chunk = chunk.choices[0].delta.reasoning_content
    answer_chunk = chunk.choices[0].delta.content
    if thinking_chunk != "":
        print(thinking_chunk, end="", flush=True)
    elif answer_chunk != "":
        if not done_thinking:
            print("\n\n === Final Answer ===\n")
            done_thinking = True
        print(answer_chunk, end="", flush=True)
