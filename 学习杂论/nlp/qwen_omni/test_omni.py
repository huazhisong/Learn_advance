from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniConfig,
)

config = Qwen2_5OmniConfig.from_pretrained(
    "/Users/hyjiang/song_ws/models/Qwen/Qwen2.5-Omni-7B"
)

print(config)
