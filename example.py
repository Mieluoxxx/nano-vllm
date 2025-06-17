import os
import torch.distributed as dist
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


path = os.path.expanduser("Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True)

sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)
prompts = [
    "introduce yourself",
    "list all prime numbers within 10",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # 保留思考过程输出
    )
    for prompt in prompts
]
outputs = llm.generate(prompts, sampling_params)

# 优化格式：分离思考过程与最终回答
for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
    print(f"\n===== 示例 {i} =====")
    # 提取并清理提示文本
    clean_prompt = prompt.split('<|im_end|>')[0].replace('<|im_start|>user\n', '').strip()
    print(f"提示: {clean_prompt}")
    
    # 分离思考过程与回答内容
    content = output['text'].split('</think>')
    if len(content) >= 3:
        thought_process = content[1].strip()
        final_answer = content[2].replace('<|im_end|>', '').strip()
        print("\n思考过程:")
        print(f"{thought_process}")
        print("\n回答:")
        print(f"{final_answer}")
    else:
        # 兼容无思考过程的情况
        print("\n回答:")
        print(f"{output['text'].replace('<|im_end|>', '').strip()}")

# 显式销毁分布式进程组
if dist.is_initialized():
    dist.destroy_process_group()
