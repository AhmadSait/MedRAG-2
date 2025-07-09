
# from vllm import LLM, SamplingParams

# # 加载本地路径的模型
# llm = LLM(model="meta-llama/Meta-Llama-3-8B", download_dir="/ibex/project/c2191/wenxuan_proj/huggingface_model_weight")

# # 设置采样参数
# sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

# # 生成文本
# prompt = "What are the benefits of using multimodal learning models?"
# outputs = llm.generate(prompt, sampling_params)

# # 打印生成结果
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
# from modelscope import snapshot_download

# model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', local_dir='/ibex/project/c2191/wenxuan_proj/huggingface_model_weight/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', revision='master')
# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE']='True'

def get_completion(prompts, model, tokenizer=None, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True, dtype="float16", download_dir="/ibex/project/c2191/wenxuan_proj/huggingface_model_weight")  # V100 不支持bfloat16 类型，需要设置为 float16
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":  
    # 初始化 vLLM 推理引擎
    model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' # 指定模型路径
    tokenizer = None
  
    text = ["请帮我制定个简短的初学者Python学习计划<think>\n", ] # 可用 List 同时传入多个 prompt，根据 DeepSeek 官方的建议，每个 prompt 都需要以 <think>\n 结尾，如果是数学推理内容，建议包含（中英文皆可）：Please reason step by step, and put your final answer within \boxed{}.
    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048) # 思考需要输出更多的 Token 数，max_tokens 设为 8K，根据 DeepSeek 官方的建议，temperature应在 0.5-0.7，推荐 0.6
    print(outputs[0].outputs[0].text)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if r"</think>" in generated_text:
            think_content, answer_content = generated_text.split(r"</think>")
        else:
            think_content = ""
            answer_content = generated_text
        print(f"Prompt: {prompt!r}, Think: {think_content!r}, Answer: {answer_content!r}")
