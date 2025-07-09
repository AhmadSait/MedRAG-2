import os
import json
import transformers
import torch
import sys
import re


def submit_vqa_to_llama(pipeline, qa):
    prompt = f"""You are a helpful medical expert, and your task is to answer a multi-choice medical question. 
        Please first think step-by-step and then choose the answer from the provided options. 
        You must strictly return in the following json format without any other text:
        {{
            "step_by_step_thinking": xxx,
            "answer_choice": xxx,
        }}
        Your responses will be used for research purposes only, so please have a definite answer.
        Here is the question and options: {qa}."""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0]["generated_text"][-1]
    return response["content"]


def extract_answer_option(text):
    idx = text.find('"answer_choice"')
    text = text[idx+16:]
    pattern = r'"([A-D])"'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0]

    
    return None


def handle_vqa_result(pred_answer, correct_answer_index):
    if pred_answer is None or not isinstance(pred_answer, str):
        print("Invalid input: prediction answer is empty or not a string")
        return -2, -1

    extracted_option = extract_answer_option(pred_answer)
    if not extracted_option:
        print("Invalid response format. No valid option (A/B/C/D) found")
        return -1, -1

    letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    pred_number = letter_to_number[extracted_option]
    
    if pred_number == correct_answer_index:
        print("Correct answer")
        return 1, pred_number
    else:
        print("Wrong answer")
        return 0, pred_number

if __name__ == "__main__":
    save_path = f"./results/baseline"
    os.makedirs(save_path, exist_ok=True)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": "/ibex/project/c2191/wenxuan_proj/huggingface_model_weight"
        },
        device_map="auto"
    )
    
    save_file_path = os.path.join(save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}
    qa_file_path = sys.argv[1]
    with open(qa_file_path, 'r') as f:
        qa_data = json.load(f)
    
    dataset_names = qa_data.keys()
    for dataset_name in dataset_names:
        dataset_qa_data = qa_data[dataset_name]
        dataset_qa_data_keys = dataset_qa_data.keys()
        for key in dataset_qa_data_keys:
            key_name = f"{dataset_name}_{key}"
            if key_name in model_results:
                print(f"skip {key_name}")
                continue
            
            unique_qa_data = dataset_qa_data[key].copy()
            correct_answer =  unique_qa_data.pop("answer")
            letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            correct_answer_index = letter_to_number[correct_answer]
            answer = submit_vqa_to_llama(pipeline, str(unique_qa_data))

            correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
        
            model_results[key_name] = {"answer": extracted_index, "correctness": correctness}
        
            with open(save_file_path, 'w') as f:
                json.dump(model_results, f, indent=4)