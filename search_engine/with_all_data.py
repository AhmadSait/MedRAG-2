import os
import json
import sys
import re
from openai import OpenAI
from colorama import init, Fore, Style

# Initialize colorama
init()

# Modified function to include retrieved knowledge in the prompt
def submit_vqa_to_llm(client, qa, retrieved_knowledge=None):
    # Create the prompt with the retrieved knowledge
    prompt = f"""You are a helpful medical expert, and your task is to answer a multi-choice medical question. 
        Please first think step-by-step and then choose the answer from the provided options.
        Your responses will be used for research purposes only, so please have a definite answer.
        Here is the question and options: {qa}
        You must strictly return in the following json format without any other text:
        {{
            "step_by_step_thinking": xxx,
            "answer_choice": xxx,
        }}
        Here is some relevant knowledge that might help you answer:
        {retrieved_knowledge}.
    """ 
    messages = [
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct",
                                      messages=messages, max_tokens=4906)
    
    print(f"{Fore.CYAN}Completion result:\n{Style.RESET_ALL}{completion.choices[0].message.content}")
    return completion.choices[0].message.content

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
        print(f"{Fore.RED}Invalid input: prediction answer is empty or not a string{Style.RESET_ALL}")
        return -2, -1

    extracted_option = extract_answer_option(pred_answer)
    if not extracted_option:
        print(f"{Fore.RED}Invalid response format. No valid option (A/B/C/D) found{Style.RESET_ALL}")
        return -1, -1

    letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    pred_number = letter_to_number[extracted_option]
    
    if pred_number == correct_answer_index:
        print(f"{Fore.GREEN}Correct answer{Style.RESET_ALL}")
        return 1, pred_number
    else:
        print(f"{Fore.RED}Wrong answer{Style.RESET_ALL}")
        return 0, pred_number

if __name__ == "__main__":
    save_path = f"./results/with_all_data"  # Changed the path to reflect KG-RAG approach
    os.makedirs(save_path, exist_ok=True)
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    engine_data_file_path = "/home/plusai/Documents/github/LightRAG/cs394/search_results.json"
    with open(engine_data_file_path, 'r') as f:
        engine_data = json.load(f)
    engine_data_keys = engine_data.keys()
    
    save_file_path = os.path.join(save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}

    qa_file_path = "/home/plusai/Documents/github/LightRAG/cs394/selected_data.json"
    with open(qa_file_path, 'r') as f:
        qa_data = json.load(f)
    
    dataset_names = qa_data.keys()
    for dataset_name in dataset_names:
        dataset_qa_data = qa_data[dataset_name]
        dataset_qa_data_keys = dataset_qa_data.keys()
        for key in dataset_qa_data_keys:
            try:
                key_name = f"{dataset_name}_{key}"
                if key_name in model_results:
                    print(f"{Fore.YELLOW}skip {key_name}{Style.RESET_ALL}")
                    continue
                
                unique_qa_data = dataset_qa_data[key].copy()
                correct_answer = unique_qa_data.pop("answer")
                letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
                correct_answer_index = letter_to_number[correct_answer]
                
                # retrieve relevant knowledge
                retrieved_knowledge = ""
                cnt = 0
                for engine_data_key in engine_data_keys:
                    if engine_data_key.startswith(key_name):
                        cnt += 1
                        retrieved_knowledge += f"relevant knowledge {cnt}:\n"
                        retrieved_knowledge += engine_data[engine_data_key]["content"]
                        retrieved_knowledge += "\n"
                print(f"{Fore.CYAN}====================================================={Style.RESET_ALL}")
                print(f"{Fore.CYAN}retrieved_knowledge: \n{Style.RESET_ALL}{retrieved_knowledge}")
                answer = submit_vqa_to_llm(client, str(unique_qa_data), retrieved_knowledge)

                correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
            
                model_results[key_name] = {
                    "answer": extracted_index, 
                    "correctness": correctness
                }
            
                with open(save_file_path, 'w') as f:
                    json.dump(model_results, f, indent=4)
                    
            except Exception as e:
                print(f"{Fore.RED}====================================================={Style.RESET_ALL}")
                print(f"{Fore.RED}Error processing {key_name}: {e}{Style.RESET_ALL}")
                model_results[key_name] = {
                    "answer": -1,
                    "correctness": -1
                }
                exit()