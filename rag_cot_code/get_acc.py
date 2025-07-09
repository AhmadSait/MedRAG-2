import json
import os
import pandas as pd
if __name__ == "__main__":
    file = "/home/plusai/Documents/github/LightRAG/cs394/rag_cot_code/results/kg_rag_qa_first_20/test_results.json"
    with open(file, 'r') as f:
        json_data = json.load(f)
    
    used_keys = list(json_data.keys())
    
    folder_path = "/home/plusai/Documents/github/LightRAG/cs394/cot_code/results/baseline"
    files = os.listdir(folder_path)
    files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            
        dataset_acc = {}
        dataset_names = list(set([key.split('_')[0] for key in json_data.keys()]))
        for dataset_name in dataset_names:
            dataset_json_data = {key: json_data[key] for key in json_data.keys() if key.split('_')[0] == dataset_name}
            dataset_json_data_keys = dataset_json_data.keys()
            acc_ant = 0
            cnt = 0
            for key in dataset_json_data_keys:
                if key in used_keys:
                    cnt = cnt + 1
                    if dataset_json_data[key]["correctness"] == 1:
                        acc_ant = acc_ant + 1
            dataset_acc[dataset_name] = {}  # Initialize the dictionary for this dataset
            dataset_acc[dataset_name]["acc"] = float(acc_ant/cnt)
            dataset_acc[dataset_name]["cnt"] = cnt

        print("================================================")
        print(f"{file} \n")
        for dataset_name in dataset_acc.keys():
            print(f"{dataset_name}: {dataset_acc[dataset_name]['acc']} ({dataset_acc[dataset_name]['cnt']})")