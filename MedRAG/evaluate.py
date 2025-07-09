#!/usr/bin/env python3

# from huggingface_hub import hf_hub_download
# import huggingface_hub

# # Make hf_hub_download available under the old name

# def cached_download(repo_id: str, filename: str, **kwargs):
#     # adapt any legacy kwargs here (e.g. map url->repo_id/filename)
#     return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)

# # Monkey‐patch for backward compatibility
# huggingface_hub.cached_download = cached_download

import os
import re
import json
import argparse
import statistics
import numpy as np

from MIRAGE.src.utils import QADataset, locate_answer
from src.medrag import MedRAG
import src.config as config

###############################################################################
# Evaluation function (from Mirage benchmark)
###############################################################################
def evaluate(dataset, save_dir, split='test', locate_fun=locate_answer):
    flag = False
    pred = []
    answer_list = ['A', 'B', 'C', 'D', 'yes', 'no', 'maybe']
    answer2idx = {ans: i for i, ans in enumerate(answer_list)}

    for q_idx in range(len(dataset)):
        question_id = dataset.index[q_idx]
        fpath = os.path.join(save_dir, f"{split}_{question_id}.json")
        answers = []
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                contents = json.load(f)
            # assume contents is a list of dicts with key 'answer'
            for it in contents[:1]:
                answers.append(it['answer'])
        except Exception as e:
            print(f'Error reading {fpath}: {e}')
        answers = [ans for ans in answers if ans != 'NA']
        if not answers:
            pred.append(-1)
        else:
            pred.append(answer2idx.get(statistics.mode(answers), -1))

    truth = [answer2idx[item['answer'].strip().upper()] for item in dataset]
    if len(pred) < len(truth):
        truth = truth[:len(pred)]
        flag = True
    acc = (np.array(truth) == np.array(pred)).mean()
    std = np.sqrt(acc * (1 - acc) / len(truth))
    return acc, std, flag

###############################################################################
# Inference using MedRAG for each Mirage question set
###############################################################################
def run_inference_for_dataset(dataset, save_dir, medrag, split='test', k=16):
    os.makedirs(save_dir, exist_ok=True)

    for q_idx in range(len(dataset)):
        question_id = dataset.index[q_idx]
        output_file = os.path.join(save_dir, f"{split}_{question_id}.json")
        if os.path.exists(output_file):
            print(f"Skipping question {question_id} (already predicted).")
            continue

        item = dataset[q_idx]
        question = item['question'].strip()
        options = item.get('options', {})
        print(question, options)

        print(f"Inferencing question {question_id}...")
        answer_raw, snippets, scores = medrag.answer(
            question=question,
            options=options,
            k=k,
        )

        m = re.search(r'"answer_choice"\s*:\s*"?([A-D])', answer_raw, re.IGNORECASE)
        if m:
            answer_choice = m.group(1).upper()
        else:
            y = re.search(r'"answer"\s*:\s*"(yes|no|maybe)"', answer_raw, re.IGNORECASE)
            answer_choice = y.group(1).lower() if y else ''

        # save as list of dict to match Mirage format
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([{'answer': answer_choice}], f, indent=2)
        print(f"Question {question_id} answered: {answer_choice}")

###############################################################################
# Main: iterate datasets, run inference and evaluation
###############################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rag', action='store_true', help='Enable RAG retrieval')
    parser.add_argument('--k', type=int, default=16, help='Number of passages to retrieve')
    parser.add_argument('--results_dir', type=str, default='./predictions', help='Where to store outputs')
    parser.add_argument('--llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='LLM model')
    parser.add_argument('--retriever', type=str, default='MedCPT', help='Retriever name')
    parser.add_argument('--corpus', type=str, default='PubMed', help='Corpus name')
    parser.add_argument('--enhanced', action='store_true', help='Enable the feature (default: False)')
    parser.add_argument('--structured', action='store_true', help='Enable the feature (default: False)')
    parser.add_argument('--steps', type=int, default=None, help='Enable the feature (default: False)')
    parser.add_argument('--reasoning', action='store_true', help='Enable the feature (default: False)')
    parser.add_argument('--rerank', action='store_true', help='Enable the feature (default: False)')

    args = parser.parse_args()
    # evaluate.py
    import src.config
    src.config.reasoning = args.reasoning

    # initialize MedRAG
    # medrag = MedRAG(
    #     llm_name=args.llm,
    #     rag=args.rag,
    #     retriever_name=args.retriever,
    #     corpus_name=args.corpus
    # )

    dataset_names = ['mmlu', 'medqa', 'medmcqa'] #, 'pubmedqa', 'bioasq']
    scores = []

    for name in dataset_names:
        split = 'dev' if name == 'medmcqa' else 'test'
        subdir = f"medrag_k{args.k}_{args.corpus}{'_rag' if args.rag else ''}{'_'+args.llm}_baseline{'_structured' if args.structured else ''}{'_enhanced' if args.enhanced else ''}{'_reasoning' if args.reasoning else ''}{'_steps'+str(args.steps) if args.steps else ''}{'_rerank' if args.rerank else ''}"
        base = os.path.join(args.results_dir, name, subdir)
        print(f"Processing dataset {name} ({split}) into {base}")

        # load dataset
        dataset = QADataset(name, dir='MIRAGE')
        # run inference
        run_inference_for_dataset(dataset, base, medrag, split=split, k=args.k)
        # evaluate
        if os.path.exists(base):
            acc, std, flag = evaluate(dataset, base, split)
            scores.append(acc)
            print(f"[{name}] acc={acc:.4f} ±{std:.4f}{' (incomplete)' if flag else ''}")
        else:
            print(f"[{name}] No predictions found at {base}, skipping eval.")

    if scores:
        print(f"Average accuracy: {np.mean(scores):.4f}")

if __name__ == '__main__':
    main()
