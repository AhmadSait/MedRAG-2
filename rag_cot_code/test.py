import os
import json
import sys
import re
import networkx as nx
from openai import OpenAI

# New function to load and initialize the knowledge graph
def initialize_knowledge_graph(graph_path):
    """Load the knowledge graph from the GraphML file"""
    print(f"Loading knowledge graph from {graph_path}...")
    G = nx.read_graphml(graph_path)
    print(f"Knowledge graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

# New function to query the knowledge graph based on the question
def retrieve_knowledge_from_graph(G, question, top_k=5):
    """
    Retrieve relevant information from the knowledge graph based on the question
    
    Args:
        G: NetworkX graph
        question: The medical question
        top_k: Number of relevant knowledge pieces to retrieve
    
    Returns:
        A string containing relevant knowledge
    """
    # Extract key terms from the question
    # In a real implementation, you'd use NLP techniques like NER, keyword extraction, etc.
    # Here we'll use a simple approach with common words filtering
    
    # Convert question to lowercase and tokenize
    stop_words = {'a', 'an', 'the', 'and', 'or', 'is', 'are', 'of', 'to', 'in', 'with', 'for'}
    question_lower = question.lower()
    words = re.findall(r'\b\w+\b', question_lower)
    key_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Find relevant nodes in the graph
    relevant_nodes = []
    for node, data in G.nodes(data=True):
        node_text = str(node).lower()
        # if 'description' in data:
        #     node_text += ' ' + data['description'].lower()
            
        # Calculate a simple relevance score based on term matching
        score = sum(1 for term in key_terms if term in node_text)
        if score > 0:
            relevant_nodes.append((node, score, data.get('description', '')))
    
    # Sort by relevance score and take top_k
    relevant_nodes.sort(key=lambda x: x[1], reverse=True)
    top_nodes = relevant_nodes[:top_k]
    
    # Get edges (relationships) for the top nodes
    relevant_knowledge = []
    for node, score, desc in top_nodes:
        # Add node information
        node_info = f"Entity: {node} - {desc}" if desc else f"Entity: {node}"
        relevant_knowledge.append(node_info)
        
        # Add edge information
        for neighbor in G.neighbors(node):
            edge_data = G.edges[node, neighbor]
            relation = edge_data.get('description', 'related to')
            neighbor_desc = G.nodes[neighbor].get('description', '')
            
            if neighbor_desc:
                edge_info = f"- {node} {relation} {neighbor} ({neighbor_desc})"
            else:
                edge_info = f"- {node} {relation} {neighbor}"
            relevant_knowledge.append(edge_info)
    
    # Convert the list to a formatted string
    return "\n".join(relevant_knowledge)

# Modified function to include retrieved knowledge in the prompt
def submit_vqa_to_llm(client, qa, knowledge_graph=None):
    # Retrieve relevant knowledge if a graph is provided
    retrieved_knowledge = ""
    if knowledge_graph is not None:
        # Extract the question text from the qa dictionary
        question_text = ""
        if isinstance(qa, dict):
            question_text = qa.get("question", "")
        elif isinstance(qa, str):
            # Try to extract question from a string representation of a dict
            match = re.search(r"'question':\s*'([^']*)'", qa)
            if match:
                question_text = match.group(1)
        
        # Retrieve knowledge based on the question
        if question_text:
            # Limit retrieved knowledge to 1024 words
            retrieved_knowledge = retrieve_knowledge_from_graph(knowledge_graph, question_text)
            retrieved_knowledge = ' '.join(retrieved_knowledge.split()[:1024*20])
            print("-=========================================================")
            print("**********************************************************")
            print(f"Retrieved knowledge: {retrieved_knowledge}")
            print("**********************************************************")            
            # print(f"Retrieved {len(retrieved_knowledge.split('\n'))} knowledge items")

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
        Here is some relevant knowledge that might help you answer:\n + {retrieved_knowledge}
    """ 
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    completion = client.completions.create(model="Qwen/Qwen2.5-7B-Instruct",
                                      prompt=prompt,max_tokens=4906)
    print("**********************************************************")
    print(f"Completion result: {completion.choices[0].text}")
    print("=========================================================")
    return completion.choices[0].text



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
    save_path = f"./results/kg_rag_qa_last_20"  # Changed the path to reflect KG-RAG approach
    os.makedirs(save_path, exist_ok=True)
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    knowledge_graph_path = "/home/plusai/Documents/github/LightRAG/examples/dickens/graph_chunk_entity_relation.graphml"
    knowledge_graph = initialize_knowledge_graph(knowledge_graph_path)
    
    save_file_path = os.path.join(save_path, f"{os.path.basename(__file__).split('.')[0]}_results.json")
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = {}

    qa_file_path = "/home/plusai/Documents/github/LightRAG/cs394/selected_data.json"
    # qa_file_path = sys.argv[1]
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
                    print(f"skip {key_name}")
                    continue
                
                unique_qa_data = dataset_qa_data[key].copy()
                correct_answer = unique_qa_data.pop("answer")
                letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
                correct_answer_index = letter_to_number[correct_answer]
                
                # Pass the knowledge graph to the submit_vqa_to_qwen function
                answer = submit_vqa_to_llm(client, str(unique_qa_data), knowledge_graph)

                correctness, extracted_index = handle_vqa_result(answer, correct_answer_index)
            
                model_results[key_name] = {
                    "answer": extracted_index, 
                    "correctness": correctness
                }
            
                with open(save_file_path, 'w') as f:
                    json.dump(model_results, f, indent=4)
                    
            except Exception as e:
                print(f"Error processing {key_name}: {e}")
                model_results[key_name] = {
                    "answer": -1,
                    "correctness": -1
                }