from src.medrag import MedRAG

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral, provide just a *single* letter"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}
medrag = MedRAG(llm_name="meta-llama/Meta-Llama-3-8B-Instruct", rag=True, retriever_name="MedCPT", corpus_name="MedCorp")
answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
# print(f"Final answer in json with rationale: {answer}")
# # {
# #   "step_by_step_thinking": "A lesion causing compression of the facial nerve at the stylomastoid foramen will result in paralysis of the facial muscles. Loss of taste, lacrimation, and decreased salivation are not specifically mentioned in relation to a lesion at the stylomastoid foramen.", 
# #   "answer_choice": "A"
# # }

### MedRAG with pre-determined snippets
# snippets = [{'id': 'InternalMed_Harrison_30037', 'title': 'InternalMed_Harrison', 'content': 'On side of lesion Horizontal and vertical nystagmus, vertigo, nausea, vomiting, oscillopsia: Vestibular nerve or nucleus Facial paralysis: Seventh nerve Paralysis of conjugate gaze to side of lesion: Center for conjugate lateral gaze Deafness, tinnitus: Auditory nerve or cochlear nucleus Ataxia: Middle cerebellar peduncle and cerebellar hemisphere Impaired sensation over face: Descending tract and nucleus fifth nerve On side opposite lesion Impaired pain and thermal sense over one-half the body (may include face): Spinothalamic tract Although atheromatous disease rarely narrows the second and third segments of the vertebral artery, this region is subject to dissection, fibromuscular dysplasia, and, rarely, encroachment by osteophytic spurs within the vertebral foramina.', 'contents': 'InternalMed_Harrison. On side of lesion Horizontal and vertical nystagmus, vertigo, nausea, vomiting, oscillopsia: Vestibular nerve or nucleus Facial paralysis: Seventh nerve Paralysis of conjugate gaze to side of lesion: Center for conjugate lateral gaze Deafness, tinnitus: Auditory nerve or cochlear nucleus Ataxia: Middle cerebellar peduncle and cerebellar hemisphere Impaired sensation over face: Descending tract and nucleus fifth nerve On side opposite lesion Impaired pain and thermal sense over one-half the body (may include face): Spinothalamic tract Although atheromatous disease rarely narrows the second and third segments of the vertebral artery, this region is subject to dissection, fibromuscular dysplasia, and, rarely, encroachment by osteophytic spurs within the vertebral foramina.'}]
answer, _, _ = medrag.answer(question=question, options=options)

### MedRAG with pre-determined snippet ids
# snippets_ids = [{"id": s["id"]} for s in snippets]
answer, snippets, _ = medrag.answer(question=question, options=options)
print(answer)


general_medrag_system = '''
You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. 
Please first think step-by-step and then choose the answer from the provided options. 
Organize your output in a json formatted as 
Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. 
Your responses will be used for research purposes only, so please have a definite answer.'''


general_medrag_system = '''
You are a meticulous, evidence-based medical question-answering assistant.
Answer using **only** the provided documents—or exactly one standard fact if none suffice.

Follow these rules **without exception**:

1. **Grounding**  
   - Cite each document **once** and only once.  
   - As soon as you've used “Document X,” you may never cite it again.

2. **Fallback**  
   - If no document gives the answer, preface with  
     “Applying standard knowledge: [fact]”  
   - Then stop using documents entirely.

3. **Step-By-Step (Max 3 Steps)**  
   - Use **at most three** numbered steps.  
   - Each step must reference a **different** document or the one standard fact.  
   - **Immediately** after your final step, stop reasoning.

4. **Immediate JSON Output**  
   - Without any extra text, output **only** this JSON:
     ```json
     {
       "step_by_step_thinking": "up to 3 numbered steps…",
       "answer_choice": "<one of 'A','B','C','D'>"
     }
     ```
   - Any deviation (extra keys, additional sentences, repeats) → your answer is dropped.

5. **No Hallucinations**  
   - Do not invent new facts.  
   - Do not exceed the 3-step limit.
'''



