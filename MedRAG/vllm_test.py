from enum import Enum
from pydantic import BaseModel
from openai import OpenAI

# Step 1: Define your schema
class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

class MedRAgOutput(BaseModel):
    step_by_step_thinking: str
    answer_choice: AnswerChoice

json_schema = MedRAgOutput.model_json_schema()

# Step 2: Setup local OpenAI client for vLLM
client = OpenAI(
    base_url="http://localhost:8003/v1",
    api_key="EMPTY"  # No key needed for local
)

# Step 3: Create message prompt
messages = [
    {"role": "system", "content": "You are a medical assistant that must return output in JSON format."},
    {"role": "user", "content": "Patient has a headache, fever, and rash. Which is the most likely diagnosis?\nA. Common Cold\nB. Flu\nC. Measles\nD. Malaria"}
]

# Step 4: Send request with guided JSON schema
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=messages,
    max_tokens=500,
    seed=42,
    extra_body={"guided_json": json_schema}
)

# Step 5: Print result
print("Raw response:", response)
print("\nGenerated JSON:")
print(response.choices[0].message.reasoning_content)