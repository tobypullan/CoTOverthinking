import os
from groq import Groq
from datasets import load_dataset
import json

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def answer_parser(response):
    if response[-1] == ".":
        response = response[:-1]
    final_sentence = response.strip().split(".")[-1]
    final_section = response[-10:]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for char in final_section:
        if char in labels:
            answer = char
    return answer

def model_answer_parser(response):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You must parse the following response and extract the letter label of the answer option chosen by the model. The answer will be one of the following letters: A, B, C, D, E, F, G, H, I, J. The answer will be at near the end of the response. Return only the letter label of the answer and nothing else. For example, if the answer is option A, return 'A' and nothing else.",
            },
            {
                "role": "user",
                "content": response[-100:] if len(response) > 100 else response[-10:] ,
            }
        ],
        model="llama-3.1-8b-instant",
    )
    return response.choices[0].message.content.strip()

def baseline_CoT(q_id):
    """
    returns the baseline CoT for a given question id, prompt_tokens, completion tokens, correctness
    """
    question = ds[q_id]["question"]
    options = ds[q_id]["options"]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    prompt = question + " The options are: " + " ".join(["\n" +labels[i] + ": " + options[i] for i in range(len(options))])
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Make sure to finish your answers to multiple choice questions with the letter label of the answer option you chose, with no following punctuation. For example, if you choose option A, end your answer with 'A' and nothing else."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="qwen/qwen3-32b",
    )
    print(response)
    answer = model_answer_parser(response.choices[0].message.content)
    correct = answer == ds[q_id]["answer"]
    completed = response.choices[0].finish_reason
    return prompt, response.choices[0].message.content, correct, completed

def save_baseline(q_id):
    prompt, response, correct, completed = baseline_CoT(q_id)
    with open("baseline_CoTs.jsonl", "a") as f:
        json.dump({
            "question_id": q_id,
            "prompt": prompt,
            "response": response,
            "correct": correct,
            "actual_answer": ds[q_id]["answer"],
            "complete_reason": completed
        }, f)
        f.write("\n")
    return correct

for i in range(100,1000):
    print(f"Processing question {i}...")
    correct = save_baseline(i)
    print(f"Question {i} processed. Correct: {correct}")
