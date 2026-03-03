import os
from groq import Groq
from datasets import load_dataset
import json

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def model_judge(response, options):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are given a model's response to a question. You must decide which option their answer best matches. Let me be clear - you are not choosing the option YOU think is correct, you are choosing the answer that best fits the RESPONSE you are given.",
            },
            {
                "role": "user",
                "content": f"response: {response}\n\n options: {options}\n\n Explain why you have chosen the option that best fits the response you have. End your explanation with the label of the option you have chosen.",
            }
        ],
        model="openai/gpt-oss-20b",
    )
    return response.choices[0].message.content.strip()

def answer_parser(model_response):
    try:
        if model_response[-1] == ".":
            model_response = model_response[:-1]
        final_sentence = model_response.strip().split(".")[-1]
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for char in final_sentence:
            if char in labels:
                answer = char
        return answer
    except:
        return None
    
def baseline_CoT(q_id):
    """
    returns the baseline CoT for a given question id, prompt_tokens, completion tokens, correctness
    """
    question = ds[q_id]["question"]
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        model="qwen/qwen3-32b",
    )
    print(response)
    labelled_options = "\n".join([f"{chr(65+i)}: {option}" for i, option in enumerate(ds[q_id]["options"])])
    answer = None
    answer = model_judge(response.choices[0].message.content, labelled_options)
    parsed_answer = answer_parser(answer)
    while parsed_answer is None:
        print("Model judge failed to produce an answer. Retrying...")
        answer = model_judge(response.choices[0].message.content, labelled_options)
        parsed_answer = answer_parser(answer)
    print()
    print(answer)
    correct = answer_parser(answer) == ds[q_id]["answer"]
    completed = response.choices[0].finish_reason
    return question, response.choices[0].message.content, answer, correct, completed

def save_baseline(q_id):
    question, response, answer_response, correct, completed = baseline_CoT(q_id)
    with open("../baseline/baseline_CoTs_open.jsonl", "a") as f:
        actual_answer_label = ds[q_id]["answer"]
        actual_answer = ds[q_id]["options"][ord(actual_answer_label) - 65]
        json.dump({
            "question_id": q_id,
            "question": question,
            "response": response,
            "answer_response": answer_response,
            "correct": correct,
            "actual_answer": actual_answer,
            "complete_reason": completed
        }, f)
        f.write("\n")
    return correct

for i in range(30):
    print(f"Processing question {i}...")
    correct = save_baseline(i)
    print(f"Question {i} processed. Correct: {correct}")
