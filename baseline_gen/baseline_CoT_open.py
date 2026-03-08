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
    
def judge_and_parse(response, options):
    judge_response = None

    for _ in range(5):
        judge_response = model_judge(response, options)
        parsed_answer = answer_parser(judge_response)
        if parsed_answer:
            return judge_response, parsed_answer

    return judge_response, "UNKNOWN"
    
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
    if response.choices[0].finish_reason == "length":
        return question, response.choices[0].message.content, None, None, None, None, None, None, response.choices[0].finish_reason
    labelled_options = "\n".join([f"{chr(65+i)}: {option}" for i, option in enumerate(ds[q_id]["options"])])
    answers = []
    respones = []
    for i in range(5):
        judge_response, parsed_answer = judge_and_parse(response.choices[0].message.content.strip(), labelled_options)
        answers.append(parsed_answer)
        respones.append(judge_response)
    votes = {option: answers.count(option) for option in set(answers)}
    # if most votes != 4 and at least one of the votes for the correct answer, run 5 more judging rounds
    if max(votes.values()) < 4 and votes.get(ds[q_id]["answer"], 0) > 0:
        for i in range(5):
            judge_response, parsed_answer = judge_and_parse(response.choices[0].message.content.strip(), labelled_options)
            answers.append(parsed_answer)
            respones.append(judge_response)
        votes = {option: answers.count(option) for option in set(answers)}
    # most common parsed answer
    parsed_answer = max(set(answers), key=answers.count)
    judge_response = respones[answers.index(parsed_answer)]
    
    print()
    print(parsed_answer)
    correct = parsed_answer == ds[q_id]["answer"]
    completed = response.choices[0].finish_reason
    actual_answer = ds[q_id]["options"][ord(ds[q_id]["answer"]) - 65]
    actual_answer_label = ds[q_id]["answer"]
    return question, response.choices[0].message.content, parsed_answer, judge_response, votes, correct, actual_answer, actual_answer_label, completed

def save_baseline(q_id):
    question, response, answer_response, judge_response, votes, correct, actual_answer, actual_answer_label, completed = baseline_CoT(q_id)
    with open("../baseline/baseline_open_voting.jsonl", "a") as f:
        json.dump({
            "question_id": q_id,
            "question": question,
            "response": response,
            "answer_response": answer_response,
            "judge_response": judge_response,
            "votes": votes,
            "correct": correct,
            "actual_answer": actual_answer,
            "actual_answer_label": actual_answer_label,
            "complete_reason": completed
        }, f)
        f.write("\n")
    return correct

# for i in range(13, 30):
#     print(f"Processing question {i}...")
#     correct = save_baseline(i)
#     print(f"Question {i} processed. Correct: {correct}")


save_baseline(2)