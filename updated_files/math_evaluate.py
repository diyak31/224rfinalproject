import os
import re
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
import random

MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

def extract_answer(text):
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None
    start = match.end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None

def normalize(s):
    if s is None:
        return None
    return s.replace(' ', '').replace('\n', '').lower()

checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)
run_name = os.environ.get("EVAL_RUN_NAME", "math-eval")

wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project="dlrr",
    name=run_name,
    config={
        "model": "Qwen2.5-7B-Instruct",
        "dataset": "MATH Level 4-5",
        "checkpoint": checkpoint_path,
        "eval_samples": 100,
    }
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

if checkpoint_path:
    print(f"Loading LoRA checkpoint from {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()
else:
    print("No checkpoint — evaluating base model")
    model = base_model

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.padding_side = "left"

dataset = concatenate_datasets([
    load_dataset("EleutherAI/hendrycks_math", s, split="test")
    for s in MATH_SUBJECTS
])
dataset = dataset.filter(lambda x: x["level"] in ["Level 4", "Level 5"])
random.seed(42)
indices = random.sample(range(len(dataset)), 100)
dataset = dataset.select(indices)

correct = 0
total = 0

for example in dataset:
    prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\n{example['problem']}\n\nSolution:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    pred = normalize(extract_answer(completion))
    gt = normalize(extract_answer(example["solution"]))
    if pred is not None and pred == gt:
        correct += 1
    total += 1

    wandb.log({
        "running_accuracy": correct / total,
        "correct": correct,
        "total": total,
    })

    if total % 25 == 0:
        print(f"{total}/100 — accuracy: {correct/total:.3f} ({correct} correct)")

wandb.log({"final_accuracy": correct / total})
wandb.finish()
print(f"\nFinal accuracy: {correct}/{total} = {correct/total:.3f}")
