import os
import re
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from sympy import simplify, sympify
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

def parse_math_expr(s):
    """Convert common LaTeX math into a sympy-parseable string."""
    if s is None:
        return None
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\sqrt\b', 'sqrt', s)
    s = s.replace('\\cdot', '*').replace('\\times', '*').replace('\\div', '/')
    s = s.replace('^', '**')
    s = re.sub(r'\\left|\\right|\\,|\\!', '', s)
    return s.strip()

def answers_equivalent(pred, gt):
    if pred is None or gt is None:
        return False
    if normalize(pred) == normalize(gt):
        return True
    try:
        p = sympify(parse_math_expr(pred))
        g = sympify(parse_math_expr(gt))
        return simplify(p - g) == 0
    except Exception:
        return False

checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)
run_name = os.environ.get("EVAL_RUN_NAME", "math-eval-sympy")

wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project="dlrr",
    name=run_name,
    config={
        "model": "Qwen2.5-7B-Instruct",
        "dataset": "MATH Level 4-5",
        "checkpoint": checkpoint_path,
        "eval_samples": 100,
        "equivalence": "sympy",
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

correct_exact = 0
correct_sympy = 0
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
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    pred = extract_answer(completion)
    gt = extract_answer(example["solution"])

    is_exact = normalize(pred) == normalize(gt) and pred is not None
    is_equiv = answers_equivalent(pred, gt)

    if is_exact:
        correct_exact += 1
    if is_equiv:
        correct_sympy += 1
    total += 1

    wandb.log({
        "running_accuracy_exact": correct_exact / total,
        "running_accuracy_sympy": correct_sympy / total,
        "total": total,
    })

    if total % 25 == 0:
        print(f"{total}/100 — exact: {correct_exact/total:.3f} sympy: {correct_sympy/total:.3f}")

wandb.log({
    "final_accuracy_exact": correct_exact / total,
    "final_accuracy_sympy": correct_sympy / total,
})
wandb.finish()
print(f"\nExact match:  {correct_exact}/{total} = {correct_exact/total:.3f}")
print(f"Sympy equiv:  {correct_sympy}/{total} = {correct_sympy/total:.3f}")
print(f"Recovered by sympy: {correct_sympy - correct_exact}")
