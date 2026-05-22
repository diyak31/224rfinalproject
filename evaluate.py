import os
import re
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import random

def extract_answer(text):
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"(?:answer is|answer:|therefore)[^\d-]*(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

# passed in as env var so we can reuse same file for both evals
checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)
run_name = os.environ.get("EVAL_RUN_NAME", "eval")

wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project="dlrr",
    name=run_name,
    config={
        "model": "Qwen2.5-7B-Instruct",
        "checkpoint": checkpoint_path,
        "eval_samples": 250,
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

dataset = load_dataset("openai/gsm8k", "main", split="test")
random.seed(42)
indices = random.sample(range(len(dataset)), 100)
dataset = dataset.select(indices)

correct = 0
total = 0

for example in dataset:
    prompt = f"Solve this math problem step by step. At the end, write your final answer as: #### [number]\n\n{example['question']}\n\nAnswer:"
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
    pred = extract_answer(completion)
    gt = extract_answer(example["answer"])
    if pred == gt:
        correct += 1
    total += 1

    wandb.log({
        "running_accuracy": correct / total,
        "correct": correct,
        "total": total,
    })

    if total % 50 == 0:
        print(f"{total}/250 — accuracy: {correct/total:.3f} ({correct} correct)")

wandb.log({"final_accuracy": correct / total})
wandb.finish()
print(f"\nFinal accuracy: {correct}/{total} = {correct/total:.3f}")