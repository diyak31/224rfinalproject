import os
import wandb
import random
import torch
import re
import asyncio
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model
from dlrr_reward import score_completion, split_into_steps

wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project="dlrr",
    name="math-hybrid",
    config={
        "model": "Qwen2.5-7B-Instruct",
        "dataset": "MATH Level 4-5",
        "reward": "hybrid",
        "lambda": 0.5,
        "length_penalty": 0.3,
        "steps": 100,
        "lora_r": 64,
        "num_generations": 2,
    }
)
wandb.define_metric("eval/step")
wandb.define_metric("eval/accuracy", step_metric="eval/step")

LAMBDA = 0.5
LENGTH_PENALTY = 0.3
LENGTH_THRESHOLD = 900

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

def hybrid_reward(prompts, completions, ground_truth, **kwargs):
    async def score_all():
        tasks = [score_completion(p, c) for p, c in zip(prompts, completions)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(score_all())

    rewards = []
    for completion, gt, result in zip(completions, ground_truth, results):
        dlrr_score = result[0]

        pred = normalize(extract_answer(completion))
        binary = 1.0 if (pred is not None and pred == normalize(gt)) else 0.0

        token_count = len(completion.split())
        length_pen = LENGTH_PENALTY if token_count > LENGTH_THRESHOLD else 0.0

        reward = max(0.0, LAMBDA * dlrr_score + (1 - LAMBDA) * binary - length_pen)
        rewards.append(reward)

    steps = split_into_steps(completions[0])[-5:]
    correctness_scores = results[0][1]
    calibration_scores = results[0][2]
    pred0 = normalize(extract_answer(completions[0]))
    binary0 = 1.0 if (pred0 is not None and pred0 == normalize(ground_truth[0])) else 0.0
    tokens0 = len(completions[0].split())

    print(f"\n--- HYBRID REWARD (MATH) ---")
    for i, (s, c, cal) in enumerate(zip(steps, correctness_scores, calibration_scores)):
        print(f"  Step {i+1}: correctness={c} calibration={cal} → score={max(0, min(1, c+cal)):.2f}")
        print(f"    {s[:100]}")
    print(f"DLRR: {results[0][0]:.3f} | binary: {binary0} | tokens: {tokens0}")
    print(f"Reward: {rewards[0]:.3f} | GT: {ground_truth[0]} | pred: {pred0}\n")

    return rewards

MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

def load_math(split):
    return concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", s, split=split)
        for s in MATH_SUBJECTS
    ])

dataset = load_math("train").filter(lambda x: x["level"] in ["Level 4", "Level 5"])
test_dataset = load_math("test").filter(lambda x: x["level"] in ["Level 4", "Level 5"])

def format_prompt(example):
    return {
        "prompt": f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\n{example['problem']}\n\nSolution:",
        "ground_truth": extract_answer(example["solution"])
    }

dataset = dataset.map(format_prompt)
dataset = dataset.filter(lambda x: x["ground_truth"] is not None)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.padding_side = "left"

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, dataset, eval_every=50, num_samples=100):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.num_samples = min(num_samples, len(dataset))
        random.seed(42)
        self.eval_dataset = dataset.select(random.sample(range(len(dataset)), self.num_samples))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every != 0 or state.global_step == 0:
            return
        print(f"\n--- EVAL at step {state.global_step} ---")
        correct = 0
        self.model.eval()
        for i, example in enumerate(self.eval_dataset):
            if i % 25 == 0:
                print(f"  eval progress: {i}/{self.num_samples}")
            prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\n{example['problem']}\n\nSolution:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            completion = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            pred = normalize(extract_answer(completion))
            gt = normalize(extract_answer(example["solution"]))
            if pred is not None and pred == gt:
                correct += 1
        accuracy = correct / self.num_samples
        print(f"Accuracy at step {state.global_step}: {accuracy:.3f} ({correct}/{self.num_samples})")
        wandb.log({"eval/accuracy": accuracy, "eval/step": state.global_step})
        self.model.train()

config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_completion_length=1024,
    num_generations=2,
    learning_rate=1e-6,
    max_steps=100,
    logging_steps=5,
    save_steps=50,
    seed=42,
    output_dir="/checkpoints/math-hybrid",
    report_to="wandb",
)

eval_callback = EvalCallback(
    model=model,
    tokenizer=tokenizer,
    dataset=test_dataset,
    eval_every=100,
    num_samples=50,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=hybrid_reward,
    args=config,
    train_dataset=dataset,
    callbacks=[eval_callback],
)
trainer.train()
