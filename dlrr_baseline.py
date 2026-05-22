import os
import wandb
import random
import torch
import re
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model
from dlrr_reward import dlrr_reward

wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project="dlrr",
    name="dlrr-reward-run",
    config={
        "model": "Qwen2.5-7B-Instruct",
        "reward": "dlrr",
        "steps": 200,
        "lora_r": 64,
        "num_generations": 2,
    }
)

def extract_answer(text):
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"(?:answer is|answer:|therefore)[^\d-]*(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

dataset = load_dataset("openai/gsm8k", "main", split="train")
test_dataset = load_dataset("openai/gsm8k", "main", split="test")

def format_prompt(example):
    return {
        "prompt": f"Solve this math problem step by step. At the end, write your final answer as: #### [number]\n\n{example['question']}\n\nAnswer:",
        "ground_truth": extract_answer(example["answer"])
    }

dataset = dataset.map(format_prompt)

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
        self.dataset = dataset
        self.eval_every = eval_every
        self.num_samples = num_samples
        random.seed(42)
        self.indices = random.sample(range(len(dataset)), num_samples)
        self.eval_dataset = dataset.select(self.indices)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every != 0 or state.global_step == 0:
            return
        print(f"\n--- EVAL at step {state.global_step} ---")
        correct = 0
        self.model.eval()
        for example in self.eval_dataset:
            prompt = f"Solve this math problem step by step. At the end, write your final answer as: #### [number]\n\n{example['question']}\n\nAnswer:"
            inputs = self.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            completion = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            pred = extract_answer(completion)
            gt = extract_answer(example["answer"])
            if pred == gt:
                correct += 1
        accuracy = correct / self.num_samples
        print(f"Accuracy at step {state.global_step}: {accuracy:.3f} ({correct}/{self.num_samples})")
        wandb.log({"eval/accuracy": accuracy}, step=state.global_step)
        self.model.train()

config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_completion_length=1024,
    num_generations=2,
    learning_rate=1e-6,
    max_steps=200,
    logging_steps=5,
    save_steps=50,
    seed=42,
    output_dir="/checkpoints/dlrr-run",
    report_to="wandb",
)

eval_callback = EvalCallback(
    model=model,
    tokenizer=tokenizer,
    dataset=test_dataset,
    eval_every=50,
    num_samples=100,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=dlrr_reward,
    args=config,
    train_dataset=dataset,
    callbacks=[eval_callback],
)
trainer.train()