# Step-Level Rewards for Mathematical Reasoning

Exploring how different reward signals affect the mathematical reasoning of language models.

Most reinforcement-learning approaches reward a model only when it reaches the correct final answer. We are experimenting with a denser alternative: scoring individual reasoning steps for correctness and calibration, then using that feedback during training. The goal is to see whether better intermediate feedback can produce reasoning that is both more accurate and more reliable.

## What we are testing

Our experiments fine-tune **Qwen2.5-7B-Instruct** with GRPO and LoRA on mathematical reasoning datasets, including **GSM8K** and harder problems from **MATH**.

We compare several reward strategies:

- **Binary reward** — credit only for a correct final answer
- **Step-level reward** — an LLM judge scores each part of the reasoning process
- **Hybrid reward** — combines step-level feedback with final-answer correctness
- **Outcome-conditioned reward** — reduces step-level credit when the final answer is wrong

Training and evaluation runs are logged with Weights & Biases and can be launched on Modal GPUs.

## Repository layout

- `baseline.py` — binary-reward training on GSM8K
- `dlrr_baseline.py` and `dlrr_reward.py` — step-level reward training
- `evaluate.py` — checkpoint evaluation
- `modal_*.py` — Modal entry points for remote training and evaluation
- `updated_files/` — newer hybrid, ablation, and MATH experiments

## Running an experiment

The scripts expect Modal secrets for Hugging Face, Weights & Biases, and—when using LLM-based rewards—OpenAI. Once those are configured, a run can be launched with:

```bash
modal run modal_baseline.py
modal run modal_dlrr.py
modal run modal_evaluate.py
```

For the newer experiment variants, run the corresponding Modal entry point from `updated_files/`.
