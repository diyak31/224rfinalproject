import asyncio
import os
import re
from openai import AsyncOpenAI
from dlrr_reward import score_completion, split_into_steps

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

LAMBDA = 0.5
LENGTH_PENALTY = 0.3
LENGTH_THRESHOLD = 900  # word-count proxy for hitting the 1024 token ceiling

def extract_answer(text):
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"(?:answer is|answer:|therefore)[^\d-]*(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

def hybrid_reward(prompts, completions, ground_truth, **kwargs):
    async def score_all():
        tasks = [score_completion(p, c) for p, c in zip(prompts, completions)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(score_all())

    rewards = []
    for completion, gt, result in zip(completions, ground_truth, results):
        dlrr_score = result[0]

        pred = extract_answer(completion)
        binary = 1.0 if (pred is not None and pred == gt.strip()) else 0.0

        token_count = len(completion.split())
        length_pen = LENGTH_PENALTY if token_count > LENGTH_THRESHOLD else 0.0

        reward = max(0.0, LAMBDA * dlrr_score + (1 - LAMBDA) * binary - length_pen)
        rewards.append(reward)

    # log first completion for debugging
    steps = split_into_steps(completions[0])[-5:]
    correctness_scores = results[0][1]
    calibration_scores = results[0][2]
    pred0 = extract_answer(completions[0])
    binary0 = 1.0 if (pred0 is not None and pred0 == ground_truth[0].strip()) else 0.0
    tokens0 = len(completions[0].split())

    print(f"\n--- HYBRID REWARD ---")
    print(f"Steps scored ({len(steps)}):")
    for i, (s, c, cal) in enumerate(zip(steps, correctness_scores, calibration_scores)):
        print(f"  Step {i+1}: correctness={c} calibration={cal} → score={max(0, min(1, c+cal)):.2f}")
        print(f"    {s[:100]}")
    print(f"DLRR: {results[0][0]:.3f} | binary: {binary0} | tokens: {tokens0} | length_pen: {LENGTH_PENALTY if tokens0 > LENGTH_THRESHOLD else 0.0}")
    print(f"Aggregated reward: {rewards[0]:.3f}")
    print(f"GT: {ground_truth[0]} | pred: {pred0}\n")

    return rewards
