import asyncio
import os
import re
from openai import AsyncOpenAI
from dlrr_reward import score_completion, split_into_steps

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

LAMBDA = 0.5
DISCOUNT = 0.3  # scale factor applied to step reward on wrong-answer trajectories

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

def hybrid_reward_oc(prompts, completions, ground_truth, **kwargs):
    """
    Outcome-conditioned hybrid reward:
      correct trajectory:  λ * dlrr + (1 - λ) * 1.0   (full credit)
      wrong trajectory:    λ * dlrr * DISCOUNT          (discounted step signal)

    Anchors step-level scores to outcome so the model can't achieve high
    reward by writing fluent-but-wrong reasoning.
    """
    async def score_all():
        tasks = [score_completion(p, c) for p, c in zip(prompts, completions)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(score_all())

    rewards = []
    for completion, gt, result in zip(completions, ground_truth, results):
        dlrr_score = result[0]

        pred = normalize(extract_answer(completion))
        correct = pred is not None and pred == normalize(gt)

        if correct:
            reward = LAMBDA * dlrr_score + (1 - LAMBDA) * 1.0
        else:
            reward = LAMBDA * dlrr_score * DISCOUNT

        rewards.append(max(0.0, reward))

    # log first completion
    steps = split_into_steps(completions[0])[-5:]
    correctness_scores = results[0][1]
    calibration_scores = results[0][2]
    pred0 = normalize(extract_answer(completions[0]))
    correct0 = pred0 is not None and pred0 == normalize(ground_truth[0])

    print(f"\n--- HYBRID-OC REWARD ---")
    for i, (s, c, cal) in enumerate(zip(steps, correctness_scores, calibration_scores)):
        print(f"  Step {i+1}: correctness={c} calibration={cal} → score={max(0, min(1, c+cal)):.2f}")
        print(f"    {s[:100]}")
    print(f"DLRR: {results[0][0]:.3f} | correct: {correct0} | reward: {rewards[0]:.3f}")
    print(f"GT: {ground_truth[0]} | pred: {pred0}\n")

    return rewards
