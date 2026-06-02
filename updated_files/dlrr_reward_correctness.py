import asyncio
import os
import re
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

def split_into_steps(completion):
    # try numbered steps first
    steps = re.split(r'\n(?=\s*(?:Step\s*)?\d+[\.\:])', completion)
    if len(steps) > 1:
        return [s.strip() for s in steps if s.strip()]
    # fall back to double newlines
    steps = [s.strip() for s in completion.split('\n\n') if s.strip()]
    if len(steps) > 1:
        return steps
    # fall back to single newlines
    return [s.strip() for s in completion.split('\n') if s.strip()]

async def score_step(problem, previous_steps, current_step, ground_truth=None):
    previous = "\n".join(previous_steps) if previous_steps else "None"
    gt_line = f"\nGround truth answer: {ground_truth}" if ground_truth else ""
    prompt = f"""You are evaluating a single step in a math reasoning chain.

Problem: {problem}{gt_line}

Previous steps:
{previous}

Current step to evaluate:
{current_step}

Rate this step on correctness only:
1 = completely correct and clearly moves toward the correct solution
0.5 = partially correct, or correct but vague/inefficient
0 = mathematically wrong, irrelevant, or misleading

Respond in exactly this format with no other text:
correctness: [0, 0.5, or 1]"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    correctness = 0.5  # default on parse failure

    for line in raw.split('\n'):
        line = line.strip().lower()
        if line.startswith('correctness:'):
            try:
                val = float(line.split(':')[1].strip())
                if val in [0.0, 0.5, 1.0]:
                    correctness = val
            except ValueError:
                pass

    # calibration zeroed out — ablation condition
    calibration = 0.0
    step_score = correctness
    return step_score, correctness, calibration

async def score_completion(problem, completion, ground_truth=None):
    steps = split_into_steps(completion)
    steps = steps[-5:] if len(steps) > 5 else steps
    if not steps:
        return 0.0, [], []

    tasks = []
    for i, step in enumerate(steps):
        previous = steps[:i]
        tasks.append(score_step(problem, previous, step, ground_truth=ground_truth))

    results = await asyncio.gather(*tasks)
    step_scores = [r[0] for r in results]
    correctness_scores = [r[1] for r in results]
    calibration_scores = [r[2] for r in results]  # all 0.0

    weights = [1.0] * len(step_scores)
    weights[-1] = 2.0
    total = sum(w * r for w, r in zip(weights, step_scores))
    aggregated = total / sum(weights)

    return aggregated, correctness_scores, calibration_scores

def dlrr_reward(prompts, completions, ground_truth, **kwargs):
    async def score_all():
        tasks = [
            score_completion(p, c, ground_truth=gt)
            for p, c, gt in zip(prompts, completions, ground_truth)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(score_all())
    rewards = [r[0] for r in results]

    steps = split_into_steps(completions[0])[-5:]
    correctness = results[0][1]
    calibration = results[0][2]

    print(f"\n--- DLRR REWARD (correctness-only ablation) ---")
    print(f"Completion[:300]: {completions[0][:300]}")
    print(f"Steps scored ({len(steps)}):")
    for i, (s, c, cal) in enumerate(zip(steps, correctness, calibration)):
        print(f"  Step {i+1}: correctness={c} calibration={cal} → score={c:.2f}")
        print(f"    {s[:100]}")
    print(f"Aggregated reward: {rewards[0]:.3f}")
    print(f"GT: {ground_truth[0]}\n")

    return rewards