import modal

volume = modal.Volume.from_name("dlrr-checkpoints")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "peft",
        "wandb",
    )
    .add_local_file("math_evaluate.py", "/root/math_evaluate.py")
)

app = modal.App("math-evaluate")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret-2"),
    ],
)
def evaluate(checkpoint_path: str, run_name: str):
    import os, subprocess
    os.chdir("/root")
    env = os.environ.copy()
    env["CHECKPOINT_PATH"] = checkpoint_path
    env["EVAL_RUN_NAME"] = run_name
    result = subprocess.run(
        ["python", "math_evaluate.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"math_evaluate.py failed for {run_name}")

@app.local_entrypoint()
def main():
    inputs = [
        ("/checkpoints/math-binary/checkpoint-100", "math-binary-eval"),
        ("/checkpoints/math-hybrid/checkpoint-100", "math-hybrid-eval"),
    ]
    handles = [evaluate.spawn(cp, name) for cp, name in inputs]
    for h in handles:
        h.get()
