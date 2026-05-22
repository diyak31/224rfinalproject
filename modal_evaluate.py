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
    .add_local_file("evaluate.py", "/root/evaluate.py")
)

app = modal.App("dlrr-evaluate")

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
        ["python", "evaluate.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"evaluate.py failed for {run_name}")

@app.local_entrypoint()
def main():
    inputs = [
        ("/checkpoints/dlrr-baseline/dlrr-baseline/checkpoint-200", "binary-baseline-eval"),
        ("/checkpoints/dlrr-run/dlrr-run/checkpoint-200", "dlrr-eval"),
    ]
    # spawn both simultaneously
    handles = [evaluate.spawn(cp, name) for cp, name in inputs]
    for h in handles:
        h.get()