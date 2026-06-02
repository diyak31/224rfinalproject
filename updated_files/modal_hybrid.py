import modal

volume = modal.Volume.from_name("dlrr-checkpoints")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "trl==1.4.0",
        "datasets",
        "wandb",
        "accelerate",
        "bitsandbytes",
        "peft",
        "openai",
    )
    .add_local_file("hybrid_baseline.py", "/root/hybrid_baseline.py")
    .add_local_file("hybrid_reward.py", "/root/hybrid_reward.py")
    .add_local_file("dlrr_reward.py", "/root/dlrr_reward.py")
)

app = modal.App("dlrr-hybrid")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 16,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret-2"),
        modal.Secret.from_name("openai-secret"),
    ],
)
def train():
    import os, subprocess
    os.chdir("/root")
    result = subprocess.run(
        ["python", "hybrid_baseline.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("hybrid_baseline.py failed")
    volume.commit()

@app.local_entrypoint()
def main():
    train.remote()
