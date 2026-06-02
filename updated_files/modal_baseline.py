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
    )
    .add_local_file("baseline.py", "/root/baseline.py")
)

app = modal.App("dlrr-baseline")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 8,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret-2"),
    ],
)
def train():
    import os, subprocess
    os.chdir("/root")
    result = subprocess.run(
        ["python", "baseline.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("baseline.py failed")
    volume.commit()
    
@app.local_entrypoint()
def main():
    train.remote()