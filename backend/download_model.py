from huggingface_hub import hf_hub_download
import os

model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
local_dir = "models"

os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {filename} from {model_id}...")
print("This may take a while (approx 4GB)...")

try:
    path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded to: {path}")
except Exception as e:
    print(f"Error downloading model: {e}")
