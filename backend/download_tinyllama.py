import requests
import os
import sys

url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
local_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

os.makedirs("models", exist_ok=True)

print(f"Downloading {url}...")
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(local_path, 'wb') as f:
    downloaded = 0
    for data in response.iter_content(chunk_size=1024*1024):
        downloaded += len(data)
        f.write(data)
        if total_size > 0:
            percent = downloaded / total_size * 100
            print(f"Downloaded {downloaded//(1024*1024)}MB / {total_size//(1024*1024)}MB ({percent:.1f}%)", end='\r')

print("\nDownload complete!")
