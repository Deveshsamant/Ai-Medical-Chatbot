import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_token = os.getenv("HUGGINGFACE_API_TOKEN")
headers = {"Authorization": f"Bearer {api_token}"}

models_to_test = [
    "google/flan-t5-base"
]

url_patterns = [
    "https://router.huggingface.co/models/{model}",
    "https://router.huggingface.co/hf-inference/models/{model}",
    "https://router.huggingface.co/v1/models/{model}",
    "https://api-inference.huggingface.co/models/{model}"
]

print(f"Testing API Token: {api_token[:4]}...{api_token[-4:]}")

for pattern in url_patterns:
    for model in models_to_test:
        url = pattern.format(model=model)
        print(f"\nTesting URL: {url}")
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json={"inputs": "Hello"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("Success!")
                print(f"Response: {response.json()}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")
