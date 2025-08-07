import requests
import json

# This script sends a request to an Ollama model to generate text based on a prompt.

# The URL of the Ollama API endpoint
url = "http://localhost:11434/api/generate"

# The data to be sent in the request, including the model and the prompt
data = {
    "model": "gemma3:4b",
    "prompt": "Write a short story about a robot and make it funny."
}

# Optional parameters for the request
response = requests.post(url, json=data, stream=True)

# Check if the request was successful
if response.status_code == 200:
    print("Generated text:", end=' ', flush=True)

    # Iterate over the streamed response
    for line in response.iter_lines():
        if line:
            # Decode the line and parse the JSON
            decoded_line = line.decode('utf-8')
            result = json.loads(decoded_line)

            # Get the text from the response
            generated_text = result.get('response', '')
            print(generated_text, end='', flush=True)

else:
    print("Error:", response.status_code, response.text)
