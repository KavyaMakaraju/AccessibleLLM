from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize FastAPI
app = FastAPI()

# Define the request body model
class TextGenerationRequest(BaseModel):
    input_text: str

# Define the response body model
class TextGenerationResponse(BaseModel):
    generated_text: str

@app.post("/generate-text", response_model=TextGenerationResponse)
def generate_text(request: TextGenerationRequest):
    text = request.input_text.strip()

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Set generation parameters
    max_new_tokens = 2  # Maximum number of tokens in the generated text
    temperature = 0.8  # Controls the randomness of the predictions
    top_k = 50  # Controls the diversity of the predictions
    num_return_sequences = 1  # Number of sequences to generate
    do_sample = True  # Enable sampling

    # Generate text based on the input
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,  # End-of-sequence token
        do_sample=do_sample  # Enable sampling
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated text
    return TextGenerationResponse(generated_text=generated_text)

# Run the API with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import requests

"""
# Define the URL of the API endpoint
url = "http://localhost:8000/generate-text"

# Define the input text
input_text = "Hello, how are you?"

# Define the payload
payload = {
    "input_text": input_text
}

# Send the POST request
response = requests.post(url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON
    response_data = response.json()
    generated_text = response_data.get("generated_text")
    print(f"Generated Text: {generated_text}")
else:
    print(f"Request failed with status code {response.status_code}")
"""