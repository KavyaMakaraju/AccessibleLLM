from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Open the input file and read all lines
with open("input_text.txt", "r", encoding="utf-8") as file:
    input_lines = file.readlines()

# Prepare a list to hold the generated texts
generated_lines = []

# Process each line independently
for line in input_lines:
    text = line.strip()

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

    # Append the generated text to the list
    generated_lines.append(generated_text)

# Write the generated texts to the output file
with open("generated_text.txt", "w", encoding="utf-8") as file:
    for generated_line in generated_lines:
        file.write(generated_line + "\n")

print("Text generation complete. Check 'generated_text.txt' for the output.")

