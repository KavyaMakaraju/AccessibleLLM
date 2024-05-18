from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2-medium"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

with open("input_text.txt", "r", encoding="utf-8") as file:
    input_lines = file.readlines()

# Open the output file for writing
with open("generated_text.txt", "w", encoding="utf-8") as output_file:
    for line in input_lines:
        text = line.strip()
        if text:
            input_ids = tokenizer.encode(text, return_tensors="pt")

            max_new_tokens = 5  # Maximum number of tokens in the generated text
            temperature = 0.7  # Controls the randomness of the predictions (higher temperature leads to more randomness)
            top_k = 50  # Controls the diversity of the predictions by only considering the top k tokens with highest probabilities
            num_return_sequences = 1  # Number of sequences to generate

            outputs = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,  # Adjust the maximum length accordingly
                temperature=temperature,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,  # Pad token ID
                eos_token_id=tokenizer.eos_token_id  # End-of-sequence token ID
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Write the generated text to the output file
            output_file.write(generated_text + "\n")

