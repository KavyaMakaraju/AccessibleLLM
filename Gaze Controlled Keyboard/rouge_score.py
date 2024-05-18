from rouge import Rouge

# Load the ground truth and generated texts
with open("validation_text.txt", "r", encoding="utf-8") as gt_file, open("generated_text.txt", "r", encoding="utf-8") as gen_file:
    ground_truth = gt_file.readlines()
    generated_text = gen_file.readlines()

# Ensure both files have the same number of lines
if len(ground_truth) != len(generated_text):
    raise ValueError("The number of lines in generated_text.txt and validation_text.txt do not match.")

# Initialize the rouge scorer
rouge = Rouge()

# Calculate ROUGE scores for each pair of lines
for idx, (gt_line, gen_line) in enumerate(zip(ground_truth, generated_text)):
    # Calculate ROUGE scores
    scores = rouge.get_scores(gen_line.strip(), gt_line.strip(), avg=True)
    
    # Print the ROUGE scores for the current pair of lines
    print(f"ROUGE Scores for Line {idx + 1}:")
    print(scores)
    print()

# Optionally, calculate and print the average ROUGE scores across all lines
avg_scores = rouge.get_scores(generated_text, ground_truth, avg=True)
print("Average ROUGE Scores for All Lines:")
print(avg_scores)
