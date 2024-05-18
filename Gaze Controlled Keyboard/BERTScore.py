import torch
from transformers import BertTokenizer, BertForSequenceClassification
import bert_score

# Read generated texts
with open("generated_text.txt", "r", encoding="utf-8") as file:
    generated_lines = file.readlines()

# Read validation texts
with open("validation_text.txt", "r", encoding="utf-8") as file:
    validation_lines = file.readlines()

# Ensure both files have the same number of lines
if len(generated_lines) != len(validation_lines):
    raise ValueError("The number of lines in generated_text.txt and validation_text.txt do not match.")

# Compute BERT Score for each line
P_A, R_A, F1_A = bert_score.score(
    generated_lines,
    validation_lines,
    lang='en',
    model_type='bert-base-uncased',  # BERT model type
    # verbose=True  # Print progress
)

# Print the results for each line
print("BERT Analysis Scores for each line:")
for i in range(len(generated_lines)):
    print(f"Line {i + 1}:")
    print(f"Generated: {generated_lines[i].strip()}")
    print(f"Reference: {validation_lines[i].strip()}")
    print(f"Precision: {P_A[i].item():.4f}")
    print(f"F1-score: {F1_A[i].item():.4f}")
    print(f"Recall: {R_A[i].item():.4f}")
    print()

# Print average results
print("Average BERT Analysis Score:")
print(f"Average Precision: {P_A.mean():.4f}")
print(f"Average F1-score: {F1_A.mean():.4f}")
print(f"Average Recall: {R_A.mean():.4f}")
