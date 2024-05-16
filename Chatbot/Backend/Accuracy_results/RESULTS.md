### Documentation

#### Model
The model used in this code is the unsloth/llama-2-7b model, which is accessed through Hugging Face's `transformers` library. Specifically, the model is loaded and utilized via a `pipeline` for text generation. This pipeline is then wrapped with LangChain's `HuggingFacePipeline` to integrate it into a conversation chain that can maintain memory of the conversation context.

#### Input
The input to the model consists of conversational prompts formatted with a system prompt. Each conversation turn is processed where:
- The human input is prefixed with "Human: ".
- The formatted prompt includes both the system prompt and the human input.

Example:
```
Human: Good morning , sir . Is there a bank near here ? 
System Prompt: You are a conversation AI assistant named Jack that helps other humans in developing their conversational skills. You are friendly and truthful and keep the conversation candid. Give only a SINGLE response to the human input.
```

#### Output
The output is the model's generated response to each input prompt. The response is formatted to include the model's persona as "Jack" and is expected to be a single, coherent response to the human input.

Example:
```
Output: Yes, there is a bank near here.
```
#### Ground Truth
The ground truth, or reference, is the actual response from the dataset that the model is expected to generate. In this context, the reference responses are extracted from the DailyDialog dataset, which contains human conversational dialogues.

Example:
```
Reference: There is one . 5 blocks away from here ? 
```

#### BERT Score
BERTScore is a metric used to evaluate the quality of text generation models. It leverages pre-trained BERT embeddings to compute similarity scores between the predicted text (output) and the reference text (ground truth). 

**Explanation of BERT Score:**
- **Precision**: Measures how many of the predicted tokens are relevant and found in the reference.
- **Recall**: Measures how many of the reference tokens are correctly predicted by the model.
- **F1 Score**: The harmonic mean of Precision and Recall, providing a balanced measure of both aspects.

In this code, the BERTScore is computed using the `evaluate` library, which compares the model's predictions against the references and provides the precision, recall, and F1 scores for a comprehensive evaluation of the model's performance.

###  Code Snippet:
```python
from evaluate import load

# Compute BERTScore
bertscore = load("bertscore")
results = bertscore.compute(predictions=prediction, references=reference, lang="en")

# Print BERTScore results
print(f"Precision: {results['precision']}")
print(f"Recall: {results['recall']}")
print(f"F1 Score: {results['f1']}")
```

### Bert score results
Average precision 0.77
Average F1 score 0.76
Average recall 0.75


![alt text](image.png)