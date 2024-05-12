## Documentation for the provided FastAPI code

### Goal of the project
The goal of the project is to create a conversational AI interface that processes user input, generates responses based on predefined prompt templates. 
This project aims to help people with social anxiety by providing them with a conversational AI interface that can help them practice social interactions in a safe environment.
### Contents 
1. FastAPI Application
2. Model accuracy

### Overview

This FastAPI application serves as a conversational AI interface that processes user input, generates responses based on predefined prompt templates, and currently supports text input/output methods.

### Models

The application uses the llama2:chat model available on Ollama. Gemma and orca-mini models have been tested before.

### Prompt Templates

The application implements three custom scenarios where it mentions a scenario. The LLM is prompted with this scenario and is told to act like the persona mentioned in the prompt template.

### API Endpoints

- **POST /process/**
   - **Description:** Process user input and generate AI responses.
   - **Request Body:**
   
   ```json
   {
       "audio_file": "Optional audio file for  input",
       "user_input": "User text input",
       "input_method": "Input method (e.g., '' or 'Text')",
       "output_method": "Output method (e.g., '' or 'Text')",
       "prompt_template": "Prompt template for conversation",
       "conversation_id": "Unique ID for the conversation"
   }
   ```
   
   - **Response:**
       - **Success Response:**
           - Status Code: 200
           - Content: AI response based on the input.
       - **Error Response:**
           - Status Code: 400
           - Content: {"error": "Error message"}

### Dependencies

- **FastAPI**: Web framework for building APIs with Python.
- **Pydantic**: Data validation and parsing library.
- **Whisper**: Library for audio processing and transcription.
- **Langchain**: Library for language processing and conversation management.
- **gTTS**: Google Text-to-Speech library for converting text to speech.
- **Pyngrok**: Library for exposing local servers to the internet.

### Functions

1. **process_conversation(background_tasks, conversation_input)**
   - **Description:** Processes user input and generates AI responses based on the provided input and prompt template.
   - **Parameters:**
       - `background_tasks`: BackgroundTasks object for handling asynchronous tasks.
       - `conversation_input`: Data model containing user input and configuration details.
   - **Returns:** AI response based on the input.
2. **transcribe(audio_file_path)** (Not using in the current implementation)
   - **Description:** Transcribes audio input using Whisper library.
   - **Parameters:**
       - `audio_file_path`: Path to the audio file for transcription.
   - **Returns:** Transcribed text from the audio file.
3. **text_to_speech(text, background_tasks)** (Not using in the current implementation)
   - **Description:** Converts text to speech using gTTS library.
   - **Parameters:**
       - `text`: Text to convert to speech.
       - `background_tasks`: BackgroundTasks object for handling asynchronous tasks.
   - **Returns:** Path to the generated audio file.

### Usage

1. Send a POST request to `/process/` with the required parameters in the request body to interact with the conversational AI interface.
2. Ensure the input method, output method, and prompt template are provided for accurate responses.

## Current Challenges

1. The model is unable to provide appropriate responses.
2. The model goes into turn-based conversation and generates an entire conversation on its own.
3. The model struggles with memory, and that functionality is currently being worked on.

























## Model Accuracy
# Overview
unsloth/mistral-7b-bnb-4bit model was used as a baseline model for evaluating the conversational AI system. The model was evaluated on the BERTScore of the generated responses against the reference responses from the Daily Dialog dataset.

Even-indexed dialogs from each conversation were given as input to the LLM model, and the odd-indexed dialog was given as the target. The model was evaluated on the BERT score of the generated text and the target text. The model was evaluated on 50 such conversations, and the BERT score was calculated.

Handling odd-length conversations: In case of odd-length conversations, the last dialog was removed, and the model was evaluated on the remaining dialogs.

F1 score , precision and recall were calculated for the BERT score of the generated text and the target text.
### Dataset
I am using the [Daily Dialog dataset](https://huggingface.co/datasets/daily_dialog) to evaluate the model. The dataset contains conversations between two speakers, and each conversation has multiple dialogs. The dataset is split into training, validation, and test sets.

Example:
```json
{
   "act": [2, 1, 1, 1, 1, 2, 3, 2, 3, 4],
   "dialog": "[\"Good afternoon . This is Michelle Li speaking , calling on behalf of IBA . Is Mr Meng available at all ? \", \" This is Mr Meng ...",
   "emotion": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```



## Package Installation
The script begins by installing necessary Python packages that are essential for running the language models and evaluating their performance.

```plaintext
%%capture
!pip install -U "xformers<0.0.26" --index-url https://download.pytorch.org/whl/cu121
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install evaluate
!pip install bert_score
# Temporary fix for https://github.com/huggingface/datasets/issues/6753
!pip install datasets==2.16.0 fsspec==2023.10.0 gcsfs==2023.10.0
```


## Additional Libraries
Installs additional libraries required for language model operations and the custom unsloth package.

```plaintext
!pip install langchain
!pip install langchain-community
```

## Model and Tokenizer Initialization
Initializes a language model and tokenizer from the `unsloth` package with specific configurations.

```python
from unsloth import FastLanguageModel
import torch

# Configuration for the language model
max_seq_length = 2048
dtype = None  # Auto-detection of data type based on GPU capabilities.
load_in_4bit = True # Utilizes 4bit quantization to reduce memory usage.

# Load the pretrained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)
```

## Dataset Loading
Loads the Daily Dialog dataset for evaluation, specifically focusing on the validation split.

```python
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("daily_dialog")
eval_dataset = load_dataset("daily_dialog", split="validation")
```

## Response Generation and Evaluation
Defines functions for formatting prompts, generating responses using the model, and parsing the output. Evaluates the generated responses against reference responses using BERTScore.

```python
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
generation_config = GenerationConfig.from_pretrained("unsloth/mistral-7b-bnb-4bit")

def format_prompt(prompt: str, system_prompt: str) -> str:
    return f"""
{system_prompt}
{prompt}
""".strip()

def generate_response(prompt: str, max_new_tokens: int = 128) -> str:
    encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            temperature=0,
            generation_config=generation_config,
        )
    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
    return tokenizer.decode(answer_tokens[0], skip_special_tokens=True)
```

## Visualization
Plots precision, recall, and F1 scores of the BERTScore evaluation to visualize the performance of the conversational AI system.

```python
import matplotlib.pyplot as plt

precision = results['precision']
recall = results['recall']
f1 = results['f1']

plt.figure(figsize=(10, 6))
plt.plot(precision, label='Precision', marker='o')
plt.plot(recall, label='Recall', marker='o')
plt.plot(f1, label='F1', marker='o')

plt.title('BERTScore Evaluation')
plt.xlabel('Conversation Turns')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
```
