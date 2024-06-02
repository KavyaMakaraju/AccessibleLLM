## Documentation for the provided FastAPI code

### Goal of the project
The goal of the project is to create a conversational AI interface that processes user input, generates responses based on predefined prompt templates. 
This project aims to help people with social anxiety by providing them with a conversational AI interface that can help them practice social interactions in a safe environment.
### Contents 
1. [Overview](#overview)
2. [Setup](#setup)
3. [Models](#models)
4. [Prompt Templates](#prompt-templates)
5. [API Endpoints](#api-endpoints)
6. [Functions](#functions)
7. [UI Interface](#ui-interface)
8. [Dependencies](#dependencies)
9. [Limitations](#limitations)

### Overview

This FastAPI application serves as a conversational AI interface that processes user input, generates responses based on predefined prompt templates, and currently supports text input/output methods.

## Setup
Install[Ollama](https://ollama.com/download)

Installing dependencies

```bash
pip install -r Chatbot/Backend/requirements.txt
```
### Models

The application uses the llama2:chat model available on Ollama. 
Previous models used in the project include:
Llama3 
unsloth LLama 2 
unsloth Gemma
Unsloth Mistral

**Model scores**

| Model | Average precision | Average F1 score | Average recall |
|-|-|-|-|
| Llama 2 | 0.83 | 0.83 | 0.83 |
| Unsloth Llama 2 | 0.52 | 0.52 | 0.51 |
| Unsloth Mistral | 0.80 | 0.79| 0.78 |
| Unsloth Gemma  | 0.77 | 0.76 | 0.76|

### Prompt Templates

The application implements three custom scenarios where it mentions a scenario. The LLM is prompted with this scenario and is told to act like the persona mentioned in the prompt template.

### API Endpoints

- **POST /process/**
   - **Description:** Process user input and generate AI responses.
   - **Request Body:**
   
   ```json
   {
       "user_input": "User text input",
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

### Functions

1. **process_conversation(background_tasks, conversation_input)**
   - **Description:** Processes user input and generates AI responses based on the provided input and prompt template.
   
    ```python
    @app.post("/process/")
    async def process_conversation(background_tasks: BackgroundTasks, conversation_input: ConversationInput):
        if not conversation_input.user_input:
            return JSONResponse(content={"error": "No user input provided"}, status_code=400)
        if not conversation_input.prompt_template:
            return JSONResponse(content={"error": "No prompt template provided"}, status_code=400)
        if not conversation_input.input_method:
            return JSONResponse(content={"error": "No input method provided"}, status_code=400)
        if not conversation_input.output_method:
            return JSONResponse(content={"error": "No output method provided"}, status_code=400)
    

        if conversation_input.conversation_id not in conversations:
        llm = Ollama(model="llama2:chat", temperature=0.3, top_k=50)
        memory = ConversationBufferMemory(memory_key="chat_history", k=6, return_messages=True)
        conversations[conversation_input.conversation_id] = LLMChain(
            llm=llm,
            prompt=prompt_templates[conversation_input.prompt_template],
            verbose=True,
            memory=memory
        )
    ```


## UI Interface
The project uses the UI from this [vercel template](https://github.com/vercel/ai-chatbot)

### Dependencies
The dependencies for the FastAPI application are included in requirements.txt file. 
The dependencies include:
- **FastAPI**: Web framework for building APIs with Python.
- **Pydantic**: Data validation and parsing library.
- **Whisper**: Library for audio processing and transcription.
- **Langchain**: Library for language processing and conversation management.
- **gTTS**: Google Text-to-Speech library for converting text to speech.
- **Pyngrok**: Library for exposing local servers to the internet.


### Usage

1. Send a POST request to `/process/` with the required parameters in the request body to interact with the conversational AI interface.
2. Ensure the input method, output method, and prompt template are provided for accurate responses.

## Limitations

1. The model is currently facing diffciulties in providing consistent responses.
2. The app currently only supports text input/output methods.

























