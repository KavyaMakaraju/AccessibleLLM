
## Documentation for the provided FastAPI code

### Overview:

This FastAPI application serves as a conversational AI interface that processes user input, generates responses based on predefined prompt templates, and currently supports text input/output methods.

### API Endpoints:

- **POST /process/**
    - **Description:** Process user input and generate AI responses.
    - **Request Body:**
    
    ```jsx
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

### Dependencies:

- **FastAPI**: Web framework for building APIs with Python.
- **Pydantic**: Data validation and parsing library.
- **Whisper**: Library for audio processing and transcription.
- **Langchain**: Library for language processing and conversation management.
- **gTTS**: Google Text-to-Speech library for converting text to speech.
- **Pyngrok**: Library for exposing local servers to the internet.

### Functions:

1. **process_conversation(background_tasks, conversation_input)**
    - **Description:** Processes user input and generates AI responses based on the provided input and prompt template.
    - **Parameters:**
        - `background_tasks`: BackgroundTasks object for handling asynchronous tasks.
        - `conversation_input`: Data model containing user input and configuration details.
    - **Returns:** AI response based on the input.
2. **transcribe(audio_file_path)**(Not using in the current implementation) 
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

### Usage:

1. Send a POST request to `/process/` with the required parameters in the request body to interact with the conversational AI interface.
2. Ensure the input method, output method, and prompt template are provided for accurate responses.
