# Introduction
This FastAPI application allows users to interact with GPT-3 for text generation via a web interface. It supports the following functionalities:

## Functionality
1. **Homepage (GET /)**
    - **Description:** Renders the HTML interface for interacting with the application.
    - **Supported Actions:**
        - Upload audio or video files.
        - Start and stop recording audio.
    - **Display:**
        - Chat history showing previous interactions.

2. **Transcription Endpoint (POST /transcribe)**
    - **Description:** Handles transcription of uploaded audio or video files.
    - **Parameters:**
        - `file`: UploadFile - The audio or video file to be transcribed.
    - **Supported Formats:**
        - Audio: .mp3, .wav
        - Video: .mp4
    - **Actions:**
        - Extracts audio from video files if necessary.
        - Utilizes the speech_recognition library to transcribe audio data.
    - **Response:**
        - Returns the transcribed text.
    - **Error Handling:**
        - Handles exceptions gracefully and returns appropriate error messages.

3. **JavaScript Functions**
    - `startRecording()`: Initiates audio recording using the browser's speech recognition API.
    - `stopRecording()`: Stops the ongoing audio recording.
    - `uploadFile(event)`: Handles file uploads by sending the file to the server for transcription.
    - **Browser Support:** Uses webkitSpeechRecognition for speech recognition, hence compatibility might vary across browsers.

## Dependencies
- FastAPI: A modern, fast (high-performance) web framework for building APIs with Python 3.6+ based on standard Python type hints.
- SpeechRecognition: Library for performing speech recognition with support for multiple engines and APIs.
- MoviePy: Library for video editing, used here for extracting audio from video files.
- Uvicorn: ASGI server implementation, which is used to run the FastAPI application.

## Usage
1. Install dependencies using `pip install fastapi[all] speechrecognition moviepy uvicorn`.
2. Run the application with `uvicorn main: app --reload`.
3. Access the application in a web browser at [http://localhost:8000](http://localhost:8000).
