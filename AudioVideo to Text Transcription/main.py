from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
import speech_recognition as sr
import os
from moviepy.editor import VideoFileClip

app = FastAPI()

# Chat history storage
chat_history = []

# Function to generate response
def generate_response(prompt, max_length=100):
    if prompt.lower() == "hello":
        return "Hello there! How can I assist you today?"
    elif prompt.lower() == "are you good?":
        return "I'm doing well, thank you for asking."
    elif "upload audio/video" in prompt.lower():
        return "Sure! Please use the button below to upload your audio or video file."
    elif prompt.lower() == "record audio":
        return "Please click the button below to start recording audio."
    else:
        return "I'm sorry, I cannot understand the request."

# Function to transcribe audio data
def transcribe_audio(audio_data):
    try:
        recognizer = sr.Recognizer()
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Chat with GPT-3</title>
            <script>
                var recognition = new webkitSpeechRecognition();
                recognition.continuous = true; // Keep listening until stopped manually

                recognition.onstart = function(event) {
                    console.log('Recording started...');
                };

                recognition.onresult = function(event) {
                    var message = event.results[event.results.length - 1][0].transcript;
                    var chatHistory = document.getElementById("chat-history");
                    var messageElement = document.createElement("div");
                    messageElement.innerHTML = "<strong>Caption:</strong> " + message;
                    chatHistory.appendChild(messageElement);
                };

                recognition.onerror = function(event) {
                    console.error('Speech recognition error:', event.error);
                };

                recognition.onend = function() {
                    console.log('Recording stopped.');
                };

                function startRecording() {
                    recognition.start();
                }

                function stopRecording() {
                    recognition.stop();
                }

                function uploadFile(event) {
                    var file = event.target.files[0];
                    var formData = new FormData();
                    formData.append('file', file);
                    
                    fetch("/transcribe", {
                        method: "POST",
                        body: formData
                    })
                    .then(response => response.text())
                    .then(data => {
                        var chatHistory = document.getElementById("chat-history");
                        var messageElement = document.createElement("div");
                        messageElement.innerHTML = "<strong>Caption:</strong> " + data;
                        chatHistory.appendChild(messageElement);
                    });
                }
            </script>
        </head>
        <body>
            <h1>Chat with GPT-3</h1>
            <div>
                <input type="file" accept=".mp3,.wav,.mp4" onchange="uploadFile(event)">
                <button onclick="startRecording()">Start Recording</button>
                <button onclick="stopRecording()">Stop Recording</button>
            </div>
            <h2>Chat History</h2>
            <div id="chat-history">
                <!-- Chat history will be displayed here -->
            </div>
        </body>
    </html>
    """

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        # Check file type and transcribe accordingly
        if file.filename.endswith('.mp4'):
            # Extract audio from the video file
            video_clip = VideoFileClip(file_path)
            audio_clip = video_clip.audio
            audio_file_path = f"uploads/{file.filename.split('.')[0]}.wav"
            audio_clip.write_audiofile(audio_file_path)

            # Close file handles associated with the video and audio clips
            video_clip.close()
            audio_clip.close()
            
            # Transcribe the extracted audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as audio_file:
                audio_data = recognizer.record(audio_file)
            text = recognizer.recognize_google(audio_data)
            
            # Remove the extracted audio file after transcription
            os.remove(audio_file_path)
        else:
            # Transcribe the uploaded audio file
            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as audio_file:
                audio_data = recognizer.record(audio_file)
            text = recognizer.recognize_google(audio_data)

        # Remove the uploaded file after transcription
        os.remove(file_path)
        
        return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
