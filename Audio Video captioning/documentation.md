Caption Generation System Documentation

### Introduction
The Caption Generation System is a Python script designed to capture audio input, either through a microphone, an uploaded audio file, or a live video feed, and transcribe it into text captions. These captions are then displayed alongside the live video feed or extracted from an uploaded video file.

### Dependencies
The Caption Generation System relies on the following Python libraries:
- `subprocess`: For executing shell commands.
- `speech_recognition`: For speech recognition functionality.
- `threading`: For running audio recording in a separate thread.
- `keyboard`: For handling keyboard events.
- `os`: For interacting with the operating system.
- `cv2` (OpenCV): For capturing video from a webcam and displaying it.
- `pyttsx3`: For text-to-speech functionality.
- `moviepy`: For extracting audio from video files.

### Usage
The system provides multiple options for generating captions:
1. **Record Audio**: Record audio using a microphone and transcribe it into text captions.
2. **Use Uploaded Audio File**: Transcribe captions from an uploaded audio file.
3. **Upload Video File**: Extract audio from an uploaded video file and generate captions.
4. **Use Live Video Feed**: Capture live video from a webcam and generate captions based on spoken words.

### Main Components
- **Global Variables**:
  - `stop_recording`: Flag to indicate whether to stop recording audio.

- **Functions**:
  - `transcribe_audio(audio_file)`: Transcribes audio from a file into text using Google Speech Recognition service.
  - `extract_audio(video_file)`: Extracts audio from a video file and saves it as a temporary WAV file.
  - `record_audio()`: Records audio from a microphone with the ability to stop recording.
  - `recognize_audio(recognizer, source)`: Recognizes audio from a microphone and saves it to a temporary WAV file.
  - `recognize_speech()`: Recognizes speech from a microphone input.
  - `capture_and_generate_caption()`: Captures video from a webcam and generates captions based on spoken words.
  - `speak(text)`: Converts text to speech and speaks it using pyttsx3.
  - `get_user_choice()`: Prompts the user to choose a caption generation method.
  - `main()`: Main function to execute the caption generation system.

### Execution
The script starts by initializing the pyttsx3 engine. It then enters a loop where the user is prompted to choose a method for generating captions. After executing the chosen method, the user is given the option to continue or exit the program.

### Conclusion
The Caption Generation System provides a flexible and efficient way to transcribe spoken words into text captions. Whether it's recording audio from a microphone or extracting audio from a video file, the system offers various options to suit different needs.
