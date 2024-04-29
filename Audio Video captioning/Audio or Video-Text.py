import subprocess

import speech_recognition as sr
import threading
import keyboard  
import os
import cv2
import pyttsx3

from moviepy.editor import VideoFileClip


# Global variable to indicate whether to stop recording
stop_recording = False

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  # Record the audio file

    try:
        text = recognizer.recognize_google(audio)
        print('Captions:')
        print(text.strip())
        return text.strip(), True  # Return the transcribed text and indicate successful transcription
    except sr.UnknownValueError:
        print("Error: Could not understand audio")
        return "", False  # Return empty text and indicate transcription error
    except sr.RequestError as e:
        print(f"Error: Could not request results from Google Speech Recognition service; {e}")
        return "", False  # Return empty text and indicate transcription error

# Function to extract audio from video file
def extract_audio(video_file):
    audio_file = "temp_audio.wav"
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(audio_file)
    return audio_file


# Function to record audio with the ability to stop recording
def record_audio():
    global stop_recording
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording audio... Press 'q' to stop recording.")
        audio_thread = threading.Thread(target=recognize_audio, args=(recognizer, source))
        audio_thread.start()

        while not stop_recording:
            if keyboard.is_pressed('q'):  # Press 'q' to stop recording
                stop_recording = True
                print("Recording stopped.")
                break

        audio_thread.join()

    text, success = transcribe_audio("temp_audio.wav")
    stop_recording = False  # Reset stop_recording flag

# Function to recognize audio
def recognize_audio(recognizer, source):
    global stop_recording
    audio = recognizer.listen(source)
    with open("temp_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        if command.lower() == "stop":  # Check if command is to stop
            return "stop"
        return command
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the command.")
        return None

# Function to capture video from webcam and generate caption
def capture_and_generate_caption():
    cap = cv2.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()  # Capture frame
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Camera", frame)  # Display camera feed

        # Recognize speech
        command = recognize_speech()
        if command == "stop":  # Check if command is to stop
            break
        elif command:
            # Generate caption from recognized speech
            caption = f"Speech: {command}"
            print("Caption:", caption)
            speak(caption)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_user_choice():
    choice = input("For Caption Generation choose one:\n"
                   "1.Record audio (press R)\n"
                   "2.Use an uploaded audio file (press U)\n"
                   "3.Upload a video file (press V)\n"
                   "4.Use live video feed (press L)\n").lower()
    while choice not in ['r', 'u', 'v', 'l']:
        choice = input("Invalid choice. Please enter 'R' for recording, 'U' for using an uploaded audio file, 'V' for uploading a video file, or 'L' for using live video feed: ").lower()
    return choice

def main():
    while True:
        choice = get_user_choice()
        if choice == 'r':
            record_audio()
        elif choice == 'v':
            uploaded_video_file = input("Enter the path to the uploaded video file: ").strip('\"')  # Remove surrounding quotes
            audio_file = extract_audio(uploaded_video_file)
            success = transcribe_audio(audio_file)
        elif choice == 'l':
            capture_and_generate_caption()
        else:
            uploaded_audio_file = input("Enter the path to the uploaded audio file: ").strip('\"')  # Remove surrounding quotes
            success = transcribe_audio(uploaded_audio_file)
            if success:
                continue  # Go back to the main loop to ask for choices again
        if input("Do you want to continue? (Y/N): ").lower() != 'y':
            break

if __name__ == "__main__":
    engine = pyttsx3.init()
    main()
