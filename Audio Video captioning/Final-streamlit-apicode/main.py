import os
from happytransformer import HappyTextToText, TTSettings
import whisper
import streamlit as st
from pydub import AudioSegment

# Initialize the T5 model for grammar correction
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)

# Supported audio and video formats
SUPPORTED_FORMATS = ['mp4', 'mp3', 'wav', 'm4a', 'flac', 'ogg']

def split_audio(file_path, chunk_length_ms=60000):
    """Split audio into chunks for better transcription results."""
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def transcribe_and_correct(audio_chunk, model):
    """Transcribe audio and correct grammar."""
    temp_audio_path = "temp_chunk.wav"
    audio_chunk.export(temp_audio_path, format="wav")
    transcription = model.transcribe(temp_audio_path)
    text = transcription["text"]
    result = happy_tt.generate_text("grammar: " + text, args=args)
    return text, result.text

def process_file(file_path):
    """Process the file: split, transcribe, and correct grammar."""
    model = whisper.load_model("medium")
    audio_chunks = split_audio(file_path)
    
    original_texts = []
    corrected_texts = []
    for chunk in audio_chunks:
        original_text, corrected_text = transcribe_and_correct(chunk, model)
        original_texts.append(original_text)
        corrected_texts.append(corrected_text)
    
    return " ".join(original_texts), " ".join(corrected_texts)

def main():
    st.title("Audio/Video Transcription and Grammar Correction")
    st.write("Upload an audio or video file to transcribe and correct its grammar.")

    uploaded_file = st.file_uploader("Choose a file", type=SUPPORTED_FORMATS)

    if uploaded_file is not None:
        file_path = os.path.join("/content", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write(f"Processing file: {uploaded_file.name}...")
        original_text, corrected_text = process_file(file_path)
        
        st.write("Original Transcription:")
        st.text_area("Original Text", original_text, height=300)

        st.write("Corrected Transcription:")
        st.text_area("Corrected Text", corrected_text, height=300)

if __name__ == "__main__":
    main()
