import streamlit as st
import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pydub import AudioSegment
import tempfile
import os


whisper_model = whisper.load_model("base")


t5_model_name = "vennify/t5-base-grammar-correction"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)


def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result['segments']

def correct_grammar(text):
    input_text = "correct: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text


def convert_to_wav(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio = AudioSegment.from_file(uploaded_file)
    audio.export(temp_file.name, format="wav")
    return temp_file.name

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"


def process_audio(file_path):
    segments = transcribe_audio(file_path)
    
    original_text = ""
    corrected_text = ""
    time_stamped_text = ""

    for segment in segments:
        original_text += segment['text'] + " "
        corrected_text += correct_grammar(segment['text']) + " "
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        time_stamped_text += f"{start_time} - {end_time}: {segment['text']} -> {correct_grammar(segment['text'])}\n"
    
    return original_text.strip(), corrected_text.strip(), time_stamped_text.strip()

st.title("Audio/Video Transcription and Grammar Correction")

uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        file_path = convert_to_wav(uploaded_file)

        original_text, corrected_text, time_stamped_text = process_audio(file_path)

        st.text_area("Original Text", original_text, height=200)
        st.text_area("Corrected Text", corrected_text, height=200)
        st.text_area("Time-stamped Text", time_stamped_text, height=400)
        
        os.remove(file_path)
