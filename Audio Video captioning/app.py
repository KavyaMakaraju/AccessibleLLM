import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from happytransformer import HappyTextToText, TTSettings
import whisper
from tempfile import NamedTemporaryFile

app = FastAPI()

# Load T5 model for grammar correction
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)

# Load Whisper model
whisper_model = whisper.load_model("medium")

@app.post("/transcribe-and-correct/")
async def transcribe_and_correct(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # Transcribe the audio file
    transcription = whisper_model.transcribe(tmp_path)
    text = transcription["text"]

    # Correct grammar
    result = happy_tt.generate_text("grammar: " + text, args=args)
    corrected_text = result.text

    # Remove temporary file
    os.remove(tmp_path)

    return JSONResponse(content={"original_transcription": text, "corrected_transcription": corrected_text})

# Run the app with: uvicorn script_name:app --reload
