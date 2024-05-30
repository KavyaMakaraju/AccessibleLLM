
# Audio Video Captioning

The primary objective of this project is to convert audio files to text and subsequently correct their grammar. 

Five different models are evaluated for audio-to-text conversion, and the results are utilized to perform grammar correction using large language models (LLMs). 

Among the models assessed, Whisper emerges as the most effective for accurate transcriptions. After transcription, various LLMs are employed for grammar correction, with Vennify T5 demonstrating superior performance in this task. 

For detailed information on each model, you can refer to the tables below.

You can have a look at the [video demo.](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/video-demo.mp4)

## Requirements
- The requirements file is mentioned [here.](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/requirements.txt)

## Dataset

The [dataset.zip](dataset.zip) file contains the following folders:

1. **Audio to Text Whisper Dataset**:
   - *Audio Files*: Contains 251 audio recordings.
   - *Transcriptions*: Contains audio file names and their respective transcriptions.

2. **Grammar Correction Dataset**:
   - *Grammer Dataset*: Contains original text transcriptions.
    
3. **Test Audios**:
   - Contains audio recordings created by the author.
   - These audios are utilized in the combined code, where the audio transcriptions are converted to text and subsequently corrected using the T5 LLM for grammar correction.
   
## List of Models
<details>
  <summary>Models</summary>

  - [Model 1: Wav2Vec2-base960h](Model1-Wav2Vec2)
  - [Model 2: Wav2Vec2-large960h](Model2-Wav2Vec2-Large960h)
  - [Model 3: Wav2Vec2-large960-lv60-self](Model3-Wav2Vec2-Large960h-lv60-self)
  - [Model 4: Speech Recognition Model](Model4-SpeechRecognition)
  - [Model 5: Whisper](Model5-Whisper)
</details>

<details>
  <summary>LLM Models For Grammar Correction</summary>

  - [LLM Model 1: BERT Fine-tuned](LLM-Model1-BERT-Finetuned)
  - [LLM Model 2: Vennify T5](LLM-Model2-Vennify-T5)
</details>

<details>
  <summary>Combined Code</summary>

  - [Transcription and Grammar Correction](own-audios-whisper-T5-model.ipynb)
</details>

## Models Description for Audio to Text transcription
| Model | Developer | Description | Details | Input | Output | Train Time | Metrics |
|-------|-----------|-------------|---------|-------|--------|------------|---------|
| Wav2Vec2-base| Facebook | Converts audio to text | pip install wav2vec2 | Audio files | [Transcriptions](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/Model1-Wav2Vec2/Run1/wav2vec2-base960h-results.csv) | 1 hour | Accuracy: 34.26% <br> Precision: 35.06% <br> Recall: 34.26% <br> F1-Score: 34.53% |
| Wav2Vec2-Large | Facebook | Converts audio to text | pip install wav2vec2 | Audio files | [Transcriptions](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/Model2-Wav2Vec2-Large960h/wav2vec2-large-960h-results.csv) | 1 hour | Accuracy: 57.71% <br> Precision: 58.08% <br> Recall: 57.71% <br> F1-Score: 57.29% <br> WER: 28.85% <br> CER: 14.63% |
| Wav2Vec2-Large-lv60 | Facebook | Converts audio to text | pip install wav2vec2 | Audio files | [Transcriptions](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/Model3-Wav2Vec2-Large960h-lv60-self/wav2vec2-large960h-lv60-self-results.csv) | 1 hour | Accuracy: 49.40% <br> Precision: 50.20% <br> Recall: 49.40% <br> F1-Score: 49.67% |
| SpeechRecognition | Google | Converts audio to text | pip install SpeechRecognition | Audio files | Displayed in notebook | 2 hours | Accuracy: 50.81% <br> Precision: 96.00% <br> Recall: 51.00% <br> F1-Score: 66.00% |
| Whisper | OpenAI | Converts audio to text | pip install -U openai-whisper | Audio files | [Transcriptions](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/Model5-Whisper/whisper-model-results.xlsx) | 2 hours | Accuracy: 93.40% Precision: 94.05% <br> Recall: 93.40% F1-Score: 93.39%  WER: 14.00% <br> CER: 9.00% |

## LLMs Description for Grammer Correction

| LLM Model | Description | Input | Output | Notebook | Results | Metrics | Limitation |
|-----------|-------------|-------|--------|----------|---------|---------|------------|
| BERT Fine-tuned | Fine-tuned BERT model for grammar correction. | Sentences with grammar mistakes | Grammatically corrected sentences | [Notebook](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/LLM-Model1-BERT-Finetuned/Fine-tuned-bertLLM.ipynb) | Within notebook itself | Loss: 0.196 | The model's performance is better for sentences within the dataset, and its predictions are not much accurate for sentences outside the dataset. |
| Vennify T5 | T5 model for grammar correction, fine-tuned on dataset available on Hugging Face. | Sentences with grammar mistakes | Grammatically corrected sentences | [Notebook](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/LLM-Model2-Vennify-T5/T5-Model.ipynb) | [Corrected Sentences CSV](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/LLM-Model2-Vennify-T5/T5-LLM-CORRECTED-SENTENCES.csv) | BLEU score: 0.79 | Occasionally the model auto-translates audio to different languages, but this happens only occasionally.
   
## Combined Model
- **Notebook**: Converts audios to text using Whisper and immediately corrects them using T5. Have a look at [Combined Model Notebook](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/own-audios-whisper-T5-model.ipynb)


# Streamlit: Transcribe & Correct Grammar in Audio/Video
This Streamlit application allows users to upload audio or video files and receive both transcriptions and grammar-corrected versions of the spoken content. By leveraging the Whisper model for transcription and the Happy Transformer for grammar correction, the app supports multiple audio and video formats including mp4, mp3, wav, m4a, flac, and ogg. Users can view the original transcriptions alongside the corrected versions, providing a clear comparison. The application processes long audio files by splitting them into manageable chunks, ensuring accurate and efficient transcription and correction. This tool is ideal for anyone needing to convert spoken content into well-formatted text.

## Code Explanation

## Installation

Install the dependencies using the following commands:
```bash
!pip install happytransformer
!pip install -U openai-whisper
!pip install pydub
```
## Importing Libraries

Import the required libraries using the following commands:
```bash
import os
import tempfile
from pydub import AudioSegment
from happytransformer import HappyTextToText, TTSettings
from transformers import T5ForConditionalGeneration, T5Tokenizer
import whisper
import subprocess
```
## Load Models 

Load the models using following commands:
```bash
# ---------------VENNIFY T5------------------------
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)
t5_model_name = "vennify/t5-base-grammar-correction"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
# --------------WHISPER--------------------------
model = whisper.load_model("base")
```

## Transcribe and Generate text 

Get the transcription and grammer corrected text generate using this commands:
```bash
transcription = model.transcribe(file_path)
result = happy_tt.generate_text("grammar: " + segment, args=args)
```
## Transcribe Audio

This is the function for transcribing audio using whisper model:
```bash
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result['segments']
```
## Correct sentences using T5

This is the function for generating the grammer corrected setences using VennifyT5 model:
```bash
def correct_grammar(text):
    input_text = "correct: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text
```

## Process Audio/Video files

This function segments the transcribed audio and each segement is passed as input to T5 LLM for grammer correction:
```bash
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
```

## Supported File Formats

The following file formats are supported:

**MP3**, **MP4**, **WAV**, **M4A**, **OGG**, **FLAC**

## How to Run

1. **Clone the Repository**: Clone the repository to your local machine using the following command:
   ```bash
   !git clone https://github.com/Yaswanth-B/AccessibleLLM.git

2. **Install Dependencies**: Navigate to the project directory and install the required libraries:
    ```
    pip install -r "AccessibleLLM/Audio Video captioning/requirements.txt"
3. **Run the Streamlit App**: Start the Streamlit app by running:
    ```
    streamlit run "AccessibleLLM/Audio Video captioning/Final-streamlit-apicode/main.py"
4. **Upload Your File**: Open the provided URL in your browser (usually http://localhost:8501), and upload an audio or video file in one of the supported formats.

5. **View Transcriptions**: The app will display both the original transcription and the corrected transcription, allowing you to compare them.
    
- **API Code**: The Final streamlit API code which converts audios to text using Whisper and immediately corrects them using T5 is available [here.](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/Final-streamlit-apicode/main.py)
- **Video Demo**: Have a look at the [video demo.](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/Audio%20Video%20captioning/video-demo.mp4)

## Limitations of the Project

- **Processing Time for Large Files**: One significant limitation of this project is the processing time required for larger audio and video files. Although the models employed are highly accurate, their performance can be slow when dealing with lengthy recordings.

- **Resource Intensive**: The models, especially Whisper and T5, are resource-intensive. Running these models requires substantial computational power and memory, which might not be available on standard consumer hardware. Users with limited resources might experience slower performance and potential system instability.

## Hardware Requirements

To achieve optimal performance, the following hardware specifications are recommended:

- **CPU**: A multi-core processor with a high clock speed (e.g., Intel i7 or AMD Ryzen 7 and above) is recommended to handle the computational load.

- **GPU**: A high-performance GPU (e.g., NVIDIA RTX 3080 or equivalent) is essential for accelerating model inference, particularly for Whisper and T5 models. CUDA compatibility is necessary for leveraging GPU capabilities.

- **RAM**: At least 16 GB of RAM is recommended, with 32 GB or more preferred, especially for processing longer audio files and running multiple processes simultaneously.

- **Storage**: An SSD with sufficient storage (at least 500 GB) is recommended for faster data access and handling large datasets, audio files, and model weights.

## Future Scope

- **Optimization for Real-Time Processing**: Enhancing the models and implementation to support real-time transcription and grammar correction can open up new use cases, such as live captioning for events, streaming services, and interactive applications.

- **Integration with Other Platforms**: Expanding the project's integration capabilities with other platforms and services, such as video conferencing tools, content management systems, and educational platforms, can enhance its utility and reach a broader audience.

- **Support for Additional Languages**: Extending support to multiple languages for both transcription and grammar correction can significantly broaden the application's applicability, catering to a global user base.


