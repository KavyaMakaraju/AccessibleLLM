## **Documentation for Object Detection Use Case**

### Overview

This project showcases how to capture an image, generate captions using a pre-trained BLIP-2 model, and save the generated caption as text and audio.
---

### Setup

- **Install the needed Dependencies**
```python
!pip install git+https://github.com/huggingface/transformers.git 
!pip install pyttsx3 
!pip install gTTS 
!pip install pydub 
!pip install playsound
```
- **Import required libraries**
```python
import requests 
import pyttsx3
from gtts import gTTS
from IPython.display import Audio 
from pydub import AudioSegment 
from IPython.display import display, Javascript, Image 
from google.colab.output import eval_js 
from base64 import b64decode, b64encode 
import cv2 
import numpy as np
import PIL 
import io
import html
import time
```

- **Download Pre-trained Model Specify the 'Salesforce/blip2-opt-2.7b' model for text generation in the script**

---

### Usage

- **Capture the image-Run the following code to capture an image using your webcam:**
```python
#capturing image code
try:
  filename = take_photo('photo.jpg')
  print('Saved to {}'.format(filename))

  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
```
- **Image Processing and Caption Generation-After capturing the image, use the following code to generate a caption:**
```python
from PIL import Image 
image = Image.open("/content/photo.jpg").convert('RGB')
image = image.resize((596, 437))
display(image)
```
```python
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
```
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```
```python
inputs = processor(image, return_tensors="pt").to(device, torch.float32) 

generated_ids = model.generate(**inputs, max_length=50, min_length=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

with open("my_saved_text.txt", "a") as output_file:
    # Write the generated text to the file
    text_with_newline = generated_text + "\n"

    output_file.write(text_with_newline)

    speech = gTTS(text=generated_text, lang='en')  

speech.save("generated_text.mp3")

from playsound import playsound
Audio("generated_text.mp3")
```
- **Customization**
  1. Adjust the image path (/path/to/photo.jpg) and model name ("Salesforce/blip2-opt-2.7b") according to your setup.
  2. Modify the 'max_length' and 'min_length' parameters for the desired caption length.
---

### Dependencies
- Python3.x: Any version of python after python 3.8
- PyTorch: Deep learning model inference
- Hugging Face Transformers: Load a pre-trained BLIP-2 model
- Pillow(PIL) : Library for opening, manipulating, and saving many different image file formats
- cv2: Library designed for real-time computer vision tasks
- gTTS(Google Text-to-Speech) : Library that interfaces with Google's Text-to-Speech API
- playsound : Library used for playing audio files
  
---


### Notes 
- Ensure your system has a webcam and necessary permissions to capture images.
- Modify language ('lang='en'') in 'gTTS' for different speech synthesis.
- Adjust file paths and names based on your project structure.
