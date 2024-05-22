<h1 align="center">Object Detection(Image-Caption-Audio)</h1>

### Overview

This project demonstrates a complete workflow for capturing an image, generating descriptive captions for it using a pre-trained BLIP (Bootstrapped Language-Image Pretraining) model, and then saving these captions in both text and audio formats. Here’s a detailed breakdown of each step involved:

Image Capture: The process begins with capturing an image, which can be done using a camera of the device the user is using. This image serves as the input for the next stage.

Caption Generation: Using the pre-trained BLIP model, which is designed to understand and describe visual content, the project generates a caption for the captured image. The BLIP model leverages its trained neural networks to analyze the image and produce a coherent and contextually relevant textual description.

Saving the Caption as Text: Once the caption is generated, it is saved as a text file. This ensures that the generated caption can be accessed and edited

Saving the Caption as Audio: In addition to the text file, the caption is also converted into an audio file. This is typically achieved using google text-to-speech (gTTS) technology, which reads the generated text aloud and saves the output as an audio file. This allows for greater accessibility, enabling users to listen to the description rather than read it.

---

## **Table of Contents**

1)[Setup](#setup)
  - Dependencies and libraries installation
    
2)[Usage](#usage)
  - Image capturing
  - Image processing
  - Image captioning and Audio generation
    
3)[Accuracy Metrics](#accuracy-metrics)
  - BERTScore
  - ROUGEScore
  - BLEU Score
  - GlEU Score
  - CLIPScore

4)[Models Used](#models-used)
  - Model 1 : salesforce/blip2-opt-2.7b
      - About the model
      - Accuracy
        
  - Model 2 : nlpconnect/vit-gpt2-image-captioning
      - About the model
      - Accuracy
        
  - Model 3 : Salesforce/blip-image-captioning-base
      - About the model
      - Model training
          - Model training steps
      - Trained model accuracy metrics
          - untrained and trained model accuracies

5)[FAST API Code](#fast-api-code)
  - POST/capture_photo/
      - Description
      - Request body
      - Output
        
  - POST/generate_caption/
      - Description
      - Request body
      - Output

6)[Dependencies](#dependencies)


### Setup

<details>
  <summary>Setup steps</summary>

>Run the notebooks on GPU for smoother and faster performance
  
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

- **Download Pre-trained Model Specify the 'Salesforce/blip-image-captioning-base' model for text generation in the script**
</details>

---

### Usage
<details>
  <summary>Usage steps</summary>
  
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
>Ensure your system has a webcam and necessary permissions to capture images.

- **Image Processing and Caption Generation-After capturing the image, use the following code to generate a caption:**
```python
from PIL import Image 
image = Image.open("/content/photo.jpg").convert('RGB')
image = image.resize((596, 437))
display(image)
```
```python
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
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

>Modify language ('lang='en'') in 'gTTS' for different speech synthesis
>
>Adjust file paths and names based on your project structure.

**Customization**
  1. Adjust the image path (/path/to/photo.jpg) and model name ("Salesforce/blip-image-captioning-base") according to your setup.
  2. Modify the 'max_length' and 'min_length' parameters for the desired caption length.

</details>

---
### Accuracy Metrics
<details>
  <summary>Metrics used</summary>
For checking the accuracy of the models, we have used different metrics like:
- BERTScore : an automatic evaluation metric used for testing the goodness of text generation systems. It produces the following output values in the range of 0.0 to 1.0:
        - Precision
        - Recall
        - F1-score
  
- ROUGE Score: Consists of Precision, Recall and F1-score
        - ROUGE-1: Looks at individual words or unigrams.
        - ROUGE-2: Considers pairs of words or bigrams.
        - ROUGE-L: Examines the longest common subsequence.
  
- BLEU Score: BLEU is a precision-based metric since during its computation it does not consider whether all the words in the reference texts are covered in the hypothesis text or not. 

- GLEU Score: Simply the minimum of recall and precision. This GLEU score's range is always between 0 (no matches) and 1 (all match) and it is symmetrical when switching output and target.

- CLIPScore: A reference free metric that can be used to evaluate the correlation between a generated caption for an image and the actual content of the image
</details>

---

### Models Used 

<details>
<summary>Model 1 : salesforce/blip2-opt-2.7b</summary>

- **About the model**: BLIP-2 consists of a CLIP-like image encoder, a Querying Transformer (Q-Former), and a large language model.
  - Visual Question Answering
  - Chat-like conversations by feeding the image and the previous conversation as prompt to the model
  - Image Captioning
- **Usage**: You can use this model for conditional and un-conditional image captioning. The model consists of 2.7 billion parameters and is very huge in size.
- **Location**: The model can be accessed from Salesforce Hugging Face library
  - [blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)

>Dataset used for the model: [dataset (50 images and captions)](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/object_detection/dataset3.zip) 
1. **BERTScore**:
    - Precision: 0.7260
    - Recall: 0.7872
    - F1-score: 0.7541
   
3. **ROUGE Score**: 
     - <details>
       <summary>ROUGE Score(BLIP)</summary> 
       
          ``` 
          - ROUGE-1:
              - Precision: 0.5711694838529204
              - Recall: 0.7057404605198723
              - F1-score: 0.6259112389882882
          - ROUGE-2:
              - Precision: 0.35838442697653206
              - Recall: 0.46176681935195857
              - F1-score: 0.39949114886886683
          - ROUGE-L:
              - Precision: 0.5119797835463471
              - Recall:  0.6315150650003591
              - F1-score:  0.560635915103244
          ```
          </details>

3. **BLEU Score**:
   - 0.2853194240578867

4. **GLEU Score**:
    - 0.3281658319708012

5. **CLIPScore**:
    - <details>
      <summary>CLIPScore(BLIP)</summary>  
      
      ``` 
          CLIP Score for 1.jpg: 64.91
          CLIP Score for 10.jpg: 66.08
          CLIP Score for 11.jpg: 65.31
          CLIP Score for 12.jpg: 61.77
          CLIP Score for 13.jpg: 66.93
          CLIP Score for 14.jpg: 66.06
          CLIP Score for 15.jpg: 62.46
          CLIP Score for 16.jpg: 65.64
          CLIP Score for 17.jpg: 64.02
          CLIP Score for 18.jpg: 65.02
          CLIP Score for 19.jpg: 61.96
          CLIP Score for 2.jpg: 64.87
          CLIP Score for 20.jpg: 67.93
          CLIP Score for 21.jpg: 68.59
          CLIP Score for 22.jpg: 65.56
          CLIP Score for 23.jpg: 65.23
          CLIP Score for 24.jpg: 62.85
          CLIP Score for 25.jpg: 62.70
          CLIP Score for 26.jpg: 65.01
          CLIP Score for 27.jpg: 67.61
          CLIP Score for 28.jpg: 67.03
          CLIP Score for 29.jpg: 66.49
          CLIP Score for 3.jpg: 65.84
          CLIP Score for 30.jpg: 64.62
          CLIP Score for 31.jpg: 65.14
          CLIP Score for 32.jpg: 65.70
          CLIP Score for 33.jpg: 64.61
          CLIP Score for 34.jpg: 64.69
          CLIP Score for 35.jpg: 62.96
          CLIP Score for 36.jpg: 66.54
          CLIP Score for 37.jpg: 62.89
          CLIP Score for 38.jpg: 68.52
          CLIP Score for 39.jpg: 65.24
          CLIP Score for 4.jpg: 67.44
          CLIP Score for 40.jpg: 67.71
          CLIP Score for 41.jpg: 69.21
          CLIP Score for 42.jpg: 63.47
          CLIP Score for 43.jpg: 66.16
          CLIP Score for 44.jpg: 65.94
          CLIP Score for 45.jpg: 64.67
          CLIP Score for 46.jpg: 67.78
          CLIP Score for 47.jpg: 63.51
          CLIP Score for 48.jpg: 65.88
          CLIP Score for 49.jpg: 65.11
          CLIP Score for 5.jpg: 65.92
          CLIP Score for 50.jpg: 63.14
          CLIP Score for 6.jpg: 64.11
          CLIP Score for 7.jpg: 69.39
          CLIP Score for 8.jpg: 64.43
          CLIP Score for 9.jpg: 63.49
      ```
      </details>
      
To view the code and the resulting accuracies [click here](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/object_detection/accuracymetrics.ipynb)

</details>

<details>

<summary>Model 2 : nlpconnect/vit-gpt2-image-captioning</summary>

- **About the model**: This is an image captioning model trained by [@ydshieh](https://huggingface.co/ydshieh) in Flax. It produces reasonable image captioning results. It was mainly fine-tuned as a proof-of-concept for the 🤗 FlaxVisionEncoderDecoder Framework.
- **Usage**: The model is used for image captioning.
- **Location**: The model can be accessed from
  - [vit-gpt-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

>Dataset used for the model: [dataset (50 images and captions)](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/object_detection/dataset3.zip) 
 
1. **BERTScore**: 
    - Precision: 0.6246
    - Recall: 0.6585
    - F1-score: 0.6362

2. **ROUGE Score**: 
   - <details>
     <summary>ROUGE Score(VitGpt)</summary>
          
      ``` 
          - ROUGE-1:
              - Precision: 0.4042870038458274
              - Recall: 0.44236701370524906
              - F1-score: 0.41503899617383466
          - ROUGE-2:
              - Precision: 0.16989874025183618
              - Recall: 0.20881120625160865
              - F1-score: 0.18336638637340974
          - ROUGE-L:
              - Precision: 0.3620736453089395
              - Recall:  0.39817976304741015
              - F1-score:  0.37246338708799215
      ```
     </details>

3. **BLEU Score**:
    - 0.09539884316244567
  
4. **GLEU Score**:
    - 0.15503852724900705

5. **CLIPScore**:
   - <details>
     <summary>CLIPScore(VitGpt)</summary>
          
      ``` 
          CLIP Score for 1.jpg: 64.58
          CLIP Score for 10.jpg: 65.52
          CLIP Score for 11.jpg: 64.85
          CLIP Score for 12.jpg: 62.17
          CLIP Score for 13.jpg: 64.66
          CLIP Score for 14.jpg: 62.72
          CLIP Score for 15.jpg: 62.32
          CLIP Score for 16.jpg: 63.36
          CLIP Score for 17.jpg: 62.97
          CLIP Score for 18.jpg: 63.88
          CLIP Score for 19.jpg: 59.82
          CLIP Score for 2.jpg: 64.43
          CLIP Score for 20.jpg: 65.72
          CLIP Score for 21.jpg: 66.40
          CLIP Score for 22.jpg: 60.49
          CLIP Score for 23.jpg: 65.09
          CLIP Score for 24.jpg: 61.92
          CLIP Score for 25.jpg: 62.12
          CLIP Score for 26.jpg: 64.54
          CLIP Score for 27.jpg: 63.21
          CLIP Score for 28.jpg: 60.72
          CLIP Score for 29.jpg: 65.66
          CLIP Score for 3.jpg: 65.30
          CLIP Score for 30.jpg: 62.51
          CLIP Score for 31.jpg: 60.74
          CLIP Score for 32.jpg: 66.16
          CLIP Score for 33.jpg: 62.90
          CLIP Score for 34.jpg: 64.77
          CLIP Score for 35.jpg: 65.96
          CLIP Score for 36.jpg: 65.58
          CLIP Score for 37.jpg: 61.71
          CLIP Score for 38.jpg: 63.07
          CLIP Score for 39.jpg: 62.97
          CLIP Score for 4.jpg: 63.93
          CLIP Score for 40.jpg: 63.50
          CLIP Score for 41.jpg: 59.71
          CLIP Score for 42.jpg: 64.38
          CLIP Score for 43.jpg: 63.02
          CLIP Score for 44.jpg: 62.95
          CLIP Score for 45.jpg: 61.48
          CLIP Score for 46.jpg: 64.52
          CLIP Score for 47.jpg: 62.67
          CLIP Score for 48.jpg: 65.25
          CLIP Score for 49.jpg: 62.48
          CLIP Score for 5.jpg: 64.78
          CLIP Score for 50.jpg: 62.02
          CLIP Score for 6.jpg: 65.42
          CLIP Score for 7.jpg: 66.45
          CLIP Score for 8.jpg: 62.39
          CLIP Score for 9.jpg: 61.55
      ```
     </details>
     
To view the code and the resulting accuracies [click here](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/object_detection/accuracymetrics.ipynb)

</details>

<details>
  <summary>Model 3 : Salesforce/blip-image-captioning-base</summary>


- **About the model**: A Salesforce model which can be used for
  - Visual Question Answering
  - Image-Text retrieval (Image-text matching)
  - Image Captioning
- **Usage**: For our use case, we use the model for image captioning. Because of its smaller size compared to blip2-opt-2.7b, it is easier to train and produces almost alike captions.
- **Location**: The model can be accessed from
  - [blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

--- 
**Model Training**

>The dataset for the model is [arian2502/firstdataset](https://huggingface.co/datasets/arian2502/firstdataset)
>>The trained model can be found on HuggingFace. the name of the model is [arian2502/blip-icb-finetuned](https://huggingface.co/arian2502/blip-icb-finetuned)

<details>
  <summary>Model training steps</summary>
  
The **salesforce/blip-image-captioning-base model** is trained to increase the accuracy for this specific usecase. The dataset consists of 1250 images and captions. It is a custom dataset of pictures which are taken from a first person point of view. 

1.The dataset is imported.

2.The dataset is converted into a pytorch dataset via tha following code: 
```python
from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
```
3.Load the processor and model
```python
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

4.A total of 10 epochs are done for training:
```python
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(10):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

    loss = outputs.loss

    print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```
5.Check if model training is succesfull: 
```python
# load image
example = dataset[123]
image = example["image"]
image
```
![image](https://github.com/Yaswanth-B/AccessibleLLM/assets/154512247/40e33bbe-07c5-42b4-ac9c-867b9af1d018)
```python
# prepare image for the model
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```
"two lines of colorful cars racing on a field."

6.Trained model is saved and uploaded/downloaded.

>Time taken to train the model on 1250 images and captions for 10 epochs: 1 hour 8 minutes

For the full working of the code [click here](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/object_detection/trained.ipynb)

</details>

---
**Trained Model Accuracy Metrics**

<details>
  <summary>Results</summary>

1. **BERTScore**: 
    - BLIP(untrained):
        - Precision: 0.5309
        - Recall: 0.6051
        - F1-score: 0.5637
    - BLIP(trained):
        - Precision: 0.8848
        - Recall: 0.8886
        - F1-score: 0.8854
        
2. **ROUGE Score**: 
    - BLIP(untrained):
        - <details>
          <summary>ROUGE Score(untrained)</summary>
          
          ``` 
          - ROUGE-1:
              - Precision: 0.26676010739518513
              - Recall: 0.3862308472077113
              - F1-score: 0.3066864136993266
          - ROUGE-2:
              - Precision: 0.0725541792011556
              - Recall: 0.1298950742068691
              - F1-score: 0.08984282440623012
          - ROUGE-L:
              - Precision: 0.23059880787255438
              - Recall:   0.3364418769543522
              - F1-score:  0.2657411209552809
          ```
          </details>
        
    - BLIP(trained):
        - <details>
          <summary>ROUGE Score(trained)</summary>
          
          ``` 
          - ROUGE-1:
              - Precision: 0.6806757571022624
              - Recall: 0.7300539801249587
              - F1-score: 0.6995404673734046
          - ROUGE-2:
              - Precision: 0.577443416297378
              - Recall: 0.6493630657374209
              - F1-score: 0.6069535829964249
          - ROUGE-L:
              - Precision: 0.6730824339010327
              - Recall:  0.7226284310678925
              - F1-score:  0.6922038904392883
          ```
          </details>

3. **BLEU Score**:
    - BLIP(untrained): 0.04215093904464002
    - BLIP(trained): 0.704256378969982
  
4. **GLEU Score**:
    - BLIP(untrained): 0.08593777080376051
    - BLIP(trained): 0.6954368111617628


Click [here](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/object_detection/accuracymetrics(trained).ipynb) to view the code

</details>
</details>

---

## **FAST API CODE**

**API Endpoints**

- **POST /capture_photo/**
  <details>
    <summary>Photo Capture</summary>
    
  - Description: Captures the photo from the users device
  - Request Body:
      ```python
      async def capture_photo():
    global cap

    ret, frame = cap.read()
    if not ret:
        return {"error": "Failed to capture image."}

    filename = os.path.join(folder, 'photo.jpg')
    cv2.imwrite(filename, frame)

    return FileResponse(filename)
      ```
  - Output: ![Screenshot 2024-05-20 155346](https://github.com/Yaswanth-B/AccessibleLLM/assets/154512247/b109ea09-41bb-413d-b761-1887bd4e1f2a)
    
  </details>
      
- **POST /generate_caption/**
  <details>
    <summary>Image Captioning</summary>
    
    - Description: Generates caption for the photo captured
    - Request Body:
      ```python
      async def generate_caption(folder_path: str = 'C:\\Users\\aryan\\OneDrive\\Desktop\\object_detection\\photos'):
      # List all files in the folder
      image_files = os.listdir(folder_path)
    
      if not image_files:
          return JSONResponse(status_code=404, content={"message": "No photos found in the specified folder."})
      
      # Select the first image for generating caption
      selected_file = image_files[0]
      file_path = os.path.join(folder_path, selected_file)
      
      # Open and resize the image
      image = Image.open(file_path).convert('RGB')
      image = image.resize((596, 437))
      
      # Caption generation
      inputs = processor(image, return_tensors="pt").to(device, torch.float32)
      generated_ids = model.generate(**inputs, max_length=50, min_length=20)

      generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()    
  
      return {"caption": generated_text}
      ```
    - Output: ![Screenshot 2024-05-20 155410](https://github.com/Yaswanth-B/AccessibleLLM/assets/154512247/df545547-911e-4f22-91ab-925667525792)
      
  </details>

[click here](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/object_detection/main.py) to view the code.
  
---
### Dependencies
<details>
<summary>Libraries and Dependencies</summary>

- Python3.x: Any version of Python after Python 3.8
- PyTorch: Deep learning model inference
- Hugging Face Transformers: Load a pre-trained BLIP-2 model
- Pillow (PIL): Library for opening, manipulating, and saving many different image file formats
- cv2: Library designed for real-time computer vision tasks
- gTTS (Google Text-to-Speech): Library that interfaces with Google's Text-to-Speech API
- playsound: Library used for playing audio files
- FastApi: Web framework for building APIs with Python based on standard Python type hints.

</details>

