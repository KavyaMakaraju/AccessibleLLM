# American Sign Language Reader
The primary objective of this project is to detect characters of American Sign Language from a Video feed and use an open-source LLM to complete the sentence to enhance ease of use.

CNN model were trained for the Detection of Americal Sign Language charactersd. The most performant model in our local systems was [EfficientNet](https://arxiv.org/abs/1905.11946)

For detailed information on each model, you can refer to the notebook [here](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/ASL-Reader/CNN_training.ipynb).

You can have a look at the [video demo.]()

## Requirements
- The requirements file is mentioned [here.](https://github.com/Yaswanth-B/AccessibleLLM/blob/main/ASL-Reader/requirements.txt)

## Dataset

The dataset -> [asl_data2.zip](https://github.com/Yaswanth-B/AccessibleLLM/tree/main/ASL-Reader/Dataset) file contains the following folders:

1. **Train**:
   - *Images*: Contains 1511 images of signs.
   - *_annotations.csv*: bounding boxes and class labels for images.

2. **Test**:
   - *Images*: Contains 71 images of signs.
   - *_annotations.csv*: bounding boxss and class labels for images.
    
3. **Val**:
   - *Images*: Contains 144 images of signs.
   - *_annotations.csv*: bounding boxes and class labels for images.
   
## List of Models for Sign Detection
### CNN (Small) without EarlyStopping
Classification report:
```bash
precision    recall  f1-score   support

           A       0.53      0.77      0.62        13
           B       0.44      0.44      0.44         9
           C       0.67      0.50      0.57         8
           D       0.71      0.83      0.77         6
           E       0.56      0.56      0.56        16
           F       0.64      0.70      0.67        10
           G       0.62      0.67      0.64        12
           H       0.83      0.38      0.53        13
           I       0.75      0.80      0.77        15
           J       0.76      0.80      0.78        20
           K       0.58      0.78      0.67         9
           L       1.00      0.55      0.71        11
           M       0.75      0.75      0.75         8
           N       0.50      0.73      0.59        11
           O       0.64      0.58      0.61        12
           P       0.86      0.55      0.67        11
           Q       0.82      0.64      0.72        14
           R       0.43      0.38      0.40         8
           S       0.61      0.69      0.65        16
           T       0.50      0.60      0.55        10
           U       0.46      0.46      0.46        13
           V       0.33      0.29      0.31         7
           W       0.89      0.84      0.86        19
           X       0.80      0.73      0.76        11
           Y       0.80      0.80      0.80        10
           Z       0.71      0.91      0.80        11

    accuracy                           0.66       303
   macro avg       0.66      0.64      0.64       303
weighted avg       0.68      0.66      0.66       303
```

### CNN (Small) with EarlyStopping
Classification Report:
```bash
precision    recall  f1-score   support

           A       1.00      0.54      0.70        13
           B       1.00      0.44      0.62         9
           C       0.62      0.62      0.62         8
           D       1.00      0.50      0.67         6
           E       0.89      0.50      0.64        16
           F       0.60      0.60      0.60        10
           G       0.78      0.58      0.67        12
           H       0.39      0.54      0.45        13
           I       0.52      0.80      0.63        15
           J       0.57      0.60      0.59        20
           K       0.50      0.56      0.53         9
           L       1.00      0.55      0.71        11
           M       0.50      0.62      0.56         8
           N       0.50      0.36      0.42        11
           O       0.54      0.58      0.56        12
           P       0.55      0.55      0.55        11
           Q       0.62      0.71      0.67        14
           R       0.14      0.38      0.21         8
           S       0.85      0.69      0.76        16
           T       0.80      0.40      0.53        10
           U       0.38      0.38      0.38        13
           V       0.21      0.43      0.29         7
           W       0.94      0.79      0.86        19
           X       0.90      0.82      0.86        11
           Y       0.88      0.70      0.78        10
           Z       0.59      0.91      0.71        11

    accuracy                           0.60       303
   macro avg       0.66      0.58      0.60       303
weighted avg       0.68      0.60      0.61       303
```
### CNN (Large) with EarlyStopping
Classification Report:
```bash
precision    recall  f1-score   support

           A       0.83      0.77      0.80        13
           B       0.78      0.78      0.78         9
           C       0.58      0.88      0.70         8
           D       0.67      1.00      0.80         6
           E       0.86      0.75      0.80        16
           F       0.78      0.70      0.74        10
           G       0.53      0.75      0.62        12
           H       0.86      0.46      0.60        13
           I       0.81      0.87      0.84        15
           J       1.00      0.75      0.86        20
           K       0.64      0.78      0.70         9
           L       0.91      0.91      0.91        11
           M       0.64      0.88      0.74         8
           N       0.88      0.64      0.74        11
           O       1.00      0.58      0.74        12
           P       0.71      0.91      0.80        11
           Q       1.00      0.71      0.83        14
           R       0.57      0.50      0.53         8
           S       0.62      0.81      0.70        16
           T       0.89      0.80      0.84        10
           U       0.64      0.54      0.58        13
           V       0.60      0.43      0.50         7
           W       0.88      0.74      0.80        19
           X       0.83      0.91      0.87        11
           Y       0.62      1.00      0.77        10
           Z       0.79      1.00      0.88        11

    accuracy                           0.76       303
   macro avg       0.77      0.76      0.75       303
weighted avg       0.79      0.76      0.76       303
```
### EfficientNet with EarlyStopping
Classification Report:
```bash
precision    recall  f1-score   support

           A       1.00      1.00      1.00        13
           B       0.90      1.00      0.95         9
           C       1.00      0.75      0.86         8
           D       1.00      0.83      0.91         6
           E       0.81      0.81      0.81        16
           F       1.00      0.80      0.89        10
           G       0.83      0.83      0.83        12
           H       0.92      0.85      0.88        13
           I       0.94      1.00      0.97        15
           J       1.00      1.00      1.00        20
           K       1.00      0.78      0.88         9
           L       1.00      1.00      1.00        11
           M       0.86      0.75      0.80         8
           N       1.00      1.00      1.00        11
           O       0.83      0.83      0.83        12
           P       0.92      1.00      0.96        11
           Q       1.00      1.00      1.00        14
           R       0.36      0.62      0.45         8
           S       0.67      1.00      0.80        16
           T       1.00      0.30      0.46        10
           U       0.89      0.62      0.73        13
           V       0.88      1.00      0.93         7
           W       0.95      0.95      0.95        19
           X       0.92      1.00      0.96        11
           Y       0.91      1.00      0.95        10
           Z       1.00      1.00      1.00        11

    accuracy                           0.89       303
   macro avg       0.91      0.87      0.88       303
weighted avg       0.91      0.89      0.89       303
```

## LLM Models For Sentence Completion

  - LLM Model 1: GPT 2
  - LLM Model 2: GPT Neo

## Installation

Install the dependencies using the following commands:
```bash
!pip install tensorflow keras 
```
## Importing Libraries

Import the required libraries using the following commands:
```bash
import cv2
import numpy as np
from keras.models import load_model
from spellchecker import SpellChecker
from efficientnet.tfkeras import EfficientNetB0
import requests
```

## How to Run

1. **Clone the Repository**: Clone the repository to your local machine using the following command:
   ```bash
   !git clone https://github.com/Yaswanth-B/AccessibleLLM.git

2. **Install Dependencies**: Navigate to the project directory and install the required libraries:
    ```
    pip install -r "AccessibleLLM/ASL-Reader/requirements.txt"
3. **Run the API**: Start the FastAPI app by running:
    ```
    uvicorn main:app --reload
4. **Run the Sign Detection App**: Start the app by running:
    ``` 
    C:\Users\yaswa\Projects\ASL-Reader> python asl_recognizer2.py
## Limitations of the Project

- **Resource Intensive**: The models for sentence completion, especially GPT2, are resource-intensive. Running these models requires substantial computational power and memory, which might not be available on standard consumer hardware. Users with limited resources might experience slower performance and potential system instability.

## Hardware Requirements

To achieve optimal performance, the following hardware specifications are recommended:

- **CPU**: A multi-core processor with a high clock speed (e.g., Intel i7 or AMD Ryzen 7 and above) is recommended to handle the computational load.

- **GPU**: A high-performance GPU (e.g., NVIDIA RTX 3080 or equivalent) is essential for accelerating model inference, particularly for Whisper and T5 models. CUDA compatibility is necessary for leveraging GPU capabilities.

- **RAM**: At least 16 GB of RAM is recommended, with 32 GB or more preferred, especially for processing longer audio files and running multiple processes simultaneously.

- **Storage**: An SSD with sufficient storage (at least 500 GB) is recommended for faster data access and handling large datasets, audio files, and model weights.

## Future Scope
- **Optimization for Real-Time Processing**: Enhancing the models and implementation to support real-time sentence completion can open up new use cases, such as live video calls and interactive applications.

- **Integration with Other Platforms**: Expanding the project's integration capabilities with other platforms and services, such as video conferencing tools, content management systems, and educational platforms, can enhance its utility and reach a broader audience.

- **Support for Additional Languages**: Extending support to multiple languages for both transcription and grammar correction can significantly broaden the application's applicability, catering to a global user base.
