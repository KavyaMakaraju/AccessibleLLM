# LLM Models Documentation

## Introduction
This documentation outlines the implementation and performance evaluation of two Language Model (LLM) models: BERT and HappyTextToText (T5).

### BERT Model
- *Training*: The BERT model was fine-tuned on a dataset Grammer-dataset in datasets zip file.
- *Loss*: During training, the loss achieved was approximately 0.35.
- *Performance*: The model performs well for sentences present in the training dataset.
- *Limitation*: However, it shows limitations when presented with string inputs not present in the dataset.

### HappyTextToText (T5) Model
- *Model*: Utilizes the T5 architecture, specifically the pretrained model "vennify/t5-base-grammar-correction".
- *Metric*: Evaluation of this model was conducted using the BLEU score, with a recorded value of 0.796.
- *Notebook*: The implementation notebook for this model is saved as "T5-Model.ipynb".

###  Working On
- The T5 based model  "vennify/t5-base-grammar-correction" can also be finetuned, trying to finetune for better performance.
