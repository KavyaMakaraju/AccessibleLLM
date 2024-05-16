*Model Documentation*

---

*Model 1*

- *Model Details:* facebook/wav2vec2-base-960h.
- *Input:* Audio files present in "dataset/Audio-to-text-whisper-dataset/cv-other-train".
- *Output:* Text transcrptions saved in "results/wav2vec2-base960h-results.csv" and "results/wav2vec2-base960h-2ndtime.csv".
- *Time taken for model to train:* 1 hour for 250 audio files.
- *Saved Model Location:* The model is saved in the directory named "saved_wav2vec2_model".
- *Metrics:*
  - Accuracy: 34.26%
  - Precision: 35.06%
  - Recall: 34.26%
  - F1 Score: 34.53%

---

*Model 2*

- *Model Details:* facebook/wav2vec2-large-960h.
- *Input:* Audio files present in "dataset/Audio-to-text-whisper-dataset/cv-other-train".
- *Output:* Text transcrptions saved in "results/wav2vec2-large960h-results.csv".
- *Time taken for model to train:* 50 minutes for 250 audio files.
- *Saved Model Location:* The model is saved in the directory "wav2vec2-large-960h".
- *Metrics:*
  - Average Accuracy: 57.71%
  - Average Precision: 58.08%
  - Average Recall: 57.71%
  - Average F1-score: 57.29%
  - Average Word Error Rate (WER): 28.85%
  - Average Character Error Rate (CER): 14.63%

---

*Model 3*

- *Model Details:* facebook/wav2vec2-large-960h-lv60-self.
- *Input:* Audio files present in "dataset/Audio-to-text-whisper-dataset/cv-other-train".
- *Output:* Text transcrptions saved in "results/wav2vec2-large-960h-lv60-self-results.csv".
- *Time taken for model to train:* 50 minutes for 250 audio files.
- *Saved Model Location:* The model is saved in "wav2vec2-large-960h-lv60-self".
- *Metrics:*
  - Accuracy: 49.40%
  - Precision: 50.20%
  - Recall: 49.40%
  - F1 Score: 49.67%

---

*Model 4*

- *Model Type:* Speech Recognition Model.
- *Input:* Audio files present in "dataset/Audio-to-text-whisper-dataset/cv-other-train".
- *Output:* Text transcrptions displayed in speechrecognition-model4 notebook.
- *Time taken for model to train:* 2 hours for 250 audio files.
- *Metrics:*
  - Total files: 246, Correctly transcribed: 125
  - Accuracy: 50.81%
  - Precision: 96.00%
  - Recall: 51.00%
  - F1 Score: 66.00%

---

*Model 5*

- *Model Type:* Whisper Model
- *Input:* Audio files present in "dataset/Audio-to-text-whisper-dataset/cv-other-train".
- *Output:* Text transcrptions saved to "results/whisper-model-results.csv".
- *Time taken for model to train:* 2 hours for 250 audio files.
- *Metrics:*
  - Overall Accuracy: 93.40%
  - Overall Precision: 94.05%
  - Overall Recall: 93.40%
  - Overall F1-score: 93.39%
  - Overall Word Error Rate (WER): 0.30%
  - Overall Character Error Rate (CER): 0.14%

---

Note: For Model 2, individual metrics are noted in the file "wav2vec2-base960h-2ndtime.xlsx". For Model 5, individual metrics are noted in "whisper-model-results.xlsx".
