
# Gaze controlled keyboard 

This project is an eye-controlled virtual keyboard that allows users to type text using only their eyes. The system uses a webcam to detect the user's eye movements and blinks to determine which keys to press.

## Working procedure

Look at the webcam.

The system will detect your face and eyes.

Look at the virtual keyboard displayed on the screen.

Blink till the sound plays to select a key.

The selected key will be typed on the virtual board.

To switch between the two keyboards, gaze at the left or right side of the screen.

To exit, press the Esc key.

## Features

Eye-controlled virtual keyboard

Two keyboard layouts: QWERTY and alphabetical

Automatic switching between keyboards using gaze detection

Blink detection for key selection

Virtual board for displaying typed text

## Known Issues

Their is an issue regarding the camera indexing on variousa devices when in an ecosystem.

The system may not work well in low-light conditions.

The system may not work well with glasses or other eye obstructions.
# Tech Stack
### Language Used:
Python

### Computer Vision: 
OpenCV

### Python Libraries Used: 
Numpy, math, time

### Large Language Model 
GPT-2

LLAMA (currently working)

### DATASETS
shape_predictor_68_face_landmarks.dat 



# Documentation
The code files have been uploaded separately, as the development is being done locally.
## Accuracy Metrics
The accuracy analysis was done by taking input for the llm model from the input_text.txt file and the generated text from the llm for each corresponding line of input was written into the generated_text.txt file.

Then, the generated text in the generated_text.txt file was compared with the ground truth in the validation_text.txt file.
## GPT-2 Model



## BERT Analysis Score for the respective Model:
BERT Analysis Scores for each line:

#### Line 1:
Generated: How are you doing?

Reference: feeling now; finding your new job?; travelling?; doing?;

Precision: 0.4482
F1-score: 0.4070
Recall: 0.3726

#### Line 2:
Generated: What are you doing there

Reference: doing over the weekend?; planning now?; working on?;

Precision: 0.3161
F1-score: 0.2973
Recall: 0.2806

#### Line 3:
Generated: What are your plans for 2017

Reference: for the weekend; after studies; post graduation; for the future

Precision: 0.3712
F1-score: 0.3755
Recall: 0.3798

#### Line 4:
Generated: Tomorrow is a good

Reference: another day; a new start;

Precision: 0.4325
F1-score: 0.4178
Recall: 0.4040

#### Line 5:
Generated: Where are you from

Reference: we; going; you; planning;

Precision: 0.3493
F1-score: 0.3305
Recall: 0.3137

#### Line 6:
Generated: How many people were going

Reference: are attending; will attend; the event; are coming; have applied;

Precision: 0.3841
F1-score: 0.3708
Recall: 0.3584

#### Line 7:
Generated: How many cars and trucks

Reference: do you have; do you drive ; do you own ;

Precision: 0.4211
F1-score: 0.4177
Recall: 0.4142

#### Line 8:
Generated: Do you think this is a

Reference: will work out ; can work out ; is a fair approach; is a good idea?;

Precision: 0.3808
F1-score: 0.3694
Recall: 0.3587

#### Line 9:
Generated: How many problems are we

Reference: have you solved?; can you solve?; did you overcome?;

Precision: 0.4261
F1-score: 0.4288
Recall: 0.4316

#### Line 10:
Generated: The sun was setting

Reference: rises in the east; sets in the west; is a star; is a source of energy; is shining;

Precision: 0.4091
F1-score: 0.3722
Recall: 0.3414

#### Average BERT Analysis Score:
Average Precision: 0.3939

Average F1-score: 0.3787

Average Recall: 0.3655

## ROUGE Score Analysis

#### ROUGE Scores for Line 1:
{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}

#### ROUGE Scores for Line 2:
{'rouge-1': {'r': 0.125, 'p': 0.2, 'f': 0.1538461491124262}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.125, 'p': 0.2, 'f': 0.1538461491124262}}

#### ROUGE Scores for Line 3:
{'rouge-1': {'r': 0.125, 'p': 0.16666666666666666, 'f': 0.14285713795918387}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.125, 'p': 0.16666666666666666, 'f': 0.14285713795918387}}

#### ROUGE Scores for Line 4:
{'rouge-1': {'r': 0.2, 'p': 0.25, 'f': 0.22222221728395072}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.2, 'p': 0.25, 'f': 0.22222221728395072}}

#### ROUGE Scores for Line 5:
{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}

#### ROUGE Scores for Line 6:
{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}

#### ROUGE Scores for Line 7:
{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}

#### ROUGE Scores for Line 8:
{'rouge-1': {'r': 0.18181818181818182, 'p': 0.3333333333333333, 'f': 0.23529411307958487}, 'rouge-2': {'r': 0.08333333333333333, 'p': 0.2, 'f': 0.11764705467128042}, 'rouge-l': {'r': 0.18181818181818182, 'p': 0.3333333333333333, 'f': 0.23529411307958487}}

#### ROUGE Scores for Line 9:
{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}

#### ROUGE Scores for Line 10:
{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}

#### Average ROUGE Scores for All Lines:
{'rouge-1': {'r': 0.06318181818181819, 'p': 0.095, 'f': 0.07542196174351457}, 'rouge-2': {'r': 0.008333333333333333, 'p': 0.02, 'f': 0.011764705467128041}, 'rouge-l': {'r': 0.06318181818181819, 'p': 0.095, 'f': 0.07542196174351457}}






# Author

- [@Spandan02](https://github.com/Spandan02)


## Feedback

If you have any feedback, please reach out to us at spandanpradhan02@gmail.com

