
# Gaze controlled keyboard 

This project is an eye-controlled virtual keyboard that allows users to type text using only their eyes. The system uses a webcam to detect the user's eye movements and blinks to determine which keys to press.

## Working procedure

- Look at the webcam.
- The system will detect your face and eyes.
- Look at the virtual keyboard displayed on the screen.
- Blink till the sound plays to select a key.
- The selected key will be typed on the virtual board.
- To switch between the two keyboards, gaze at the left or right side of the screen.
- To exit, press the Esc key.

## Demo Video

https://vimeo.com/951838236?share=copy

## Scope of the project

- A fully functional virtual keyboard application using eye blinks for key selections.
- Integrated large language model for real-time text prediction.
- Deployment on an app specifically targeted for specially abled people.
- Can be deployed on other applications as an extended feature using the developed FAST API

## Features

- Blink-controlled virtual keyboard(uses blink of the eye followed by a selection sound for confirmation)
- Automatic switching between keyboards using gaze detection(left or right side of the screen)
- Virtual board for displaying typed text
- Future text predictions from a large language model using the typed text on the board as the input to the model.

## Limitations

- Their may be an issue regarding the camera indexing on various devices when in an ecosystem.
- The system may not work well in low-light conditions.
- The system may not work well with glasses or other eye obstructions.


# Tech Stack
#### Language Used:
- Python

#### Computer Vision Library: 
- OpenCV

#### Other Python Libraries Used: 
- Numpy
- math
- time
- pyglet

#### Large Language Model 
- GPT-2
- LLAMA (currently working)

#### Dataset Used
- shape_predictor_68_face_landmarks.dat 

#### Hardware Requirements
- OS: Windows 10+ / Linux 8+ / Mac10+
- RAM:8GB RAM Minimum



## Installations Required
Install the various python libraries and frameworks required for running the project locally
```bash
  pip install python3
  pip install transformers
  pip install numpy
  pip install dlib
  pip install fastapi
  pip install opencv-python
  pip install pyglet
  pip install streamlit
```
    

# Accuracy Metrics
The accuracy analysis was done by taking input for the llm model from the input_text.txt file and the generated text from the llm for each corresponding line of input was written into the generated_text.txt file.

Then, the generated text in the generated_text.txt file was compared with the ground truth in the validation_text.txt file.
## GPT-2 Model

## BERT Analysis Scores

| Line | Generated Sentence                 | Reference Sentences                                                                 | Precision | F1-score | Recall |
|------|------------------------------------|-------------------------------------------------------------------------------------|-----------|----------|--------|
| 1    | How are you doing?                 | feeling now; finding your new job?; travelling?; doing?                             | 0.4482    | 0.4070   | 0.3726 |
| 2    | What are you doing there           | doing over the weekend?; planning now?; working on?                                 | 0.3161    | 0.2973   | 0.2806 |
| 3    | What are your plans for 2017       | for the weekend; after studies; post graduation; for the future                     | 0.3712    | 0.3755   | 0.3798 |
| 4    | Tomorrow is a good                 | another day; a new start;                                                           | 0.4325    | 0.4178   | 0.4040 |
| 5    | Where are you from                 | we; going; you; planning;                                                           | 0.3493    | 0.3305   | 0.3137 |
| 6    | How many people were going         | are attending; will attend; the event; are coming; have applied;                    | 0.3841    | 0.3708   | 0.3584 |
| 7    | How many cars and trucks           | do you have; do you drive; do you own;                                              | 0.4211    | 0.4177   | 0.4142 |
| 8    | Do you think this is a             | will work out; can work out; is a fair approach; is a good idea?                    | 0.3808    | 0.3694   | 0.3587 |
| 9    | How many problems are we           | have you solved?; can you solve?; did you overcome?                                 | 0.4261    | 0.4288   | 0.4316 |
| 10   | The sun was setting                | rises in the east; sets in the west; is a star; is a source of energy; is shining;  | 0.4091    | 0.3722   | 0.3414 |

| Metric                  | Average |
|-------------------------|---------|
| Average Precision       | 0.3939  |
| Average F1-score        | 0.3787  |
| Average Recall          | 0.3655  |

## ROUGE Score Analysis

| Line | ROUGE-1 (r, p, f)                         | ROUGE-2 (r, p, f)                         | ROUGE-L (r, p, f)                         |
|------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| 1    | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            |
| 2    | {'r': 0.125, 'p': 0.2, 'f': 0.1538}       | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.125, 'p': 0.2, 'f': 0.1538}       |
| 3    | {'r': 0.125, 'p': 0.167, 'f': 0.1429}     | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.125, 'p': 0.167, 'f': 0.1429}     |
| 4    | {'r': 0.2, 'p': 0.25, 'f': 0.2222}        | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.2, 'p': 0.25, 'f': 0.2222}        |
| 5    | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            |
| 6    | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            |
| 7    | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            |
| 8    | {'r': 0.182, 'p': 0.333, 'f': 0.2353}     | {'r': 0.0833, 'p': 0.2, 'f': 0.1176}      | {'r': 0.182, 'p': 0.333, 'f': 0.2353}     |
| 9    | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            |
| 10   | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            | {'r': 0.0, 'p': 0.0, 'f': 0.0}            |

| ROUGE Metric | Average Score |
|--------------|---------------|
| ROUGE-1      | {'r': 0.0632, 'p': 0.095, 'f': 0.0754} |
| ROUGE-2      | {'r': 0.0083, 'p': 0.02, 'f': 0.0118}  |
| ROUGE-L      | {'r': 0.0632, 'p': 0.095, 'f': 0.0754} |




## API Reference

#### Framework Used: FAST API

### Start the Keyboard

```http
  POST /start
```

```bash
# Start Keyboard endpoint
@app.post("/start")
def start_keyboard():
    global keyboard_thread, stop_keyboard_flag
    if keyboard_thread is None or not keyboard_thread.is_alive():
        stop_keyboard_flag.clear()  # Clear the flag to allow the thread to run
        keyboard_thread = threading.Thread(target=run_keyboard)
        keyboard_thread.start()
        return {"message": "Keyboard started"}
    else:
        return {"message": "Keyboard already running"}


```

| Description                |
 | :------------------------- |
 | Starts the keyboard, opens up the opencv boards for the keyboard, text board and the blink detection .|

| Message                |
 | :------------------------- |
 | 'Keyboard Started'|

### Get the text typed on the keyboard

```http
  GET / text
```
```bash
@app.get("/text")
def get_text():
    return text_data
```

| Description                |
 | :------------------------- |
 | Returns the text that is being typed on the openCV board which is then sent as an input to the large language model for generating the text predictions .|

| Message                |
 | :------------------------- |
 | 'WHY', as an example from the demo video|

### Text generated by the LLM Model  

```http
  POST / generate-text
```
```bash
  @app.post("/generate-text")
def generate_text():
    text = text_data["text"]

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Set generation parameters
    max_new_tokens = 2  # Maximum number of tokens in the generated text
    temperature = 0.8  # Controls the randomness of the predictions
    top_k = 50  # Controls the diversity of the predictions
    num_return_sequences = 10 # Number of sequences to generate, in this case, 5 different predictions
    do_sample = True  # Enable sampling

    # Generate text based on the input
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,  # End-of-sequence token
        do_sample=do_sample  # Enable sampling
    )

    # Decode the generated texts
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Write the generated texts to a separate file
    output_filename = "generated_texts.txt"
    with open(output_filename, "w", encoding="utf-8") as file:
        for generated_text in generated_texts:
            file.write(generated_text + "\n")

    # Return the generated texts as a response
    return {"message": "Text generation complete. Check 'generated_texts.txt' for the output."}

```

| Description                |
 | :------------------------- |
 | Text that is generated by the large language model depending upon the text that was passed as input using openCV .|

| Message                |
 | :------------------------- |
 | 'WHY IS IT', as an example from the demo video|

### STOP the keyboard 

 ```http
  POST / stop
```
```bash
@app.post("/stop")
def stop_keyboard():
    global keyboard_thread, stop_keyboard_flag
    if keyboard_thread is not None and keyboard_thread.is_alive():
        stop_keyboard_flag.set()  # Signal the keyboard thread to stop
        keyboard_thread.join()  # Wait for the thread to finish
        shutdown_server()  # Shut down the FastAPI server
        return {"message": "Keyboard stopped"}
    else:
        shutdown_server()  # Shut down the FastAPI server even if the keyboard is not running
        return {"message": "Keyboard is not running"}

```

| Description                |
 | :------------------------- |
 | Terminates the entire functionality of the keyboard.|

| Message                |
 | :------------------------- |
 | 'Keyboard Stopped.'|



## Run Locally

To deploy this project localy, download all the code files and run the following commands

```bash
  python main2.py
```

Then run the app by using the command 

```bash
  streamlit run app.py
```



## FAQs

#### What problems does the project aim to solve?

The virtual keyboard enabled with blink detection for typing aims to solve the following problems:
- Specially abled people who face problems in typing due to motory impairments can utilise the virtual keyboard for their input tasks.
- Another class of specially abled people with speech impairments who face peoblems in utilising the feature of voice commands can utilise the virtual keyboard as well as a mode of input.

#### How does the eye blink detection work?

The eye blink detection works by using dlib's facial landmark predictor to identify the positions of the eyes. By analyzing the distance between specific points on the eyes(gaze ratio), the software can determine if the eyes are closed (blinking) or open.

#### How is the text prediction generated?
The text prediction is generated using a large language model from the transformers library. Once the user types some text, it is sent to the model, which then generates possible future text predictions based on the input.

#### Which LLM Model is currently being used for the project?
GPT-2

#### What technologies are used to create this virtual keyboard?
This virtual keyboard is created using OpenCV for computer vision tasks, dlib for facial landmark detection, and transformers for text prediction using a large language model(GPT-2). It also uses FastAPI for creating APIs and Streamlit for the user interface.


# Author

- [@Spandan02](https://github.com/Spandan02)


## Feedback

If you have any feedback, please do reach out at spandanpradhan02@gmail.com

