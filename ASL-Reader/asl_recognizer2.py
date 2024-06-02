import cv2
import sys
import numpy as np
import keras
from keras.models import load_model
from spellchecker import SpellChecker
from efficientnet.tfkeras import EfficientNetB0
import requests

def nothing(x):
    pass


def get_class_label(val, dictionary):
    """
    Function returns the key (Letter: a/b/c/...) value from the alphabet dictionary
    based on its class index (1/2/3/...)
    """
    for key, value in dictionary.items():
        if value == val:
            return key


model = load_model('./efficient_model.h5')
spell = SpellChecker()

# create alphabet dictionary to label the letters {'a':1, ..., 'nothing':29}
alphabet = {chr(i+96).upper():i for i in range(1,27)}
#alphabet['del'] = 27
#alphabet['nothing'] = 28
#alphabet['space'] = 29

video_capture = cv2.VideoCapture(0)
cv2.namedWindow('Model Image')

# set the ration of main video screen
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# set track bar of threshold values for Canny edge detection
# more on Canny edge detection here:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
#cv2.createTrackbar('lower_threshold', 'Model Image', 0, 255, nothing)
#cv2.createTrackbar('upper_threshold', 'Model Image', 0, 255, nothing)
#cv2.setTrackbarPos('lower_threshold', 'Model Image', 100)
#cv2.setTrackbarPos('upper_threshold', 'Model Image', 0)

# VARIABLES INITIALIZATION
# THRESHOLD - ratio of the same letter in the last N_FRAMES predicted letters
THRESHOLD = 0.85
N_FRAMES = 25

IMG_SIZE = 224
SENTENCE = '' # string that will hold the final output
letter = '' # temporary letter
LETTERS = np.array([], dtype='object') # array with predicted letters
generated_text="hi"
START = False # start/pause controller
# supportive text
description_text_1 = "Press 'S' to Start/Pause gesture recognition."
description_text_2 = "Press 'D' to erase the output. "
description_text_3 = "Press 'Q' to quit."
def softmax(x):
    return(np.exp(x)/np.exp(x).sum())
  
  
def get_completion(text):
  # Define the payload
	payload = {
			"input_text": text
	}
	response = requests.post("http://localhost:8000/generate-text", json=payload)
	if response.status_code == 200:
		# Parse the response JSON
		response_data = response.json()
		generated_text = response_data.get("generated_text")
		return generated_text
	else:
		print(f"Request failed with status code {response.status_code}")   
		return "Error"
while True:
    blank_image = np.zeros((100,800,3), np.uint8) # black image for the output
    ret, frame = video_capture.read() # capture frame-by-frame
    # set the corners for the square to initialize the model picture frame
    x_0 = int(frame.shape[1] * 0.1)
    y_0 = int(frame.shape[0] * 0.25)
    x_1 = int(x_0 + 200)
    y_1 = int(y_0 + 200)

    # MODEL IMAGE INITIALIZATION
    hand = frame.copy()[y_0:y_1, x_0:x_1] # crop model image
    resized_image = cv2.resize(hand, (224,224))
    
    model_image = np.array(resized_image)
    model_image = model_image.astype('float32') / 255.0

    try:
        model_image = model_image.reshape(1, IMG_SIZE, IMG_SIZE, 3)
        predict = model.predict(model_image)
        #predict = softmax(predict)
        print(predict)
        for values in predict:
            if np.all(values < 0.5):
                # if probability of each class is less than .5 return a message
                letter = 'Cannot classify:('
            else:
                predict = np.argmax(predict, axis=1) + 1
                letter = get_class_label(predict, alphabet)
                LETTERS = np.append(LETTERS, letter)
    except:
        pass


    if START == True:
        # append the final output with the letter
        if (np.mean(LETTERS[-N_FRAMES:] == letter) >= THRESHOLD) & (len(LETTERS) >= N_FRAMES):
            if letter == 'space':
                SENTENCE = SENTENCE[:-1] + ' ' + '_'
                LETTERS = np.array([], dtype='object')
            elif letter == 'del':
                SENTENCE = SENTENCE[:-2] + '_'
                LETTERS = np.array([], dtype='object')
            elif letter == 'nothing':
                pass
            else:
                SENTENCE = SENTENCE[:-1] + letter + '_'
                LETTERS = np.array([], dtype='object')
                
        if(len(SENTENCE)>2):
          generated_text = get_completion(SENTENCE)


				



    if START == False:
        paused_text = 'Paused'
    else:
        paused_text = ''

    # TEXT INITIALIZATION
    cv2.putText(
        img=frame,
        text=letter,
        org=(x_0+140,y_0+100),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(0,0,255),
        fontScale=1
    )
    cv2.putText(
        img=frame,
        text=generated_text,
        org=(x_0+160,y_0+150),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(0,0,255),
        fontScale=1
    )
    # paused text
    cv2.putText(
        img=frame,
        text=paused_text,
        org=(x_0+140,y_0+195),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(0,0,255),
        fontScale=1
    )

    # helper texts
    cv2.putText(
        img=frame,
        text=description_text_1,
        org=(10,440),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    cv2.putText(
        img=frame,
        text=description_text_2,
        org=(10,455),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    cv2.putText(
        img=frame,
        text=description_text_3,
        org=(10,470),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    cv2.putText(
        img=frame,
        text='Place your hand here:',
        org=(x_0-30,y_0-10),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        color=(255,255,255),
        fontScale=1
    )

    # current letter
    cv2.putText(
        img=frame,
        text=letter,
        org=(x_0+10,y_0+20),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    # final output
    cv2.putText(
        img=blank_image,
        text='Result: ' + SENTENCE,
        org=(10, 50),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        thickness=1,
        color=(0,0,255),
        fontScale=1
    )

    # draw rectangle for hand placement
    cv2.rectangle(frame, (x_0, y_0), (x_1, y_1), (0, 255, 0), 2)

    # display the resulting frames
    cv2.imshow('Main Image', frame)
    #cv2.imshow('Model Image', model_image)
    cv2.imshow('Output', blank_image)

    if cv2.waitKey(30) & 0xFF == ord('s'):
        START = not START

    if cv2.waitKey(30) & 0xFF == ord('d'):
        SENTENCE = ''

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# save the resulted string into the file
#text_file = open("Output.txt", "w")
#text_file.write("You said: %s" % SENTENCE)
#text_file.close()
