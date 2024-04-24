
# Documentation of the gaze controlled keyboard 

Eye-Controlled Virtual Keyboard

This project is an eye-controlled virtual keyboard that allows users to type text using only their eyes. The system uses a webcam to detect the user's eye movements and blinks to determine which keys to press.

Requirements :

Python 3.x

OpenCV 4.x

dlib 19.x

pyglet 1.5.x

A webcam
Installation

Install the required libraries using pip: pip install opencv-python dlib pyglet

Download the shape_predictor_68_face_landmarks.dat file from the dlib website and place it in the same directory as the script.

Run the script using Python: python eye_controlled_keyboard.py
Usage

Run the script and look at the webcam.
The system will detect your face and eyes.
Look at the virtual keyboard displayed on the screen.
Blink to select a key.
The selected key will be typed on the virtual board.
To switch between the two keyboards, gaze at the left or right side of the screen.
To exit, press the Esc key.

Features

Eye-controlled virtual keyboard
Two keyboard layouts: QWERTY and alphabetical
Automatic switching between keyboards using gaze detection
Blinking detection for key selection
Virtual board for displaying typed text

Known Issues

Their is an issue regarding the camera indexing on variousa devices when in an ecosystem.

The system may not work well in low-light conditions.

The system may not work well with glasses or other eye obstructions.