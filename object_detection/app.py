import os
import cv2
import streamlit as st
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS
import base64

# Initialize the Streamlit app
st.title("Image Caption Generator")
st.write("An app for generating captions for images")

# Define folder for saving images
folder = 'C:\\Users\\aryan\\OneDrive\\Desktop\\object_detection\\dataset\\images'
if not os.path.exists(folder):
    os.makedirs(folder)

# Load the model and processor
processor = AutoProcessor.from_pretrained("arian2502/blip-icb-finetuned")
model = BlipForConditionalGeneration.from_pretrained("arian2502/blip-icb-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


cap = cv2.VideoCapture(0)
# Capture photos using webcam
def capture_photos():
    print("Press 'c' to capture a photo and 'q' to quit.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Wait for key event
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Determine the next filename based on existing files in the folder
            existing_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            next_number = len(existing_files) + 1
            filename = os.path.join(folder, f'{next_number}.jpg')
            
            cv2.imwrite(filename, frame)
            print(f"Photo captured and saved as {filename}")

        elif key == ord('q'):
            print("Quitting...")
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Sidebar options
option = st.sidebar.selectbox(
    'Choose an action:',
    ['Capture Photo', 'List Photos', 'Generate Caption']
)

if option == 'Capture Photo':
    if st.button('Start Capture'):
        capture_photos()

elif option == 'List Photos':
    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    if not image_files:
        st.write("No photos found in the specified folder.")
    else:
        st.write("Photos:")
        for image_file in image_files:
            st.write(image_file)

elif option == 'Generate Caption':
    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    if not image_files:
        st.write("No photos found in the specified folder.")
    else:
        selected_photo = st.selectbox('Select a photo:', image_files)
        if st.button('Generate Caption'):
            file_path = os.path.join(folder, selected_photo)
            if not os.path.exists(file_path):
                st.write("Photo not found.")
            else:
                image = Image.open(file_path).convert('RGB')
                image = image.resize((596, 437))
                st.image(image, caption="Selected Photo")

                inputs = processor(images=image, return_tensors="pt").to(device)
                generated_ids = model.generate(**inputs, min_length=10, temperature=0.7, repetition_penalty=1.2, num_beams=5)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                st.write("Generated Caption: ", generated_text)

                speech = gTTS(text=generated_text, lang='en')
                audio_file_path = "generated_text.mp3"
                speech.save(audio_file_path)

                with open(audio_file_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')

                # Optionally provide a download link for the audio file
                b64 = base64.b64encode(audio_bytes).decode()
                href = f'<a href="data:audio/mp3;base64,{b64}" download="generated_text.mp3">Download MP3</a>'
                st.markdown(href, unsafe_allow_html=True)
