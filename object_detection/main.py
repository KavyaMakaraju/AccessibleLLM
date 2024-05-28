from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.responses  import FileResponse, RedirectResponse, Response, JSONResponse
from pydantic import BaseModel
import os
import cv2
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
from gtts import gTTS
from playsound import playsound


app = FastAPI(title="Image caption generator API", description="An API for generation captions for images")

folder = 'C:\\Users\\aryan\\OneDrive\\Desktop\\object_detection\\dataset\\images'
processor = AutoProcessor.from_pretrained("arian2502/blip-icb-finetuned")
model = BlipForConditionalGeneration.from_pretrained("arian2502/blip-icb-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.get("/", include_in_schema=False )
def index():
    return RedirectResponse(url="/docs")
    

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    file_path = "path/to/your/favicon.ico"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return Response(status_code=204)  

if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)

@app.on_event("startup")
async def startup_event():
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

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

@app.post("/start_capture/")
async def start_capture(background_tasks: BackgroundTasks):
    background_tasks.add_task(capture_photos)
    return JSONResponse(content={"message": "Started capturing photos. Press 'c' to capture a photo and 'q' to quit."})

@app.get("/list_photos/")
async def list_photos():
    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    if not image_files:
        return JSONResponse(status_code=404, content={"message": "No photos found in the specified folder."})
    
    return {"photos": image_files}


class PhotoSelection(BaseModel):
    filename: str


@app.post("/select_photo/")
async def display_photo(filename: str):
    file_path = os.path.join(folder, filename)
    print(f"Looking for file at: {file_path}")  # Debugging statement
    if not os.path.exists(file_path):
        print("Photo not found.")  # Debugging statement
        raise HTTPException(status_code=404, detail="Photo not found.")

    image = Image.open(file_path).convert('RGB')
    image = image.resize((596, 437))

     # Save the image temporarily
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    # Return the image along with the generated caption
    return FileResponse(temp_image_path, media_type="image/jpeg")


@app.post("/generate_caption/")
async def generate_caption(filename: str):
    file_path = os.path.join(folder, filename)
    print(f"Looking for file at: {file_path}")  # Debugging statement
    if not os.path.exists(file_path):
        print("Photo not found.")  # Debugging statement
        raise HTTPException(status_code=404, detail="Photo not found.")
    
    # Open and resize the image
    image = Image.open(file_path).convert('RGB')
    image = image.resize((596, 437))
    
    # Caption generation
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, min_length=10, temperature=0.7, repetition_penalty=1.2, num_beams=5)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()    

     # Text-to-speech conversion
    speech = gTTS(text=generated_text, lang='en')
    audio_file_path = "generated_text.mp3"
    speech.save(audio_file_path)

    # Play the generated audio
    playsound(audio_file_path)

    return {"caption": generated_text}

@app.on_event("shutdown")
async def shutdown_event():
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
