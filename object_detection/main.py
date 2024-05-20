from fastapi import FastAPI
from fastapi.responses  import FileResponse, RedirectResponse, Response, JSONResponse
from typing import List
from pydantic import BaseModel
import os
import cv2
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch



app = FastAPI(title="Image caption generator API", description="An API for generation captions for images")

folder = 'C:\\Users\\aryan\\OneDrive\\Desktop\\object_detection\\photos'
processor = AutoProcessor.from_pretrained("arian2502/blip-icb-finetuned")
model = BlipForConditionalGeneration.from_pretrained("arian2502/blip-icb-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


class ImageCaption(BaseModel):
    caption:str

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
if not cap.isOpened():
    print("Error: Could not open webcam.")

@app.post("/capture_photo/")
async def capture_photo():
    global cap

    ret, frame = cap.read()
    if not ret:
        return {"error": "Failed to capture image."}

    filename = os.path.join(folder, 'photo.jpg')
    cv2.imwrite(filename, frame)

    return FileResponse(filename)


@app.post("/generate_caption/")
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
