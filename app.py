import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from model.model1 import predict_image_class, train_generator  # Import the predict function and class labels from model.py
from requests import request

# Initialize the FastAPI app
app = FastAPI()

# Define paths and settings
UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded images
PREDICTED_FOLDER = 'predicted_images'  # Folder to save predicted images
MODEL_PATH = 'trained_lung_cancer_model.h5'  # Path to the trained model
CLASS_LABELS = list(train_generator.class_indices.keys())  # Class labels from the model

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file part in the request")

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Predict and save the result using the imported `predict_image_class` function
    predicted_label, confidence_score = predict_image_class(
        img_path=file_path,
        model_path=MODEL_PATH,
        class_labels=CLASS_LABELS,
        save_directory=PREDICTED_FOLDER
    )
    
    # Predicted image path
    predicted_image_path = os.path.join(PREDICTED_FOLDER, os.path.basename(file_path))
    
