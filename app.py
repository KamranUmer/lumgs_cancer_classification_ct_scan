import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from model.model1 import predict_image_class, train_generator  # Import functions from model.py

# Initialize FastAPI app
app = FastAPI()

# Define paths and settings
UPLOAD_FOLDER = "uploads"
PREDICTED_FOLDER = "static/predicted_images"
MODEL_PATH = "trained_lung_cancer_model.h5"
CLASS_LABELS = list(train_generator.class_indices.keys())

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

# Static file serving and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Save the uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Predict and save the result using `predict_image_class`
    predicted_label, confidence_score = predict_image_class(
        img_path=file_path,
        model_path=MODEL_PATH,
        class_labels=CLASS_LABELS,
        save_directory=PREDICTED_FOLDER,
    )

    # Predicted image path (served through /static)
    predicted_image_path = f"/static/predicted_images/{os.path.basename(file_path)}"

    # Return results
    return {
        "label": predicted_label,
        "confidence": confidence_score,
        "predicted_image_url": predicted_image_path,
    }
