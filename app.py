from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import uvicorn

# Initialize FastAPI app (disable docs)
app = FastAPI()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
CLASS_NAMES = ["Good Claims", "Handwritten Claims", "Real Bad Claims"]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomApply(
        [transforms.RandomRotation(degrees=2)],
        p=0.2
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
def load_model(model_path: str = "bad_claim_classifier.pth"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Initialize model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    print(f"Model loaded on device: {device}")

# Pydantic model for file path input
class ImagePath(BaseModel):
    file_path: str

@app.post("/classify_stream")
async def classify_stream(file: UploadFile = File(...)):
    """
    Classify image from uploaded file (bytes/stream)
    """
    # Read and process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    
    # Return only predicted class
    predicted_class = CLASS_NAMES[predicted.item()]
    
    return {"predicted_class": predicted_class}

@app.post("/classify")
async def classify(data: ImagePath):
    """
    Classify image from file path
    """
    # Load image from file path
    image = Image.open(data.file_path).convert("RGB")
    
    # Preprocess
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    
    # Return only predicted class
    predicted_class = CLASS_NAMES[predicted.item()]
    
    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
