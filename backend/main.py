from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model import predict_breed
import shutil
import os
import uuid
import logging
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure maximum upload size (20MB)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
registered_animals = []

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Received prediction request for file: {image.filename}")
        
        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        ext = image.filename.split('.')[-1].lower()
        if ext not in allowed_extensions:
            logger.warning(f"Invalid file type: {ext}")
            return JSONResponse(
                content={"error": f"File type not allowed. Must be one of: {', '.join(allowed_extensions)}"},
                status_code=400
            )
            
        # Check file size (20MB limit)
        max_size = 20 * 1024 * 1024  # 20MB in bytes
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        # Read file in chunks to get size
        while chunk := await image.read(chunk_size):
            file_size += len(chunk)
            if file_size > max_size:
                await image.seek(0)  # Reset file pointer
                logger.warning(f"File too large: {file_size} bytes")
                return JSONResponse(
                    content={"error": "File size exceeds 20MB limit"},
                    status_code=400
                )
        
        # Reset file pointer for later use
        await image.seek(0)
        
        # Save the uploaded file
        file_id = str(uuid.uuid4())
        file_location = f"{UPLOAD_FOLDER}/{file_id}.{ext}"
        
        try:
            with open(file_location, "wb") as f:
                shutil.copyfileobj(image.file, f)
            logger.info(f"File saved successfully to: {file_location}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return JSONResponse(
                content={"error": "Failed to save uploaded file"},
                status_code=500
            )
        
        # Get predictions
        try:
            top_breed, top_conf, top3 = predict_breed(file_location)
            logger.info(f"Predictions generated: {top3}")
            
            response_data = {
                "breed": top_breed,
                "confidence": top_conf,
                "predictions": top3,
                "image_path": file_location
            }
            return JSONResponse(content=response_data)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return JSONResponse(
                content={"error": "Failed to generate predictions"},
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            content={"error": "An unexpected error occurred"},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error processing image: {str(e)}"},
            status_code=500
        )

@app.post("/register")
async def register_animal(payload: dict):
    breed = payload.get("breed")
    confidence = payload.get("confidence", 0)
    image_path = payload.get("image_path", "N/A")
    if not breed:
        return JSONResponse(content={"error": "Breed required"}, status_code=400)
    
    animal_id = str(uuid.uuid4())
    registered_animals.append({
        "id": animal_id,
        "breed": breed,
        "confidence": confidence,
        "image_path": image_path
    })
    return JSONResponse(content={"message": "Animal registered successfully", "id": animal_id})

@app.get("/health")
def health():
    return {"status": "ok"}
