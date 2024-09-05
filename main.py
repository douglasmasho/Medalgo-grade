from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Compose, ScaleIntensity, Resize, ToTensor
from monai.networks.nets import DenseNet121
import logging
from io import BytesIO
# import asyncio


# Set event loop policy for Windows
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define transformations
transforms = Compose([
    ScaleIntensity(),
    Resize((128, 128)),
    ToTensor()
])

# Load the trained model
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
model.load_state_dict(torch.load('tumorgrade.pth', map_location=torch.device('cpu')))
model.eval()

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict_mri_image(img):
    # Apply transformations
    img = transforms(img)

    # Add batch dimension
    img = img.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return 'HGG' if predicted.item() == 1 else 'LGG'

@app.get("/")
async def health_check():
    return "The health check is successful!"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Log the received file
        logger.info(f"Received file: {file.filename}")
        
        # Load the uploaded file into memory
        file_data = await file.read()

        # Convert the file data into a BytesIO object and load it using nibabel
        file_stream = BytesIO(file_data)
        img = nib.Nifti1Image.from_bytes(file_stream.read())

        # Log shape of the loaded image
        img_data = img.get_fdata()
        logger.info(f"Image shape: {img_data.shape}")

        # Choose a random slice along the axial plane (e.g., middle slice)
        slice_index = img_data.shape[2] // 2
        image_slice = img_data[:, :, slice_index]

        # Add channel dimension for grayscale
        image_slice = np.expand_dims(image_slice, axis=0)

        # Convert to numpy array
        image_slice = np.array(image_slice)

        # Log the shape of the image slice
        logger.info(f"Image slice shape: {image_slice.shape}")

        # Predict the class
        prediction = predict_mri_image(image_slice)
        logger.info(f"Prediction: {prediction}")
        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        # Log the error
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
