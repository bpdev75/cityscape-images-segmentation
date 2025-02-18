import sys
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from keras.utils import img_to_array
from PIL import Image
import io
import base64
from src.models.unet_models import build_unet_mini
from src.preprocessing.preprocessing import category_name_to_id, category_id_to_colors
from src.predictions.predictions import predict_mask

target_size = (256, 256)
input_shape = (target_size[0], target_size[1], 3)
num_classes = len(category_name_to_id)

API_VERSION = "1.0"

app = FastAPI()

# Chargement du model
model = build_unet_mini(input_shape, num_classes)
checkpoint_path = f"weights/model_{model.name}_best_weights.keras"
model.load_weights(checkpoint_path)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint qui prend une image en entrée et retourne un masque de segmentation.
    """
    try:
        # Lire l'image depuis les bytes avec PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        original_size = image.size

        # Redimensionner si nécessaire
        image = image.resize(target_size)

        # Convertir en tableau numpy
        image_array = img_to_array(image)

        # Prédiction
        mask = predict_mask(model, image_array, category_id_to_colors)

        # Convertir en image PNG
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.resize(original_size)
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format="PNG")
        
        # Encoder l'image en base64
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        # Réponse JSON
        response_data = {
            "api_version": API_VERSION,
            "model_name": model.name,
            "image_base64": img_base64
        }

        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "exception": str(e)
            }
        )
