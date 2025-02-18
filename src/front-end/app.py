import streamlit as st
import requests
import base64
import io
from PIL import Image

# Configuration de l'API
API_URL = "https://cityscape-segmentation-api-cb52023cf02a.herokuapp.com/predict"

st.set_page_config(layout="wide")

# Interface Streamlit
st.title("Segmentation d'image avec VGG16-UNet")

# Upload de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Créer deux colonnes pour afficher les images côte à côte
    col1, col2 = st.columns(2)

    # Affichage de l'image originale
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Image originale", use_column_width=True)

    # Conversion en base64 pour l'envoi
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Envoi à l'API
    with st.spinner("Analyse en cours..."):
        # Envoyer l'image à l'API en tant que fichier binaire
        files = {"file": uploaded_file.getvalue()}  # Envoi en binaire
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()
        mask_base64 = data.get("image_base64", "")

        if mask_base64:
            # Décodage du masque reçu
            mask_data = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_data))

            # Affichage du masque
            with col2:
                st.image(mask_image, caption="Masque de segmentation", use_column_width=True)
        else:
            st.error("Erreur : L'API n'a pas retourné de masque valide.")
    else:
        st.error(f"Erreur API : {response.status_code}")