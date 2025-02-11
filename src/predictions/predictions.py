import numpy as np
from src.preprocessing.preprocessing import preprocess_image_func

def predict_mask(model, image, category_id_to_colors):
    """
    Prédire le masque de segmentation d'une image en utilisant un modèle de segmentation.

    Cette fonction prend une image d'entrée, l'envoie au modèle de segmentation pour obtenir la prédiction du masque,
    puis convertit la prédiction en une image RGB où chaque classe est représentée par une couleur spécifique.

    Args:
        model (tensorflow.keras.Model): Le modèle de segmentation pré-entraîné utilisé pour prédire le masque.
        image (numpy.ndarray): L'image d'entrée sur laquelle effectuer la prédiction. Elle doit avoir la forme (hauteur, largeur, canaux).

    Returns:
        numpy.ndarray: L'image RGB résultant du masque prédit. Chaque pixel est coloré selon la classe prédite pour ce pixel.

    Example:
        rgb_mask = predict_mask(model, image)
        plt.imshow(rgb_mask)
        plt.show()

    Notes:
        - La fonction suppose que le modèle prédit des masques de segmentation avec un nombre de classes `num_classes` et utilise `softmax` en sortie.
        - Le dictionnaire `category_id_to_colors` doit être défini avant d'appeler cette fonction pour correspondre les classes avec les couleurs.
        - La fonction convertit l'image de sortie de la forme (hauteur, largeur, 1) à une image RGB (hauteur, largeur, 3).
    """
    image = preprocess_image_func(model)(image)
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour correspondre au batch_size

    # Prédiction du masque
    pred_mask = model.predict(image)

    pred_mask = np.argmax(pred_mask, axis=-1)  # On prend la classe prédite pour chaque pixel
    pred_mask = np.squeeze(pred_mask, axis=0)  # Enlever la dimension batch_size

    height, width = pred_mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Remplir l'image RGB en utilisant le mappage de couleurs
    for category_id, colors in category_id_to_colors.items():
        rgb_image[pred_mask == category_id] = list(colors)[0]

    return rgb_image