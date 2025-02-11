from collections import defaultdict
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from segmentation_models import get_preprocessing
import numpy as np

labels = [
    {
        "name": 'unlabeled',
        "category": 'void',
        "color": (0,  0,  0)
    },
    {
        "name": 'ego vehicle',
        "category": 'void',
        "color": (0,  0,  0)
    },
    {
        "name": 'rectification border',
        "category": 'void',
        "color": (0,  0,  0)
    },
    {
        "name": 'out of roi',
        "category": 'void',
        "color": (0,  0,  0)
    },
    {
        "name": 'static',
        "category": 'void',
        "color": (0,  0,  0)
    },
    {
        "name": 'dynamic',
        "category": 'void',
        "color": (111, 74,  0)
    },
    {
        "name": 'ground',
        "category": 'void',
        "color": (81,  0, 81)
    },
    {
        "name": 'road',
        "category": 'flat',
        "color": (128, 64,128)
    },
    {
        "name": 'sidewalk',
        "category": 'flat',
        "color": (244, 35,232)
    },
    {
        "name": 'parking',
        "category": 'flat',
        "color": (250,170,160)
    },
    {
        "name": 'rail track',
        "category": 'flat',
        "color": (230,150,140)
    },
    {
        "name": 'building',
        "category": 'construction',
        "color": (70, 70, 70)
    },
    {
        "name": 'wall',
        "category": 'construction',
        "color": (102,102,156)
    },
    {
        "name": 'fence',
        "category": 'construction',
        "color": (190,153,153)
    },
    {
        "name": 'guard rail',
        "category": 'construction',
        "color": (180,165,180)
    },
    {
        "name": 'bridge',
        "category": 'construction',
        "color": (150,100,100)
    },
    {
        "name": 'tunnel',
        "category": 'construction',
        "color": (150,120, 90)
    },
    {
        "name": 'pole',
        "category": 'object',
        "color": (153,153,153)
    },
    {
        "name": 'polegroup',
        "category": 'object',
        "color": (153,153,153)
    },
    {
        "name": 'traffic light',
        "category": 'object',
        "color": (250,170, 30)
    },
    {
        "name": 'traffic sign',
        "category": 'object',
        "color": (220,220, 0)
    },
    {
        "name": 'vegetation',
        "category": 'nature',
        "color": (107,142, 35)
    },
    {
        "name": 'terrain',
        "category": 'nature',
        "color": (152,251,152)
    },
    {
        "name": 'sky',
        "category": 'sky',
        "color": (70,130,180)
    },
    {
        "name": 'person',
        "category": 'human',
        "color": (220, 20, 60)
    },
    {
        "name": 'rider',
        "category": 'human',
        "color": (255,  0,  0)
    },
    {
        "name": 'car',
        "category": 'vehicle',
        "color": ( 0,  0,142)
    },
    {
        "name": 'truck',
        "category": 'vehicle',
        "color": (0,  0, 70)
    },
    {
        "name": 'bus',
        "category": 'vehicle',
        "color": (0, 60,100)
    },
    {
        "name": 'caravan',
        "category": 'vehicle',
        "color": (0, 0, 90)
    },
    {
        "name": 'trailer',
        "category": 'vehicle',
        "color": (0, 0, 110)
    },
    {
        "name": 'train',
        "category": 'vehicle',
        "color": (0, 0, 230)
    },
    {
        "name": 'motorcycle',
        "category": 'vehicle',
        "color": (0, 80,100)
    },
    {
        "name": 'bicycle',
        "category": 'vehicle',
        "color": (119, 11, 32)
    },
        {
        "name": 'license plate',
        "category": 'vehicle',
        "color": (0,  0,142)
    }
]

# Create a mapping between a color and a category id
category_name_to_id = {}
color_to_category_id = {}
category_id_to_colors = defaultdict(set)
next_cat_id = 0
for label_data in labels:
    category_name = label_data["category"]
    color = label_data["color"]
    if category_name in category_name_to_id:
        cat_id = category_name_to_id[category_name]
    else:
        cat_id = next_cat_id
        category_name_to_id[category_name] = cat_id
        next_cat_id += 1
    color_to_category_id[color] = cat_id
    category_id_to_colors[cat_id].add(color)


def preprocess_mask(mask_path, target_size, num_classes):
    """
    Pré-traite un masque de segmentation en une représentation one-hot encodée à partir d'une carte de catégories.

    Args:
        mask_path (str): Chemin du fichier masque (image avec des couleurs représentant les classes).
        target_size (tuple): Taille cible pour le redimensionnement du masque, donnée sous la forme (largeur, hauteur).
        num_classes (int): Nombre total de classes dans le dataset.
        color_to_category_id (dict): Dictionnaire mappant chaque couleur RGB (tuple) à un identifiant de catégorie (int).

    Returns:
        np.ndarray: Masque de segmentation one-hot encodé de forme (H, W, num_classes),
                    où H et W sont les dimensions redimensionnées spécifiées par `target_size`.
    """
    # Chargement du masque avec Keras et redimensionnement
    mask = load_img(mask_path, target_size=target_size)
    mask = img_to_array(mask)  # Convertir l'image PIL en tableau numpy de forme (H, W, 3)

    # Création d'une carte de catégories vide (même taille que l'image)
    category_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    # Conversion des couleurs en indices de catégories
    for color, category_id in color_to_category_id.items():
        # Créer un masque booléen pour chaque couleur
        color_mask = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        # Assigner l'identifiant de catégorie
        category_map[color_mask] = category_id

    # Encodage one-hot
    one_hot_mask = np.eye(num_classes)[category_map]  # Création d'un tableau one-hot
    return one_hot_mask.astype(np.float32)  # Retourner un tableau numpy avec des float32

def preprocess_image_func(model):
    if model.backbone:
        return  get_preprocessing(model.backbone)
    return lambda img: img / 255.0
    