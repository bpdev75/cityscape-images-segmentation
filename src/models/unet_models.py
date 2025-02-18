import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D, BatchNormalization, Activation, Concatenate, Input, Conv2DTranspose
from tensorflow.keras import Model

def conv_block(inputs, filters, pool=False):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x
    
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet_mini(input_shape, num_classes):
    """
    Crée un modèle de segmentation sémantique basé sur une version simplifiée de l'architecture UNet.

    Cette fonction définit un modèle UNet léger avec un nombre réduit de canaux et de convolutions,
    ce qui le rend plus rapide et moins exigeant en ressources que les versions complètes.

    Args:
        input_shape (tuple): La forme des images d'entrée au modèle (hauteur, largeur, canaux). Par exemple, (512, 512, 3).
        num_classes (int): Le nombre de classes pour la segmentation.

    Returns:
        tensorflow.keras.Model: Un modèle Keras prêt à être compilé et entraîné, avec l'architecture UNet Mini.

    Example:
        model = build_unet_mini(input_shape=(512, 512, 3), num_classes=8)
        model.summary()

    Notes:
        - Le modèle utilise une activation softmax dans la couche finale pour prédire les classes.
        - La taille de l'image d'entrée doit correspondre à l'input_shape (par exemple, 512x512).
    """
    inputs = Input(input_shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 48, pool=True)
    x4, p4 = conv_block(p3, 64, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 128, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)

    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)

    model = Model(inputs, output, name="Unet_mini")
    model.backbone = None

    return model

def build_vgg16_unet(input_shape, num_classes):
    """
    Construire un U-Net avec VGG16 comme backbone pour une segmentation multiclasses.

    Args:
        input_shape (tuple): Dimensions des images d'entrée, par ex. (512, 512, 3).
        num_classes (int): Nombre total de classes.

    Returns:
        Model: Modèle compilé U-Net basé sur VGG16.
    """
    from segmentation_models import Unet
    backbone = "vgg16"
    model = Unet(backbone, 
        classes=num_classes, 
        activation='softmax', 
        input_shape=input_shape, 
        encoder_weights='imagenet', 
        encoder_freeze=False
    )
    model.name = "VGG16_UNet"
    model.backbone = backbone

    return model

def build_mobilenet_fpn(input_shape, num_classes):
    """
    Construire un FPN (Feature Pyramid Network) avec MobileNet comme backbone pour une segmentation multiclasses.

    Args:
        input_shape (tuple): Dimensions des images d'entrée, par ex. (512, 512, 3).
        num_classes (int): Nombre total de classes.

    Returns:
        Model: Modèle compilé FPN basé sur MobileNet.
    """
    from segmentation_models import FPN
    backbone = "mobilenet"
    model = FPN(backbone, 
        classes=num_classes, 
        activation='softmax', 
        input_shape=input_shape, 
        encoder_weights='imagenet', 
        encoder_freeze=False
    )
    model.name = "Mobilenet_FPN"
    model.backbone = backbone

    return model