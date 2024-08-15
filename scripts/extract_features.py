# Script pour extraire les caractéristiques des images
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Charger le modèle VGG16 pré-entraîné
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(filename):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    features = np.reshape(features, (features.shape[0], features.shape[1]))  # Aplatir la sortie
    return features


# Exemple d'utilisation
features = extract_features('images/image1.jpeg')
print(features)
