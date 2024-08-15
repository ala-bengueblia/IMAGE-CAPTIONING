from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def generate_desc(model, tokenizer, photo, max_length):
    in_text = '<start>'
    photo = np.expand_dims(photo, axis=0)  # Ajouter un axe pour le batch
    photo = np.expand_dims(photo, axis=0)  # Assurez-vous que la forme est (1, 4096)
    
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        sequence = np.expand_dims(sequence, axis=0)  # Assurez-vous que la forme est (1, max_length)
        
        # Prédiction
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        
        # Ajouter le mot à la séquence
        in_text += ' ' + word
        
        # Arrêter si le mot de fin est généré
        if word == '<end>':
            break
    
    return in_text


# Utilisation pour générer une légende pour une nouvelle image
from extract_features import extract_features

photo = extract_features('images/image1.jpeg')
