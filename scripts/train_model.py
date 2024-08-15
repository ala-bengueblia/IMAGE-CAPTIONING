# Script pour entraîner le modèle de légendes
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer

# Simulateur de descriptions (à remplacer par un vrai ensemble de données)
descriptions = {
    "image1.jpeg": ["un chat assis sur un canapé", "un chat blanc sur un canapé noir"]
}

# Préparation des données pour l'entraînement
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sum(descriptions.values(), []))
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(desc.split()) for desc in sum(descriptions.values(), []))

def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Définition du modèle
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Extraction des caractéristiques et entraînement du modèle
from extract_features import extract_features

# Pas besoin de np.squeeze ici
photo = extract_features('images/image1.jpeg')

# photo devrait déjà être de la forme (4096,)
X1, X2, y = create_sequences(tokenizer, max_length, descriptions['image1.jpeg'], photo)

print(f"Shape of photo: {photo.shape}")



model_caption = define_model(vocab_size, max_length)
model_caption.fit([X1, X2], y, epochs=10, verbose=2)
