import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
import os

# Charger le modèle VGG16 pré-entraîné
def load_vgg16_model():
    model = VGG16()
    return Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Extraction des caractéristiques de l'image
def extract_features(filename, model):
    try:
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = model.predict(image, verbose=0)
        return np.reshape(features, (4096,))
    except Exception as e:
        raise RuntimeError(f"Error extracting features: {e}")

# Création des séquences pour l'entraînement
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            photo = np.reshape(photo, (1, 4096))
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1).squeeze(), np.array(X2), np.array(y)

# Définir le modèle de légende
def define_model(vocab_size, max_length, dropout_rate, dense_units):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(dropout_rate)(inputs1)
    fe2 = Dense(dense_units, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, dense_units, mask_zero=True)(inputs2)
    se2 = Dropout(dropout_rate)(se1)
    se3 = LSTM(dense_units)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(dense_units, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Génération de la description de l'image
def generate_desc(model, tokenizer, photo, max_length, temperature=1.0):
    in_text = '<start>'
    photo = photo.reshape((1, 4096))
    generated_words = set()
    final_words = []

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat_probs = model.predict([photo, sequence], verbose=0)[0]
        yhat_probs = np.asarray(yhat_probs).astype('float64')
        yhat_probs = np.log(yhat_probs + 1e-10) / temperature
        yhat_probs = np.exp(yhat_probs) / np.sum(np.exp(yhat_probs))
        yhat = np.random.choice(len(yhat_probs), p=yhat_probs)
        
        word = tokenizer.index_word.get(yhat, '')
        if word == '' or word == '<end>':
            break
        
        if word not in generated_words:
            final_words.append(word)
            generated_words.add(word)
        
        in_text += ' ' + word
    
    # Nettoyer et formater la description
    description = ' '.join(final_words).strip()
    if description:
        description = description[0].upper() + description[1:] + '.'
    
    return description

# Fonction d'exécution des scripts
def execute_scripts():
    try:
        log_text.delete(1.0, tk.END)

        image_file = image_path.get()
        if not image_file or not os.path.isfile(image_file):
            messagebox.showwarning("Warning", "No valid image file selected.")
            return

        photo = extract_features(image_file, vgg_model)
        log_text.insert(tk.END, "Extraction des caractéristiques de l'image terminée.\n")
        
        X1, X2, y = create_sequences(tokenizer, max_length, descriptions.get(image_file, []), photo)
        log_text.insert(tk.END, f"Création des séquences pour l'entraînement terminée. Nombre de séquences : {len(X1)}\n")
        
        epochs = int(epochs_entry.get())
        dropout_rate = float(dropout_rate_entry.get())
        dense_units = int(dense_units_entry.get())
        model_caption = define_model(vocab_size, max_length, dropout_rate, dense_units)
        log_text.insert(tk.END, "Début de l'entraînement du modèle...\n")
        model_caption.fit([X1, X2], y, epochs=epochs, verbose=2)
        log_text.insert(tk.END, "Entraînement du modèle terminé avec succès.\n")
        
        description = generate_desc(model_caption, tokenizer, photo, max_length)
        log_text.insert(tk.END, "Génération de la légende terminée.\n")
        
        # Afficher la description dans la zone de texte
        desc_text.delete(1.0, tk.END)
        desc_text.insert(tk.END, f"Description générée : {description}\n")
        
        if save_results.get():
            save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(f"Generated description: {description}\n")
        log_text.insert(tk.END, "Processus terminé.\n")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        log_text.insert(tk.END, f"Error: {str(e)}\n")

# Fonction de chargement de l'image
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")])
    if file_path:
        image_path.set(file_path)
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

# Simulateur de descriptions
descriptions = {
    "C:/Users/alabe/OneDrive/Bureau/CODSOFT/task3/images/image1.jpeg": ["chat  , canapé", "chat_blanc , canapé_noir"],
    "C:/Users/alabe/OneDrive/Bureau/CODSOFT/task3/images/image2.jpeg": [" chien joue  ballon", " chien, jardin"],
    "C:/Users/alabe/OneDrive/Bureau/CODSOFT/task3/images/image3.jpeg": [" oiseausur ,l'arbre", " oiseau-multicolore"],
    "C:/Users/alabe/OneDrive/Bureau/CODSOFT/task3/images/image4.jpeg": [" vache  , jardin", " VacheNoireBlanche"],
    "C:/Users/alabe/OneDrive/Bureau/CODSOFT/task3/images/image5.jpeg": [" homme bateau", " homme ,mer"]
}

# Préparation des données
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sum(descriptions.values(), []))
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(desc.split()) for desc in sum(descriptions.values(), []))

# Charger le modèle VGG16
vgg_model = load_vgg16_model()

# Interface utilisateur Tkinter
root = tk.Tk()
root.title("Image Captioning Application")
root.geometry("900x600")
root.configure(bg='#303F9F')

# Variables
image_path = tk.StringVar()

# Cadre principal
main_frame = tk.Frame(root, bg='#ffffff')
main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Cadre pour les contrôles de l'image
image_frame = tk.Frame(main_frame, bg='#ffffff')
image_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

# Cadre pour les paramètres du modèle
params_frame = tk.Frame(main_frame, bg='#ffffff')
params_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

# Cadre pour les résultats
results_frame = tk.Frame(main_frame, bg='#ffffff')
results_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Section des contrôles de l'image
load_button = tk.Button(image_frame, text="Load Image", command=load_image, bg='#FF5722', fg='#ffffff', font=('Arial', 12, 'bold'), relief=tk.RAISED, borderwidth=2)
load_button.pack(pady=10, fill=tk.X)

img_label = tk.Label(image_frame, bg='#ffffff', relief=tk.SUNKEN)
img_label.pack(pady=10)

# Section des paramètres du modèle
epochs_label = tk.Label(params_frame, text="Number of Epochs:", bg='#ffffff', font=('Arial', 12, 'bold'))
epochs_label.pack(pady=5)
epochs_entry = tk.Entry(params_frame, width=10, font=('Arial', 12), relief=tk.RAISED, borderwidth=2)
epochs_entry.insert(0, "10")
epochs_entry.pack(pady=5)

dropout_rate_label = tk.Label(params_frame, text="Dropout Rate:", bg='#ffffff', font=('Arial', 12, 'bold'))
dropout_rate_label.pack(pady=5)
dropout_rate_entry = tk.Entry(params_frame, width=10, font=('Arial', 12), relief=tk.RAISED, borderwidth=2)
dropout_rate_entry.insert(0, "0.5")
dropout_rate_entry.pack(pady=5)

dense_units_label = tk.Label(params_frame, text="Dense Units:", bg='#ffffff', font=('Arial', 12, 'bold'))
dense_units_label.pack(pady=5)
dense_units_entry = tk.Entry(params_frame, width=10, font=('Arial', 12), relief=tk.RAISED, borderwidth=2)
dense_units_entry.insert(0, "256")
dense_units_entry.pack(pady=5)

save_results = tk.BooleanVar()
save_results_check = tk.Checkbutton(params_frame, text="Save Results", variable=save_results, bg='#ffffff', font=('Arial', 12))
save_results_check.pack(pady=5)

execute_button = tk.Button(params_frame, text="Execute Scripts", command=execute_scripts, bg='#4CAF50', fg='#ffffff', font=('Arial', 12, 'bold'), relief=tk.RAISED, borderwidth=2)
execute_button.pack(pady=10, fill=tk.X)

# Section des résultats
log_text = scrolledtext.ScrolledText(results_frame, width=80, height=10, bg='#f1f8e9', fg='#000000', font=('Arial', 12))
log_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

desc_text = scrolledtext.ScrolledText(results_frame, width=80, height=10, bg='#f1f8e9', fg='#000000', font=('Arial', 12))
desc_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

root.mainloop()
