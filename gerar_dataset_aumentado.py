# Script: scripts/gerar_dataset_aumentado.py
# Descrição: Carrega as 115 imagens originais e aplica 10 transformações
# (Data Augmentation Offline) para gerar o dataset de treino de 1150 imagens.
# Referência: Seção 3.1

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import shutil

# --- 1. Configurações de Caminho ---
BASE_PROJECT_DIR = "/Users/tainapocharski/Documents/pfc1_sinalizacao_horizontal"
ORIGINAL_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "dataset", "unico")
AUGMENTED_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "dataset_aumentado")
CLASSES = ['LBO', 'LFO2', 'LFO3']

# Número de variações sintéticas a serem criadas por imagem original
N_AUGMENTATIONS = 10

# --- 2. Configuração do Gerador de Aumento ---
# Conforme descrito na Seção 3.5.2 [cite: 1745-1749]
datagen = ImageDataGenerator(
    rotation_range=15,         # Rotação [cite: 1746]
    width_shift_range=0.10,    # Deslocamento horizontal [cite: 1748]
    height_shift_range=0.10,   # Deslocamento vertical [cite: 1748]
    zoom_range=0.10,           # Zoom [cite: 1747]
    brightness_range=[0.8, 1.2], # Alterações de brilho [cite: 1749]
    fill_mode='nearest'
)

# --- 3. Função de Geração ---
def augment_images():
    print("--- Iniciando Geração de Dataset Aumentado (Offline) ---")
    
    # Limpa o diretório antigo, se existir
    if os.path.exists(AUGMENTED_DATA_DIR):
        print(f"Limpando diretório antigo: {AUGMENTED_DATA_DIR}")
        shutil.rmtree(AUGMENTED_DATA_DIR)
        
    os.makedirs(AUGMENTED_DATA_DIR)

    total_generated = 0
    total_original = 0

    for classe in CLASSES:
        original_class_path = os.path.join(ORIGINAL_DATA_DIR, classe)
        augmented_class_path = os.path.join(AUGMENTED_DATA_DIR, classe)
        os.makedirs(augmented_class_path, exist_ok=True)
        
        print(f"\nProcessando classe: {classe}")
        
        # Encontra todas as imagens originais na pasta da classe
        image_files = [f for f in os.listdir(original_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_original += len(image_files)

        for img_name in image_files:
            img_path = os.path.join(original_class_path, img_name)
            
            try:
                # Carrega a imagem
                img = load_img(img_path)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                
                i = 0
                # Gera N_AUGMENTATIONS novas imagens
                for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_class_path, save_prefix=f"aug_{img_name.split('.')[0]}", save_format='jpg'):
                    i += 1
                    total_generated += 1
                    if i >= N_AUGMENTATIONS:
                        break  # Sai do loop após criar N variações
                        
            except Exception as e:
                print(f"Erro ao processar {img_path}: {e}")

    print("\n--- Geração Concluída ---")
    print(f"Imagens Originais Processadas: {total_original}")
    print(f"Imagens Sintéticas Geradas: {total_generated} (Esperado: {total_original * N_AUGMENTATIONS})")
    print(f"Dataset aumentado salvo em: {AUGMENTED_DATA_DIR}")

# --- 4. Execução ---
if __name__ == "__main__":
    augment_images()
