import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

print("--- INICIANDO SCRIPT DE RELATÓRIOS (V5 - Correção do NameError) ---")

# --- 1. Configurações de Caminho ---
BASE_PROJECT_DIR = "/Users/tainapocharski/Documents/pfc1_sinalizacao_horizontal"
MODELS_SOURCE_DIR = "/Users/tainapocharski/Documents/models/grid_search_completo"
IMAGES_DIR = os.path.join(BASE_PROJECT_DIR, "dataset", "unico", "val")
REPORTS_BASE_DIR = os.path.join(BASE_PROJECT_DIR, "relatorios_finais")
os.makedirs(REPORTS_BASE_DIR, exist_ok=True)

# --- 2. Configurações do Modelo ---
INPUT_SHAPE = (224, 224)
CLASS_NAMES = ['LFO2', 'LFO3', 'LBO'] 

# Apontando para os modelos "Campeões" da sua Tabela 4.1
MODELS_TO_TEST = {
    "MobileNetV2_CAMPEAO": "MobileNetV2_E20_B32.h5",
    "VGG16_CAMPEAO": "VGG16_E50_B32.h5",
    "ResNet50_FALHA": "ResNet50_E100_B32.h5",
    "EfficientNetB0_FALHA": "EfficientNetB0_E20_B32.h5" 
}

# =======================================================================
# 🎯 PARÂMETROS DE VISÃO COMPUTACIONAL
# =======================================================================
ROI_Y_START_RATIO = 0.50 
MIN_CONTOUR_AREA = 30 
# =======================================================================

# --- 3. Funções de Visão Computacional ---

def criar_mascara_faixa(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180]); upper_white = np.array([180, 50, 255]) 
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
    lower_yellow = np.array([15, 100, 100]); upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    full_mask = cv2.bitwise_or(mask_white, mask_yellow)
    kernel = np.ones((5, 5), np.uint8) 
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel, iterations=2) 
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=1) 
    return full_mask

def calcular_centro_faixa_unico(img_resized_bgr):
    largura = img_resized_bgr.shape[1]
    altura = img_resized_bgr.shape[0]
    y_start = int(altura * ROI_Y_START_RATIO) 
    roi = img_resized_bgr[y_start:altura, 0:largura]
    mask_roi = criar_mascara_faixa(roi)
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centro_x_list = [] 
    
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA: continue 
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx_roi = int(M["m10"] / M["m00"])
        centro_x_list.append(cx_roi)
        
    if not centro_x_list: return None
    
    cx_final = int(np.mean(centro_x_list))
    cy_final = int(altura * 0.9) 
    
    return (cx_final, cy_final)

# --- 4. Funções de Desenho e Busca ---

def draw_predictions(frame, classe_real, classe_prevista, confianca, centro_cv):
    display_img = cv2.resize(frame, (480, 480))
    cor_texto = (0, 0, 255) # Vermelho (Errado)
    if classe_prevista == classe_real:
        cor_texto = (0, 255, 0) # Verde (Certo)
        
    cv2.putText(display_img, f"Real: {classe_real}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, f"Prev: {classe_prevista} ({confianca * 100:.1f}%)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_texto, 2)
    
    if centro_cv is not None:
        scale_x = 480 / INPUT_SHAPE[1]
        scale_y = 480 / INPUT_SHAPE[0]
        centro_scaled = (int(centro_cv[0] * scale_x), int(centro_cv[1] * scale_y))
        
        cv2.circle(display_img, centro_scaled, 5, (0, 0, 255), -1) # Ponto Vermelho
        cv2.putText(display_img, f"CV Ponto: {centro_scaled}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Ciano
        
    return display_img

def find_images(images_dir):
    image_paths = []
    true_labels = []
    for dirpath, _, filenames in os.walk(images_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                classe_real = os.path.basename(dirpath) 
                if classe_real in CLASS_NAMES:
                    image_paths.append(os.path.join(dirpath, f))
                    true_labels.append(classe_real)
    return image_paths, true_labels

# --- 5. Função Principal de Inferência ---

def run_inference_for_model(model_name, h5_filename):
    
    print(f"\n=======================================================")
    print(f"PROCESSANDO: {model_name} (Arquivo: {h5_filename})")
    print(f"=======================================================")

    model_path = os.path.join(MODELS_SOURCE_DIR, h5_filename)
    MODEL_OUTPUT_DIR = os.path.join(REPORTS_BASE_DIR, model_name)
    VIZ_DIR = os.path.join(MODEL_OUTPUT_DIR, "debug_inferencia")
    OUTPUT_CSV = os.path.join(MODEL_OUTPUT_DIR, f"resultados_{model_name}.csv")
    CONFUSION_MATRIX_FILE = os.path.join(MODEL_OUTPUT_DIR, f"matriz_confusao_{model_name}.png")

    os.makedirs(VIZ_DIR, exist_ok=True)
    
    print(f"Carregando modelo: {h5_filename}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"ERRO: Falha ao carregar o modelo '{model_path}'. Pulando. Detalhe: {e}")
        return

    image_paths, true_labels = find_images(IMAGES_DIR)
    if not image_paths:
        print(f"ERRO: Nenhuma imagem de validação encontrada em {IMAGES_DIR}. Abortando.")
        sys.exit(1)
    
    print(f"Encontradas {len(image_paths)} imagens. Iniciando inferência...")

    results = []
    pred_labels = []

    for img_path, classe_real in zip(image_paths, true_labels):
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized_rgb = cv2.resize(img_rgb, INPUT_SHAPE)
            img_array = img_to_array(img_resized_rgb)
            img_normalized = np.expand_dims(img_array, axis=0) / 255.0

            img_resized_bgr = cv2.resize(img_bgr, INPUT_SHAPE)
            centro_cv = calcular_centro_faixa_unico(img_resized_bgr)

            predicoes = model.predict(img_normalized, verbose=0)
            
            pred_class_probs = predicoes[0]
            pred_class_index = np.argmax(pred_class_probs)
            pred_class_name = CLASS_NAMES[pred_class_index]
            confianca = np.max(pred_class_probs)

            results.append({
                "Arquivo": os.path.basename(img_path),
                "Classe Real": classe_real,
                "Classe Prevista (DL)": pred_class_name,
                "Confianca (DL)": confianca,
                "Ponto Detectado (CV)": "Sim" if centro_cv else "Nao",
                "Ponto Coords (CV)": f"{centro_cv}" if centro_cv else "N/A"
            })
            pred_labels.append(pred_class_name)

            viz_img = draw_predictions(img_bgr, classe_real, pred_class_name, confianca, centro_cv)
            viz_filename = f"VIZ_{classe_real}_{os.path.basename(img_path)}"
            cv2.imwrite(os.path.join(VIZ_DIR, viz_filename), viz_img)

        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
            
    # --- 6. Gerar Relatório Final (para este modelo) ---
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📊 Relatório CSV salvo em: {OUTPUT_CSV}")
    print(f"🖼️ Imagens de visualização salvas em: {VIZ_DIR}")

    report = classification_report(true_labels, pred_labels, labels=CLASS_NAMES, digits=4, zero_division=0)
    acc = accuracy_score(true_labels, pred_labels)
    
    print("\n--- RELATÓRIO DE CLASSIFICAÇÃO (Deep Learning) ---")
    print(f"📈 Acurácia Geral ({model_name}): {acc * 100:.2f}%")
    print(report) 

    cm = confusion_matrix(true_labels, pred_labels, labels=CLASS_NAMES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    # 🛑🛑🛑 CORREÇÃO AQUI (linha 207) 🛑🛑🛑
    plt.title(f"Matriz de Confusão - {model_name} ({acc * 100:.2f}%)") # Corrigido de 'model_a_name'
    # 🛑🛑🛑 FIM DA CORREÇÃO 🛑🛑🛑
    
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close() 
    print(f"✅ Matriz de Confusão salva em: {CONFUSION_MATRIX_FILE}")
    print("-----------------------------------------------------")


# --- 6. Execução Principal ---
if __name__ == "__main__":
    
    if os.path.exists(REPORTS_BASE_DIR):
        print(f"Limpando pasta de relatórios antiga: {REPORTS_BASE_DIR}")
        shutil.rmtree(REPORTS_BASE_DIR)
    os.makedirs(REPORTS_BASE_DIR, exist_ok=True)
    
    src_csv_resumo = os.path.join(MODELS_SOURCE_DIR, "grid_search_results.csv")
    dst_csv_resumo = os.path.join(REPORTS_BASE_DIR, "resumo_geral_treinamento.csv")
    if os.path.exists(src_csv_resumo):
        shutil.copy(src_csv_resumo, dst_csv_resumo)

    for model_name, h5_filename in MODELS_TO_TEST.items():
        run_inference_for_model(model_name, h5_filename)
        
    print("\n\n✅✅✅ Processo de Geração de Relatórios Concluído! ✅✅✅")
    print(f"Todos os resultados estão organizados em: {REPORTS_BASE_DIR}")