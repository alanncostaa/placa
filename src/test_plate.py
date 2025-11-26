import cv2
import joblib
import numpy as np
from src.segment_chars import segment_characters_from_plate
from src.preprocess import preprocess_image
from src.utils import read_image


def predict_character(model, char_img):
    if char_img is None:
        raise ValueError("Erro: char_img chegou como None.")

    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

    char_img = cv2.resize(char_img, (28, 28))

    img_flat = char_img.flatten().reshape(1, -1)

    img_normalized = img_flat / 255.0 

    pred = model.predict(img_normalized)[0]
    return pred


def predict_plate(image_path):

    print("Carregando imagem:", image_path)
    img = read_image(image_path, to_gray=False)
    if img is None:
        print("[ERRO] Falha ao carregar imagem!")
        return

    chars = segment_characters_from_plate(img)
    print(f"Caracteres encontrados: {len(chars)}")

    if len(chars) == 0:
        print("Nenhum caractere detectado!")
        return

    print("Carregando modelos...")    

    models = {
        "KNN": joblib.load("outputs/models/knn_chars.pkl"),
        "SVM": joblib.load("outputs/models/svm_chars.pkl"),
        "RF":  joblib.load("outputs/models/rf_chars.pkl"),
    }

    plate_predictions = {name: "" for name in models}

    for img in chars:
        for name, model in models.items():
            pred_label = predict_character(model, img) 
            plate_predictions[name] += pred_label

    print("\n===== RESULTADO FINAL =====")
    for name, pred in plate_predictions.items():
        print(f"{name}: {pred}")

    return plate_predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Caminho da imagem da placa")
    args = parser.parse_args()

    predict_plate(args.image)