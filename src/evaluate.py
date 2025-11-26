import argparse
import cv2
import joblib
import os
from preprocess import preprocess_char
from segment_chars import segment_characters_from_plate
from features import extract_features


def predict_plate(image_path, model_path="outputs/best_model.pkl"):
    
    model = joblib.load(model_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    chars = segment_characters_from_plate(image)

    if len(chars) == 0:
        print("Nenhum caractere encontrado!")
        return ""

    print(f"➡ Caracteres detectados: {len(chars)}")

    plate_string = ""

    os.makedirs("outputs/test_segments", exist_ok=True)

    for i, char_img in enumerate(chars):
        gray = preprocess_char(char_img)

        feat = extract_features(gray)
        feat = feat.reshape(1, -1)

        pred = model.predict(feat)[0]

        plate_string += pred

        cv2.imwrite(f"outputs/test_segments/char_{i}_{pred}.png", char_img)

    return plate_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Caminho da imagem da placa")

    args = parser.parse_args()

    result = predict_plate(args.image)

    print("\n========================")
    print("➡ Resultado final:", result)
    print("========================\n")


if __name__ == "__main__":
    main()
