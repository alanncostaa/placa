import os
import cv2
import pandas as pd

def segment_characters_from_plate(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if h < 25 or w < 10:
            continue

        ratio = h / float(w)
        if ratio < 1.2 or ratio > 6.0:  
            continue

        area = w * h
        if area < 200:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    for (x, y, w, h) in boxes:
        char_img = gray[y:y+h, x:x+w]

        mean_val = char_img.mean()

        if mean_val < 40 or mean_val > 220:
            continue 

        char_images.append(char_img)

    return char_images


def save_characters(dataset_path, csv_path, output_base):
    

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        filename = row['filename']
        plate_text = row['plate']   

        img_path = os.path.join(dataset_path, filename)
        if not os.path.exists(img_path):
            print(f"[ERRO] Imagem não encontrada: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERRO] Falha ao abrir: {img_path}")
            continue

        print(f"Processando {filename} -> {plate_text}")

        chars_img = segment_characters_from_plate(image)

        if len(chars_img) != len(plate_text):
            print(f"[AVISO] Número de caracteres não bate ({len(chars_img)} vs {len(plate_text)})")
            continue

        for i, char in enumerate(plate_text):
            folder = os.path.join(output_base, char.upper())
            os.makedirs(folder, exist_ok=True)

            char_img = chars_img[i]

            save_path = os.path.join(folder, f"{filename}_char{i}.png")
            cv2.imwrite(save_path, char_img)


if __name__ == "__main__":
    save_characters(
        dataset_path="dataset/train",
        csv_path="dataset/labels.csv",
        output_base="dataset_chars"
    )
