import os
import cv2
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from train_models import train_knn, train_svm, train_rf, save_model

IMG_SIZE = 28

def load_character_dataset(base_path="dataset_chars"):
    X = []
    y = []
    classes = sorted(os.listdir(base_path))

    for label in classes:
        class_dir = os.path.join(base_path, label)

        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"[ERRO] Não foi possível carregar {img_path}")
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten() / 255.0  # normalização

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)


def main_train():
    
    print("Carregando dataset de caracteres...")
    X, y = load_character_dataset("dataset_chars")

    print(f"Total de caracteres: {len(X)}")
    print(f"Total de classes: {len(set(y))}")

    counts = Counter(y)
    valid_classes = [cls for cls, qt in counts.items() if qt >= 2]

    X_filtered = []
    y_filtered = []

    for xi, yi in zip(X, y):
        if yi in valid_classes:
            X_filtered.append(xi)
            y_filtered.append(yi)

    X = np.array(X_filtered)
    y = np.array(y_filtered)

    print(f"Classes removidas: {len(counts) - len(valid_classes)}")
    print(f"Dataset final: {len(X)} imagens e {len(valid_classes)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\nTreinando modelo KNN...")
    knn = train_knn(X_train, y_train, use_grid=False)
    knn_pred = knn.predict(X_test)
    
    print("\nKNN Classification Report:")
    print(classification_report(y_test, knn_pred, zero_division=0))

    save_model(knn, "outputs/models/knn_chars.pkl")

    print("\nTreinando modelo SVM...")
    svm = train_svm(X_train, y_train, use_grid=False)
    svm_pred = svm.predict(X_test)

    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_pred, zero_division=0))
    
    save_model(svm, "outputs/models/svm_chars.pkl")

    print("\nTreinando modelo Random Forest...")
    rf = train_rf(X_train, y_train, use_grid=False)
    rf_pred = rf.predict(X_test)
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred, zero_division=0))

    save_model(rf, "outputs/models/rf_chars.pkl")


if __name__ == "__main__":
    main_train()
