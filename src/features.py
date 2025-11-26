import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def scale_and_pca(X_train, X_test=None, n_components=200, do_pca=True):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if X_test is not None else None

    if do_pca:
        pca = PCA(n_components=n_components, random_state=42)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s) if X_test_s is not None else None
        return X_train_p, X_test_p, scaler, pca
    else:
        return X_train_s, X_test_s, scaler, None
