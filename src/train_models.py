from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV

def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, use_grid=False):
    if use_grid:
        param_grid = {"n_neighbors":[3,5,7]}
        gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
    else:
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, use_grid=False):
    if use_grid:
        param_grid = {"C":[1,10], "kernel":["rbf","linear"]}
        gs = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
    else:
        model = SVC(kernel="rbf", probability=True)
        model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train, use_grid=False):
    if use_grid:
        param_grid = {"n_estimators":[100,200], "max_depth":[None,20]}
        gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
    else:
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)
