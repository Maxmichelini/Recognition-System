import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from dataset import AppleBrowningDataset
from model import SVMModel, RandomForestModel
from model import optimize_svm_model, optimize_rf_model
import numpy as np


def train_svm(model: SVMModel, dataset: Dataset):
    X_train, y_train = [], []  # Inizializza due liste vuote per contenere le features e le labels utilizzate per l'addestramento
    
    # Itera sul dataset per estrarre features e labels
    for i in range(len(dataset)):
        features, labels = dataset[i]
        
        # Appiattimento delle dimensioni, se necessario
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        X_train.append(features)
        y_train.append(labels)
    
    # Converti le liste in array numpy
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    
    
    # Debug: Controlla le dimensioni e la distribuzione delle classi nei dati di addestramento
    print(f"Training data shape: {len(X_train)}, Labels distribution: {sum(y_train)} positive, {len(y_train) - sum(y_train)} negative")
    
    optimized_model, best_params = optimize_svm_model(X_train, y_train)
    model.model = optimized_model
    
    print("Best SVM Params:", best_params)





def train_rf(model: RandomForestModel, dataset: Dataset):
    X_train, y_train = [], []  # Inizializza due liste vuote per contenere le features e le labels utilizzate per l'addestramento
    
    # Itera sul dataset per estrarre features e labels
    for i in range(len(dataset)):
        features, labels = dataset[i]
        
        # Appiattimento delle dimensioni, se necessario
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        X_train.append(features)
        y_train.append(labels)

    # Debug: Controlla le dimensioni e la distribuzione delle classi nei dati di addestramento
    print(f"Training data shape: {len(X_train)}, Labels distribution: {sum(y_train)} positive, {len(y_train) - sum(y_train)} negative")
        
    #model.model.fit(X_train, y_train)
    optimized_model, best_params = optimize_rf_model(X_train, y_train)
    model.model = optimized_model

    print("Best RF Params:", best_params)

