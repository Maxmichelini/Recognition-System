import torch
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, make_scorer





class SVMModel:
    def __init__(self):
        self.model = SVC()

    def predict(self, x):
        return self.model.predict(x)
    
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def predict(self, x):
        return self.model.predict(x)






def optimize_svm_model(X_train, y_train):

    

    new_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ]
    )

    param_grid = {
        #'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        #'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #'model__gamma': [0.001, 0.01, 0.1, 1, 10],
        #'model__degree': [2, 3, 4, 5, 6],  # Rilevante per il kernel 'poly'
        #'model__coef0': [0.0, 0.1, 0.5, 1.0],  # Rilevante per i kernel 'poly' e 'sigmoid'
        #'model__class_weight': ['balanced']  # Per il bilanciamento delle classi
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'model__kernel': ['linear', 'poly', 'rbf'],
        'model__gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto'],
        'model__degree': [2, 3, 4],  # Limito i valori per evitare overfitting
        'model__coef0': [0.0, 0.1, 0.5],  # Limito i valori per kernel 'poly' e 'sigmoid'
        'model__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]  # Diversi pesi per il bilanciamento
    }
    grid_search = GridSearchCV(new_model, param_grid,verbose=5, cv=5, refit=True, scoring="accuracy") # cv=5 indica la divisione in 5 fold per la cross-validation
    grid_search.fit(X_train, y_train) #vengono ricercati i parametri migliori (tramite la cross-validation) ed inoltre viene effettuato l'addestramento del modello utilizzando il classificatore iniziale utilizzando i dati di addestramento X_train e y_train
    #print(f"Scaler: {best_scaler}")

    return grid_search.best_estimator_, grid_search.best_params_#ritorno il modello ottimizzato e i parametri migliori

    # random_search = RandomizedSearchCV(new_model, param_grid, n_iter=100, verbose=5, cv=StratifiedKFold(n_splits=5), refit= True, scoring='f1_macro', random_state=42)
    # random_search.fit(X_train, y_train)
    # return random_search.best_estimator_, random_search.best_params_
    





def optimize_rf_model(X_train, y_train):

    new_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier()),   
        ]
    )

    param_grid = {
        # 'model__n_estimators': [100, 200, 300, 500, 700, 1000],
        # 'model__max_depth': [None, 10, 20, 30, 50, 70, 100],
        # 'model__min_samples_split': [2, 5, 10, 20],
        # 'model__bootstrap': [True, False],
        # 'model__min_samples_leaf': [1, 2, 4, 10],
        # 'model__max_features': [None, 'sqrt', 'log2'],
        # 'model__class_weight': [None, 'balanced'],
        
        'model__n_estimators': [100, 300, 500, 700],
        'model__max_depth': [None, 20, 40, 50],
        'model__min_samples_split': [2, 5, 10],
        'model__bootstrap': [True, False],
        'model__min_samples_leaf': [1, 4],
        'model__max_features': [None, 'sqrt', 'log2'],
        'model__class_weight': [None, 'balanced']
    }


    grid_search = GridSearchCV(new_model, param_grid,verbose=5, cv=5, refit=True, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_