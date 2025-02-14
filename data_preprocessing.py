import pandas as pd
from sklearn.model_selection import train_test_split 

def load_and_preprocess_data():
    """
    Charge et prétraite les données du dataset Heart Disease UCI.
    Retourne les ensembles d'entraînement et de test.
    """
    # Charger le dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    data = pd.read_csv(url, names=column_names, na_values="?")

    # Remplacer les valeurs manquantes
    data["ca"].fillna(data["ca"].median(), inplace=True)
    data["thal"].fillna(data["thal"].mode()[0], inplace=True)

    # Convertir la cible en binaire (1 = maladie cardiaque, 0 = pas de maladie)
    data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)

    # Séparer les caractéristiques (X) et la cible (y)
    X = data.drop("target", axis=1)
    y = data["target"]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, data