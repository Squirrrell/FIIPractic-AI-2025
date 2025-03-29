import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.knn import knn_predict, plot_misclassified_points

df = pd.read_csv("data/dataset_hipertensiune.csv")
X = df[["IMC", "Colesterol"]].values
y = df["Hipertensiune"].values

"""
    imc -> indice masa corporala
    colesterol -> ml / g
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn_y_pred = knn_predict(X_train, y_train, X_test, k=5)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print(f"Precizia KNN: {knn_accuracy:.2f}")
plot_misclassified_points(X_test, y_test, knn_y_pred)
plt.show()
