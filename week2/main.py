import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.knn import knn_predict, plot_misclassified_points , plot_classified_points

df = pd.read_csv("data/dataset_hipertensiune2.csv")
X = df [["Varsta" , "IMC" , "Colesterol"]].values
# X = df[["IMC", "Colesterol" ]].values
y = df["Hipertensiune"].values

"""
    imc -> indice masa corporala
    colesterol -> ml / g
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
for k in range(1, 52, 2):
#k numarul de vecini

    knn_y_pred = knn_predict(X_train, y_train, X_test , k )
    knn_accuracy = accuracy_score(y_test, knn_y_pred)
    print(f"Precizia KNN: {knn_accuracy:.5f} ({k} vecini)")
    print(f" ")
"""
knn_y_pred = knn_predict(X_train, y_train, X_test , k = 3 )

plot_misclassified_points(X_test, y_test, knn_y_pred)

plot_classified_points(X_train, X_test, y_train, y_test)
plt.show()
