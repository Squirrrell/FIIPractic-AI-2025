import os
from src.utils import load_dataset
from src.id3 import build_tree , Node , best_split , entropy , split_dataset , most_common_label , predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

dataset_path = os.path.join("data" , "diabetes_dataset.csv")
data = load_dataset(dataset_path)


target = "diabet"
train_size = int(0.7 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

tree = build_tree(train_data, target, data.columns)

y_true = test_data[target].values

y_pred = [predict(tree, row) for _, row in test_data.iterrows()]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='RdPu')
plt.title("Matricea de Confuzie")
plt.savefig("confusion_matrix.png")



