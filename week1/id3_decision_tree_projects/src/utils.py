import pandas as pd 
import numpy as np
from sklearn.utils import shuffle 
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return shuffle(data,random_state=42)

#unique_values - data[column].unique()
def unique_values(data,column): 
    unique_values = data[column].unique()
    return unique_values

#split_dataset - data[data[column] == value], data[data[column] != value]
def split_dataset(data,column,value):
    return data[data[column] == value], data[data[column] != value]

#ost_common_label - data[target].mode()[0]
def most_common_label(data , target):
    return data[target].mode()[0]

#entropy - calculeaza proporția, urmată de -sum(p*log2(p))
def entropy(data , target):
    labels = data[target].value_counts(normalize=True)
    return sum([-p * np.log2(p) for p in labels])

