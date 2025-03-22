from src.utils import entropy, split_dataset, most_common_label

class Node:
    def __init__(self, column=None, value=None, true_branch=None, false_branch=None, label=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.label = label
    
    def is_leaf(self):
        return self.label is not None

def best_split(data, target, features):
    best_gain = 0
    best_column = None
    best_value = None
    current_entropy = entropy(data, target)

    for column in features:
        unique_values = data[column].unique()
        for value in unique_values:
            true_data, false_data = split_dataset(data, column, value)
            
            # Skip if split results in empty set
            if len(true_data) == 0 or len(false_data) == 0:
                continue
            
            # Calculate information gain
            p = len(true_data) / len(data)
            gain = current_entropy - (p * entropy(true_data, target) + (1-p) * entropy(false_data, target))
            
            if gain > best_gain:
                best_gain = gain
                best_column = column
                best_value = value
                
    return best_column, best_value

def build_tree(data, target, features):
    # Base case 1: all samples have same label
    if len(data[target].unique()) == 1:
        return Node(label=data[target].iloc[0])
    
    # Base case 2: no features left
    if len(features) == 0:
        return Node(label=most_common_label(data, target))
    
    # Find best split
    column, value = best_split(data, target, features)
    
    # If no good split found, return most common label
    if column is None:
        return Node(label=most_common_label(data, target))
    
    # Split data
    true_data, false_data = split_dataset(data, column, value)
    
    # Create node and build subtrees
    node = Node(column=column, value=value)
    node.true_branch = build_tree(true_data, target, features)
    node.false_branch = build_tree(false_data, target, features)
    
    return node

def predict(node, row):
    if node.is_leaf():
        return node.label
    else:
        if row[node.column] == node.value:
            return predict(node.true_branch, row)
        else:
            return predict(node.false_branch, row)