import numpy as np
from collections import deque
from treelib import Node, Tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
 
#Here you should put the directory path of your python code
directory = 'C:/Users/ivoko/Documents/Studie/Python/'

##Tree visualisation
def represent_content(node):
    som = (np.sum(node.y))
    lengte = (len(node.y))
    return str((round((lengte - som)), round(som)))

def visualize_tree(root):
    parent_dict = dict()
    tree = Tree()   
    queue = deque()
    
    # Function to add nodes to queue and parent dictionary
    def printer(node):
        try:
            if node.left is not None:
                queue.append(node.left)
                parent_dict[represent_content(node.left)] = represent_content(node)
 
            if node.right is not None:
                queue.append(node.right)
                parent_dict[represent_content(node.right)] = represent_content(node)
        except Exception as e:
            print(f"Error in node traversal: {e}")
    
    # Start with the root node
    root_tag = represent_content(root)
    queue.append(root)
 
    # Create the root node in the tree
    tree.create_node(root_tag, root_tag)
 
    # Loop through the queue and add nodes to the tree
    while len(queue) != 0:
        current_node = queue.pop()
        printer(current_node)
        
    # Add child nodes to the tree
    for key, value in parent_dict.items():
        if not tree.contains(key):
            if not tree.contains(value):
                # This means parent node is missing
                print(f"Error: Parent node {value} is missing before adding {key}.")
            else:
                tree.create_node(key, key, parent=value)
 
    tree.show(line_type='ascii')


##Part 1 Functions

#TreeNode is the class used for trees. Each TreeNode can be a leaf node, and if not has a (possible) left and right node.
class TreeNode:
    def __init__(self, x, y, best_val=None, best_column=None, is_leaf=False):
        self.x = x #The feature values, consisting of a subset of values of the input columns x
        self.y = y #The vector of remaining class labels
        self.left = None
        self.right = None
        self.best_column = best_column
        self.best_val = best_val
        self.is_leaf = is_leaf
        self.majority = np.argmax(np.bincount(y))

#Impurity function calculating the impurity of an array (from lecture notes)
def impurity(array):
    if len(array) == 0:
        return 0
    else:
        a = 1 - (np.sum(array[:] == 0) / len(array))**2 - (np.sum(array[:] == 1) / len(array))**2
        return a

def bestsplit(x,y, minleaf, nfeat):
    best_reduction = 0
    best_column = -1
    best_value = -1
    nfeat = min(nfeat, x.shape[1])
    columns = np.random.choice(x.shape[1], nfeat, replace=False)
    for column in columns:
        xcolumn = x[:, column]
        sorted_xcolumn = np.unique(xcolumn) #Because of duplicate values
        splitpoints = (sorted_xcolumn[:-1] + sorted_xcolumn[1:]) / 2

        #Calculate the best split
        for i in splitpoints:
            l = y[xcolumn <= i]
            r = y[xcolumn > i]
    
            #Skip the false split with less than minleaf elements in l or r
            if len(l) < minleaf or len(r) < minleaf:
                continue
    
            #Calculate the impurity reduction of the split and remember the best
            impurity_reduction = impurity(y) - (len(l)/len(y))*impurity(l) - (len(r)/len(y))*impurity(r)
            if impurity_reduction > best_reduction:
                best_reduction = impurity_reduction
                best_value = i
                best_column = column
    return (best_column, best_value)

def recursive_growth(node, nmin, minleaf, nfeat):
    x = node.x
    y = node.y
    if len(y) <= nmin:
        node.is_leaf = True
        return
 
    (column, value) = bestsplit(x, y, minleaf, nfeat)
 
    if column == -1:
        node.is_leaf = True
        return
 
    leftcondition = x[:, column] <= value
    rightcondition = x[:, column] > value
    node.left = TreeNode(x[leftcondition], y[leftcondition], is_leaf=False)
    node.right = TreeNode(x[rightcondition], y[rightcondition], is_leaf=False)
    node.best_column = column
    node.best_val = value
    recursive_growth(node.left, nmin, minleaf, nfeat)
    recursive_growth(node.right, nmin, minleaf, nfeat)
 
def tree_grow(x, y, nmin, minleaf, nfeat):
    tree = TreeNode(x, y, is_leaf=False)
    recursive_growth(tree, nmin, minleaf, nfeat)
    return tree
 
def tree_pred(x, tr):
    preds = []
    for row in x:
        node = tr
        while not node.is_leaf:
            if row[node.best_column] <= node.best_val:
                node = node.left
            else:
                node = node.right
        preds.append(node.majority)
    return np.array(preds)
 
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    trs = []
    for _ in range(m):
        indcs =np.random.choice(len(y), len(y), replace=True)
        x_sample = x[indcs]
        y_sample = y[indcs]
 
        tree = tree_grow(x_sample, y_sample, nmin, minleaf, nfeat)
        trs.append(tree)
    return trs
 
def tree_pred_b(x, trs):
    preds = np.zeros((len(x), len(trs)))
    for item, tree in enumerate(trs):
        preds[:, item] = tree_pred(x, tree)
    final_result = []
    for row in preds:
        majority = np.bincount(row.astype(int)).argmax()
        final_result.append(majority)
    return np.array(final_result)
 
def randomf_grow(x, y, nmin, minleaf, m):
    nfeat = int(np.sqrt(x.shape[1]))
    return tree_grow_b(x, y, nmin, minleaf, nfeat, m)

##PART 2 functions

#NOTE: Ask if it's ok to use sklearn for the evaluation
def accuracy(y_true, y_pred):
    corrects = sum(y_true == y_pred)
    return corrects / len(y_true)
def precision(y_true, y_pred):
    true_pos = sum((y_true==1) & (y_pred == 1))
    pred_pos = sum(y_pred ==1)
    return 0 if pred_pos == 0  else true_pos / pred_pos
def recall(y_true, y_pred):
    true_pos = sum((y_true==1) & (y_pred == 1))
    actual_pos = sum(y_true ==1)
    return 0 if actual_pos == 0  else true_pos / actual_pos
def conf_matrix(y_true, y_pred):
    true_pos = sum((y_true == 1) & (y_pred == 1))
    true_neg = sum((y_true == 0) & (y_pred == 0))
    false_pos = sum((y_true == 0) & (y_pred == 1))
    false_neg = sum((y_true == 1) & (y_pred == 0))
    return np.array([[true_neg, false_pos], [false_neg, true_pos]])


##Formatting the data

#Open the pima indians data file
file = open(directory+'Pima_Indians.txt')
data = file.read()

# Split the string data into lines and then convert each line into a list of floats
data_lines = data.strip().split('\n')
data_array = np.array([list(map(float, line.split(','))) for line in data_lines])

# Now X can be properly sliced
X = data_array[:, :-1]  # Features
y = data_array[:, -1].astype(int)  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##Using functions on the formatted data: single classification tree NOTE: this is now done for the test in the hints with pima indians data, the data matches exactly!
tree = tree_grow(X, y, nmin=20, minleaf=5, nfeat=8)
y_pred_tree = tree_pred(X, tree)

acc_tree = accuracy(y, y_pred_tree)
prec_tree = precision(y, y_pred_tree)
rec_tree = recall(y, y_pred_tree)
conf_matrix_tree = conf_matrix(y, y_pred_tree)


##Using functions on the formatted data: bagging
forest_bagging = tree_grow_b(X_train, y_train, nmin=20, minleaf=5, nfeat=2, m=5)
y_pred_bagging = tree_pred_b(X_test, forest_bagging)
 
acc_bagging = accuracy(y_test, y_pred_bagging)
prec_bagging = precision(y_test, y_pred_bagging)
rec_bagging = recall(y_test, y_pred_bagging)
conf_matrix_bagging = conf_matrix(y_test, y_pred_bagging)



##Using function on the formatted data: random forests
forest_rf = randomf_grow(X_train, y_train, nmin=20, minleaf=5, m=5)
y_pred_rf = tree_pred_b(X_test, forest_rf)

acc_rf = accuracy(y_test, y_pred_rf)
prec_rf = precision(y_test, y_pred_rf)
rec_rf = recall(y_test, y_pred_rf)
conf_matrix_rf = conf_matrix(y_test, y_pred_rf)


##Result output
print(f"Single Tree | Accuracy: {acc_tree}, Precision: {prec_tree}, Recall: {rec_tree}")
print(f"Confusion Matrix:\n{conf_matrix_tree}")
print(f"Bagging | Accuracy: {acc_bagging}, Precision: {prec_bagging}, Recall: {rec_bagging}")
print(f"Confusion Matrix:\n{conf_matrix_bagging}")
print(f"Random Forest | Accuracy: {acc_rf}, Precision: {prec_rf}, Recall: {rec_rf}")
print(f"Confusion Matrix:\n{conf_matrix_rf}")


models = ['Single Tree', 'Bagging', 'Random Forest']
accuracy_scores = [acc_tree, acc_bagging, acc_rf]
precision_scores = [prec_tree, prec_bagging, prec_rf]
recall_scores = [rec_tree, rec_bagging, rec_rf]

bar_width = 0.2
index = np.arange(len(models))

"""
plt.figure(figsize=(10, 6))
plt.bar(index, accuracy_scores, bar_width, label='Accuracy', color='b')
plt.bar(index + bar_width, precision_scores, bar_width, label='Precision', color='g')
plt.bar(index + 2 * bar_width, recall_scores, bar_width, label='Recall', color='r')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Performance Comparison')
plt.xticks(index + bar_width, models)
plt.legend()
plt.show()
"""
