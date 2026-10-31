import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10],
    [2], [4], [6], [8], [10]
], dtype=float)
y = np.array([
    15, 22, 30, 38, 45,
    55, 62, 70, 82, 90,
    25, 40, 58, 75, 95
], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train)
print(y_train)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

"""
trees = []
lamda_ = 1
pred = np.full(len(y_train),np.mean(y_train))

for iteration in range(5):
    thresholds = []
    gains = []
    gradients = pred - y_train
    root_similarity = np.sum(gradients)**2 / (len(gradients) + lamda_)
    x_unique = np.unique(X_train)
    for i in range(len(x_unique)-1):
        thes = (x_unique[i] + x_unique[i+1])/2
        thresholds.append(thes)
    for threshold in thresholds:
        right_nodes = X_train.flatten() > threshold
        left_nodes = X_train.flatten() <= threshold
        left_gradient = gradients[left_nodes]
        right_gradient = gradients[right_nodes]
        #Similary score 
        G_left = np.sum(left_gradient)
        G_right = np.sum(right_gradient)
        H_left = np.sum(left_nodes)
        H_right = np.sum(right_nodes)
        score_left = G_left**2/(H_left + lamda_)
        score_right = G_right**2/(H_right + lamda_)
        gain = 0.5 * (score_left + score_right - root_similarity)
        gains.append(gain)
        #print(f"left: {score_left} right: {score_right} gain: {gain}")
    #print(gains)
    best_index = np.argmax(gains)
    best_thresholds = thresholds[best_index]
    left_nodes = X_train.flatten() <= best_thresholds
    right_nodes = X_train.flatten() > best_thresholds
    left_gradient = gradients[left_nodes]
    right_gradient = gradients[right_nodes]
    G_left = np.sum(left_gradient)
    G_right = np.sum(right_gradient)
    H_left = np.sum(left_nodes)
    H_right = np.sum(right_nodes)
    left_weight = -G_left / (H_left + lamda_)
    right_weight = -G_right / (H_right + lamda_)
    trees.append({
    "threshold": best_thresholds,
    "left_weight": left_weight,
    "right_weight": right_weight
    })
    pred[left_nodes] += 0.1 * left_weight
    pred[right_nodes] += 0.1 * right_weight
    #gradients = pred - y_train
    #print("Left weight:", left_weight)
    #print("Right weight:", right_weight)
    mse = np.mean((y_train - pred) ** 2)
    #print(mse)

trees
"""
lamda_ = 1
pred = np.full(len(y_train),np.mean(y_train))
gradients = pred - y_train
hessians = np.ones(len(y_train))
#root_similarity = (np.sum(gradients)**2 /(np.sum(hessians) + lamda_))
tree = {}
def all_threshold(X_train):
    thresholds = []
    x_unique = np.unique(X_train)
    for i in range(len(x_unique)-1):
        thes = (x_unique[i] + x_unique[i+1])/2
        thresholds.append(thes)
    print(f"Thresholds: {thresholds}")
    return thresholds 

def similarity(gradients,hessians,left_nodes,right_nodes):
    #Similarity score
    root_similarity = (np.sum(gradients)**2 /(np.sum(hessians) + lamda_)) 
    print(f"gradients: {gradients}")
    print(f"left_nodes: {left_nodes}")
    print(f"right_nodes: {right_nodes}")
    left_gradient = gradients[left_nodes]
    right_gradient = gradients[right_nodes]
    G_left = np.sum(left_gradient)
    G_right = np.sum(right_gradient)
    H_left = np.sum(hessians[left_nodes])
    H_right = np.sum(hessians[right_nodes])
    score_left = G_left**2/(H_left + lamda_)
    score_right = G_right**2/(H_right + lamda_)
    score = (score_left + score_right) - root_similarity
    print(f"score: {score}")
    return score

def splitting(X_train,threshold):
    left_nodes = X_train.flatten() > threshold
    right_nodes = X_train.flatten() <= threshold
    return left_nodes,right_nodes

def gain(X_train, gradients,hessians):
    gains = []
    thresholds = all_threshold(X_train)
    for threshold in thresholds:
        left_nodes,right_nodes = splitting(X_train,threshold)
        score = similarity(gradients,left_nodes,right_nodes)
        gain = 0.5 * score
        gains.append(gain)
    best_index = np.argmax(gains)
    best_threshold = thresholds[best_index]
    print(f"Best gain: {best_threshold}")
    return best_threshold

def build_tree(X_train,gradients,hessians,depth=0):
    if depth == 2 or len(X_train) < 2:
        return {"leaf": -np.sum(gradients) / (len(gradients) + lamda_)}
    best_threshold = gain(X_train, gradients,hessians)
    left_nodes,right_nodes = splitting(X_train,best_threshold)
    left_gradient = gradients[left_nodes]
    right_gradient = gradients[right_nodes]
    X_left = X_train[left_nodes]
    X_right = X_train[right_nodes]
    return {
        "threshold": best_threshold,
        "left": build_tree(X_left, left_gradient, depth + 1),
        "right": build_tree(X_right, right_gradient, depth + 1)
    }

import json
tree = build_tree(X_train, gradients)
print(json.dumps(tree, indent=4))