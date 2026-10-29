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