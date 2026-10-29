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


lamda_ = 1
pred = np.full(len(y_train),np.mean(y_train))
gradients = pred - y_train
root_similarity = np.sum(gradients)**2/len(gradients) + lamda_
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
    score_left = G_left**2/H_left * lamda_
    score_right = G_right**2/H_right * lamda_
    gain = 1/2*(((score_left + score_right) - root_similarity)-lamda_)
   
    print(f"left: {score_left} right: {score_right} gain: {gain}")
