import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load dataset
df = pd.read_csv("dataset/iris.csv")

X = df.drop("target", axis=1)
y = df["target"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 1: clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)

centroids = kmeans.cluster_centers_

# STEP 2: create error-based features
def create_features(X):
    new_features = []
    
    for x in X.values:
        distances = [np.linalg.norm(x - c) for c in centroids]
        new_features.append(distances)
    
    return np.hstack((X.values, new_features))

X_train_new = create_features(X_train)
X_test_new = create_features(X_test)

# STEP 3: train model
model = LogisticRegression()
model.fit(X_train_new, y_train)

# STEP 4: prediction
y_pred = model.predict(X_test_new)

# accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# STEP 5: save outputs
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")

# plot confusion matrix for accuracy.png
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Accuracy: {acc:.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("outputs/accuracy.png")

