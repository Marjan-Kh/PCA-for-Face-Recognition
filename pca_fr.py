# Applying PCA to face recognition problem

#=========================================
# Author: Marjan Khamesian
# Date: August 2020
#=========================================

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# ----------------------------------------
df=pd.read_csv('https://raw.githubusercontent.com/daradecic/Python-Eigenfaces/master/face_data.csv') 
print(df.head())

# Shape of the dataset
df.shape

# The number of unique elements of the target column
df['target'].nunique()

X = df.drop('target', axis=1)
y = df['target']

# Transforming 1D vector to a 2D matrix
def plot_faces(pixels):
    fig, axes = plt.subplots(6, 6, figsize=(3, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
    plt.show()

plot_faces(X)

# Splitting the data into training/testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA().fit(X_train)
plt.figure(figsize=(18, 7))
plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3)

np.where(pca.explained_variance_ratio_.cumsum() > 0.95)

# Applying PCA with additional n_components
pca = PCA(n_components=105).fit(X_train)

# Transforming the training features
X_train_pca = pca.transform(X_train)

# Model Training 
from sklearn.svm import SVC
classifier = SVC().fit(X_train_pca, y_train)

# Evaluation
X_test_pca = pca.transform(X_test)
predictions = classifier.predict(X_test_pca)

# Performance
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, predictions))

