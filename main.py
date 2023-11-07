import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load the MNIST dataset in CSV format
print("Loading MNIST dataset...")
subset_size = 1000 # Adjust this to the size of the subset you want
print(f"Loading a subset of {subset_size} samples from the MNIST dataset...")
mnist = pd.read_csv('/Users/harshrajmishra/Documents/MEGAMINDS/mnist_train.csv', nrows=subset_size)

X = mnist.drop(columns=['label']).values / 255.0
y = mnist['label'].values.astype(int)

# Perform PCA to reduce dimensionality
print("Performing PCA for dimensionality reduction...")
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X)

# Implement t-SNE 
def perform_tsne(X, n_components, k_neighbors):
    print(f"Performing t-SNE with {n_components} components and {k_neighbors} neighbors...")
    tsne = TSNE(n_components=n_components, perplexity=30, method="exact")
    X_tsne = tsne.fit_transform(X)
    return X_tsne

# K-NN sampling
def knn_sampling(X, k_neighbors):
    print(f"Performing k-NN sampling with {k_neighbors} neighbors...")
    knn = NearestNeighbors(n_neighbors=k_neighbors)
    knn.fit(X)
    indices = knn.kneighbors(X, return_distance=False)
    return indices

# Main function
def main():
    n_components = 2
    k_neighbors_tsne = 10
    k_neighbors_knn = 20

    # Perform t-SNE with k-NN sampling
    X_tsne = perform_tsne(X_pca, n_components, k_neighbors_tsne)

    # Plot the t-SNE-transformed data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
    plt.title('t-SNE Transformation of MNIST Data')
    plt.colorbar()
    plt.show()
    
    # Perform k-NN sampling on the t-SNE-transformed data
    indices = knn_sampling(X_tsne, k_neighbors_knn)
    X_knn_sampled = X_tsne

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_knn_sampled, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier on k-NN sampled data
    print("Training Random Forest on k-NN sampled data...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()

