import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import seaborn as sns

def k_means():
    df = pd.read_csv('Mall_Customers.csv')
    
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vectors = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        vectors.append(kmeans.inertia_)
    
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    
    df['Cluster'] = kmeans.labels_
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis')
    plt.show()

def tsne():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X_subset, y_subset = X[:10000], y[:10000]  
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    X_tsne = tsne.fit_transform(X_subset)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_subset, palette=sns.color_palette("hsv", 10), legend='full')
    plt.title('t-SNEvisualization')
    plt.show()
    
if __name__ == "__main__":
    k_means()
    tsne()
