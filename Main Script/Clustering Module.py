from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=250, centers=4, cluster_std=0.60, random_state=0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_scaled)

cluster_labels = kmeans.labels_

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Results')
plt.show()



