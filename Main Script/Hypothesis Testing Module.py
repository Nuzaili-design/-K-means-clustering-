from scipy.stats import f_oneway
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=250, centers=4, cluster_std=0.60, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_scaled)

cluster_labels = kmeans.labels_

# Example hypothesis: Test if there are significant differences in feature 1 among clusters
feature1 = X[:, 0]
f_statistic, p_value = f_oneway(feature1[cluster_labels == 0], feature1[cluster_labels == 1],
                                feature1[cluster_labels == 2], feature1[cluster_labels == 3])

print("F-statistic:", f_statistic)
print("P-value:", p_value) 
    


   
