import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

def plot_kmeans_variations(X, Ks, dataset_index): #crtanje, predajemo podatak, popis razlicitih vrijednosti i broj dataseta
    fig, axes = plt.subplots(1, len(Ks), figsize=(18, 4))
    fig.suptitle(f'Dataset {dataset_index} — Usporedba različitih K', fontsize=14) #stvaramo subplot 
    
    for i, K in enumerate(Ks): #prolazimo kroz sve K vrijednosti
        kmeans = KMeans(n_clusters=K, random_state=0)
        labels = kmeans.fit_predict(X)
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10) #crta graf u i tom subplotu
        axes[i].set_title(f'K = {K}')
        axes[i].set_xlabel('$x_1$')
        axes[i].set_ylabel('$x_2$')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


# Definiraj koje K vrijednosti želiš usporediti
K_values = [2, 3, 4, 5]

# Prikaži usporedbe za svaki dataset
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    plot_kmeans_variations(X, K_values, dataset_index=flagc)

optimal_K = {1: 3, 2: 3, 3: 4, 4: 2, 5: 2}
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('K-means grupiranje — optimalni K za svaki dataset', fontsize=16)

for i, flagc in enumerate(range(1, 6)):
    X = generate_data(500, flagc)
    kmeans = KMeans(n_clusters=optimal_K[flagc], random_state=0)
    labels = kmeans.fit_predict(X)
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
    axes[i].set_title(f'Dataset {flagc} (K={optimal_K[flagc]})')
    axes[i].set_xlabel('$x_1$')
    axes[i].set_ylabel('$x_2$')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()