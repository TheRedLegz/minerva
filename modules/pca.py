from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(matrix):
    (x, y) = matrix.shape

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    
    pca = PCA(n_components=y)
    pca.fit(scaled)

    cumsum = pca.explained_variance_ratio_.cumsum()
    optimal_components = y
    
    for i, sum in enumerate(cumsum):
        if sum > .80:
            optimal_components = i + 1
            break

    pca_final = PCA(n_components=optimal_components)
    res = pca_final.fit_transform(scaled)

    return res
