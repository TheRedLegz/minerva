from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(matrix):
    (x, y) = matrix.shape

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    
    if y <= x:
        initial_components = y
    else:
        initial_components = x

    pca = PCA(n_components=initial_components)
    pca.fit(scaled)

    cumsum = pca.explained_variance_ratio_.cumsum()
    optimal_components = initial_components

    optimal_sum = []

    for i, sum in enumerate(cumsum):
        optimal_sum.append(sum)

        if sum > .80:
            optimal_components = i + 1
            break

    pca_final = PCA(n_components=optimal_components)
    res = pca_final.fit_transform(scaled)

    return (res, optimal_sum)
