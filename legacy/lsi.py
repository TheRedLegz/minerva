from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

def lsi(matrix):
    (x, y) = matrix.shape
    
    if y <= x:
        initial_components = y - 1
    else:
        initial_components = x - 1
        
    lsi = TruncatedSVD(n_components=initial_components)
    lsi.fit(matrix)

    cumsum = lsi.explained_variance_ratio_.cumsum()
    optimal_components = initial_components

    optimal_sum = []

    for i, sum in enumerate(cumsum):
        optimal_sum.append(sum)

        if sum > .80:
            optimal_components = i + 1
            break

    lsi_final = TruncatedSVD(n_components=optimal_components)
    res = lsi_final.fit_transform(matrix)

    return (res, optimal_sum)
