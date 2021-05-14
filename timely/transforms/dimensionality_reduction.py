from sklearn.decomposition import PCA


def series_reduction(ds, n_components=0.90):
    """
    application of pca for
    multivariate timely series reduction
    """
    pca = PCA(n_components=n_components)
    pca.fit(ds)
    return pca
