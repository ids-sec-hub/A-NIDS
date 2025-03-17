import joblib
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans


class Cluster():
    def __init__(self, 
                 n_cluster: int,
                 model_path: str = None):
        self._cluster = n_cluster
        self._model_path = model_path
        if model_path is None:
            self._model = KMeans(n_clusters=n_cluster, random_state=42, n_init='auto')
        else:
            self._model = joblib.load(model_path)

    
    def Train(self, X):
        if self._model_path is None:
            self._model.fit(X)
            joblib.dump(self._model, f'pkl/kmeans_{self._cluster}_model.pkl')
        else:
            pass

    
    def Test(self, X):
        return self._model.predict(X)
    

    def MeanD(self, X, y_c):
        # 计算每个样本到所属簇的平均距离
        distances = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            cluster_center = self._model.cluster_centers_[y_c[i]]
            distances[i] = np.linalg.norm(X.iloc[i].values - cluster_center)
        
        return distances
    
    
    def ComputeT(self, distances, alpha):
        return np.mean(distances) + alpha * np.std(distances)