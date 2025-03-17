import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class DataSets:
    def __init__(self):
        pass


    def LoadAll(self, path1, path2, test_size=0.0, norm=False):
        df1 = pd.read_csv(path1, header=None)
        df2 = pd.read_csv(path2, header=None)
        y1 = df1.iloc[:, -1]
        X1 = df1.iloc[:, :-1]

        y2 = df2.iloc[:, -1]
        X2 = df2.iloc[:, :-1]
        X11, X12, y11, y12 = train_test_split(X1, y1, test_size=test_size, random_state=42)
        X21, X22, y21, y22 = train_test_split(X2, y2, test_size=test_size, random_state=42)

        df1 = pd.concat([X11, y11], axis=1)
        df2 = pd.concat([X21, y21], axis=1)

        df = pd.concat([df1, df2])
        df = df.sample(frac=1.0)

        df = df.drop(0)

        data = np.array(df)
        label_pos = data.shape[1] - 1
        X = data[1:,:label_pos]  
        y = data[1:,label_pos]

        if norm is True:
            X12 = preprocessing.MinMaxScaler().fit(X).transform(X12)
            X22 = preprocessing.MinMaxScaler().fit(X).transform(X22)
            X = preprocessing.MinMaxScaler().fit_transform(X)

        # pd.df & np to torch.tensor
        return torch.from_numpy(X.astype(float)), torch.from_numpy(X12.values.astype(float)), torch.from_numpy(X22.values.astype(float)), \
            torch.from_numpy(y.astype(float)), torch.from_numpy(y12.values.astype(float)), torch.from_numpy(y22.values.astype(float))


    def LoadDataset(self, path, test_size=0.0):
        df = pd.read_csv(path).sample(frac=1.0)

        data = np.array(df)
        label_pos = data.shape[1] - 1
        X = data[:,:label_pos]  
        y = data[:,label_pos]

        if test_size != 0.0:
            X1, X2, y1, y2 = train_test_split(X, y, test_size=test_size, random_state=42)
            # pd.df & np to torch.tensor
            return torch.from_numpy(X1), torch.from_numpy(X2), \
                torch.from_numpy(y1), torch.from_numpy(y2)
        return torch.from_numpy(X), torch.from_numpy(y)
    

    def LoadDataloader(self, X, y, batch_size):
        dataset = Data.TensorDataset(X, y)
        data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, 
                shuffle=False, num_workers=4)
        return data_loader
    