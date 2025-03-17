import torch
import numpy as np
from Config import *
from Train.Datasets import DataSets
from Train.ModelOperators import ModelOperators
from Train.ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def cls_report(y, preds, title):
    report = classification_report(y, preds, digits=4)
    print(title)
    print(report)


if __name__ == '__main__':
    torch.manual_seed(1234)

    batch_size = 256
    learning_rate = 1e-4
    epoch = 1000
    
    no_cuda = True  # 强制只使用 CPU
    cuda_available = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")  # 只使用 CPU

    _data = DataSets()
    _conf = ConfusionMatrix(label_names)
    _operator = ModelOperators(learning_rate, epoch, cuda_available, device)

    TrainX, TestX, TrainY, TestY = _data.LoadDataset('../data/Domain-CICIDS-2017-Norm.csv', test_size=0.3)
    TrainDataloader = _data.LoadDataloader(TrainX, TrainY, batch_size)
    #_operator.Train(TrainDataloader, 'pkl/NaiveIDS.pkl')

    TestDataloader = _data.LoadDataloader(TestX, TestY, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2017', 'pkl/NaiveIDS.pkl')
    _conf.UpdateAndPlot(TestY.long(), preds, 'cm/naive/2017_On_NaiveIDS.jpg')
    #cls_report(TestY, preds, '## 2017 On NaiveIDS1 Cls Report:')
    trues = torch.where(TestY == 0, 0, 1)
    preds = torch.where(preds == 0, 0, 1)
    print(confusion_matrix(trues, preds))

    _2018X, _2018Y = _data.LoadDataset('../data/Domain-CICIDS-2018-Norm.csv')
    TestDataloader = _data.LoadDataloader(_2018X, _2018Y, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2018', 'pkl/NaiveIDS.pkl')
    _conf.UpdateAndPlot(_2018Y.long(), preds, 'cm/naive/2018_On_NaiveIDS.jpg')
    #cls_report(_2018Y, preds, '## 2018 On NaiveIDS1 Cls Report:')
    np.save('../data/2018_labels.npy', _2018Y.cpu().numpy())
    np.save('../data/2018_preds.npy', preds.cpu().numpy())

    trues = torch.where(_2018Y == 0, 0 ,1)
    preds = torch.where(preds == 0, 0, 1)
    print(confusion_matrix(trues, preds))
