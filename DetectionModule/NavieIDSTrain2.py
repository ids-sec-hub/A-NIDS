import torch
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

    TrainX, TestX, TrainY, TestY = _data.LoadDataset('../data/Domain-CICIDS-2018-Norm.csv', test_size=0.3)
    TrainDataloader = _data.LoadDataloader(TrainX, TrainY, batch_size)
    #_operator.Train(TrainDataloader, 'pkl/NaiveIDS2.pkl')

    TestDataloader = _data.LoadDataloader(TestX, TestY, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2018', 'pkl/NaiveIDS2.pkl')
    _conf.UpdateAndPlot(TestY.long(), preds, 'cm/naive/2018_On_NaiveIDS2.jpg')
    #cls_report(TestY, preds, '## 2018 On NaiveIDS2 Cls Report:')
    trues = torch.where(TestY == 0, 0 ,1)
    preds = torch.where(preds == 0, 0, 1)
    print(confusion_matrix(trues, preds))

    _2017X, _2017Y = _data.LoadDataset('../data/Domain-CICIDS-2017-Norm.csv')
    TestDataloader = _data.LoadDataloader(_2017X, _2017Y, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2017', 'pkl/NaiveIDS2.pkl')
    _conf.UpdateAndPlot(_2017Y.long(), preds, 'cm/naive/2017_On_NaiveIDS2.jpg')
    #cls_report(_2017Y, preds, '## 2017 On NaiveIDS2 Cls Report:')
    trues = torch.where(_2017Y == 0, 0 ,1)
    preds = torch.where(preds == 0, 0, 1)
    print(confusion_matrix(trues, preds))
