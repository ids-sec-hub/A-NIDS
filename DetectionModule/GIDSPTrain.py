import torch
from Config import *
from Train.Datasets import DataSets
from Train.ModelOperators import ModelOperators
from Train.ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

CUDA_LAUNCH_BLOCKING=1


def cls_report(y, preds, title):
    report = classification_report(y, preds, digits=4)
    print(title)
    print(report)


if __name__ == '__main__':
    torch.manual_seed(1234)

    batch_size = 256
    learning_rate = 1e-4
    epoch = 1500
    
    no_cuda = True  # 强制只使用 CPU
    cuda_available = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")  # 只使用 CPU

    _data = DataSets()
    _conf = ConfusionMatrix(label_names)
    _operator = ModelOperators(learning_rate, epoch, cuda_available, device)

    TrainX, Test1X, Test2X, TrainY, Test1Y, Test2Y = _data.LoadAll('../data/Domain-CICIDS-2017-G-Norm.csv', \
                                                '../data/Domain-CICIDS-2018-Norm.csv', test_size=0.3)
    TrainDataloader = _data.LoadDataloader(TrainX, TrainY, batch_size)
    #_operator.Train(TrainDataloader, 'pkl/GIDSP.pkl')

    TestDataloader = _data.LoadDataloader(Test1X, Test1Y, batch_size * 5)
    preds = _operator.Test(TestDataloader, 'syn', 'pkl/GIDSP.pkl')
    _conf.UpdateAndPlot(Test1Y.long(), preds, 'cm/gidsp/syn_On_GIDSP.jpg')

    TestDataloader = _data.LoadDataloader(Test2X, Test2Y, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2018', 'pkl/GIDSP.pkl')
    _conf.UpdateAndPlot(Test2Y.long(), preds, 'cm/gidsp/2018_On_GIDSP.jpg')
    #cls_report(Test2Y, preds, '## 2018 On GIDSP Cls Report:')
    trues = torch.where(Test2Y == 0, 0 ,1)
    preds = torch.where(preds == 0, 0, 1)
    print(confusion_matrix(trues, preds))

    _TestX, _TestY = _data.LoadDataset('../data/Domain-CICIDS-2017-Norm.csv')
    _TestDataloader = _data.LoadDataloader(_TestX, _TestY, batch_size * 5)
    preds = _operator.Test(_TestDataloader, '2017', 'pkl/GIDSP.pkl')
    _conf.UpdateAndPlot(_TestY.long(), preds, 'cm/gidsp/2017_On_GIDSP.jpg')
    #cls_report(_TestY, preds, '## 2017 On GIDSP Cls Report:')
    trues = torch.where(_TestY == 0, 0 ,1)
    preds = torch.where(preds == 0, 0, 1)
    print(confusion_matrix(trues, preds))
