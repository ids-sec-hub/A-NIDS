import torch
from Config import *
from Train.Datasets import DataSets
from Train.ModelOperators import ModelOperators
from Train.ConfusionMatrix import ConfusionMatrix


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

    TrainX, Test1X, Test2X, TrainY, Test1Y, Test2Y = _data.LoadAll('../data/Domain-CICIDS-2017-Norm.csv', \
                                                '../data/Domain-CICIDS-2018-Norm.csv', test_size=0.3)
    TrainDataloader = _data.LoadDataloader(TrainX, TrainY, batch_size)
    #_operator.Train(TrainDataloader, 'pkl/AllIDS.pkl')

    TestDataloader = _data.LoadDataloader(Test1X, Test1Y, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2017', 'pkl/AllIDS.pkl')
    _conf.UpdateAndPlot(Test1Y.long(), preds, 'cm/2017_and_2018/2017_On_AllIDS.jpg')

    TestDataloader = _data.LoadDataloader(Test2X, Test2Y, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2018', 'pkl/AllIDS.pkl')
    _conf.UpdateAndPlot(Test2Y.long(), preds, 'cm/2017_and_2018/2018_On_AllIDS.jpg')
