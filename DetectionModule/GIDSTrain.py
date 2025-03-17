import torch
from Config import *
from Train.Datasets import DataSets
from Train.ModelOperators import ModelOperators
from Train.ConfusionMatrix import ConfusionMatrix


if __name__ == '__main__':
    torch.manual_seed(1234)

    batch_size = 256
    learning_rate = 1e-4
    epoch = 200
    
    no_cuda = True  # 强制只使用 CPU
    cuda_available = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")  # 只使用 CPU

    _data = DataSets()
    _conf = ConfusionMatrix(label_names)
    _operator = ModelOperators(learning_rate, epoch, cuda_available, device)

    TrainX, TestX, TrainY, TestY = _data.LoadDataset('../data/Domain-CICIDS-2017-G-Norm.csv', test_size=0.3)
    TrainDataloader = _data.LoadDataloader(TrainX, TrainY, batch_size)
    _operator.Train(TrainDataloader, 'pkl/GIDS.pkl')

    TestDataloader = _data.LoadDataloader(TestX, TestY, batch_size * 5)
    preds = _operator.Test(TestDataloader, 'syn', 'pkl/GIDS.pkl')
    _conf.UpdateAndPlot(TestY.long(), preds, 'cm/gids/Generate_On_GIDS.jpg')

    _2017X, _2017Y = _data.LoadDataset('../data/Domain-CICIDS-2017-Norm.csv')
    TestDataloader = _data.LoadDataloader(_2017X, _2017Y, batch_size * 5)
    preds = _operator.Test(TestDataloader, '2017', 'pkl/GIDS.pkl')
    _conf.UpdateAndPlot(_2017Y.long(), preds, 'cm/gids/2017_On_GIDS.jpg')
