import torch
from torch import nn


class RobustIDSEncoder(nn.Module):
    def __init__(self, 
            feature_num: int,
            hiddens1: int,
            hiddens2: int,
            output_num: int,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(feature_num, hiddens1)
        self.dense2 = nn.Linear(hiddens1, hiddens2)
        self.dense3 = nn.Linear(hiddens2, output_num)

    
    def forward(self, x):
        return self.dense3(self.relu(self.dense2(self.relu(self.dense1(x)))))
        

class RobustIDS(nn.Module):
    def __init__(self, 
            feature_num: int,
            hiddens1: int,
            hiddens2: int,
            output1_num: int,
            output2_num: int,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = RobustIDSEncoder(feature_num, hiddens1, hiddens2, output1_num)
        self.dense = nn.Linear(output1_num, output2_num)


    def forward(self, x):
        _eout = self.encoder(x)
        return self.dense(_eout), _eout
