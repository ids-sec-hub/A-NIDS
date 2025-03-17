import time
import torch
import pandas as pd
from torch import nn
from torch.autograd import Variable
from Models.RobustIDS import RobustIDS
from Config import *


class ModelOperators():
    def __init__(self, lr, epoch, cuda, device) -> None:
        self.__lr = lr
        self.__epoch = epoch
        self.__cuda = cuda
        self.__device = device


    def Train(self, train_loader, model_path):
        start = time.time()

        model = RobustIDS(cic_feature_num, cic_hiddens1, cic_hiddens2, \
            cic_output1, cic_output2).to(self.__device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.__lr)
        loss_func = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(self.__epoch):
            train_loss = 0
            for (X, labels) in train_loader:
                X = Variable(X.float())
                labels = Variable(labels.long())
                if self.__cuda: 
                    X = X.cuda()
                    labels = labels.cuda()

                preds, _ = model(X)
                
                _labels = torch.zeros(len(labels), cic_output2).to(self.__device)
                _labels = _labels.scatter_(1, labels.unsqueeze(1), 1)

                loss = loss_func(preds, _labels)
            
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, 
                        self.__epoch, train_loss / len(train_loader)))

        torch.save(model, model_path)
        
        end = time.time()
        run_time = end - start
        print('run time: %.2fs.' % run_time)


    def FTTrain(self, train_loader, model_path, base_path):
        start = time.time()

        model = torch.load(base_path)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.__lr)
        loss_func = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(self.__epoch):
            train_loss = 0
            for (X, labels) in train_loader:
                X = Variable(X.float())
                labels = Variable(labels.long())
                if self.__cuda: 
                    X = X.cuda()
                    labels = labels.cuda()

                preds, _ = model(X)
                
                _labels = torch.zeros(len(labels), cic_output2).to(self.__device)
                _labels = _labels.scatter_(1, labels.unsqueeze(1), 1)

                loss = loss_func(preds, _labels)
            
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, 
                        self.__epoch, train_loss / len(train_loader)))

        torch.save(model, model_path)
        
        end = time.time()
        run_time = end - start
        print('run time: %.2fs.' % run_time)


    def Test(self, test_loader, set_type, model_path, smt_path=None):
        _model = torch.load(model_path)
        _model = _model.to(self.__device)
        _model.eval()  
        
        total, correct = 0, 0
        index = 0
        
        for (X_test, labels) in test_loader:
            X_test = Variable(X_test.float())
            labels = Variable(labels.long())
            if self.__cuda:
                X_test = X_test.cuda()
                labels = labels.cuda()

            y, _smt = _model(X_test)
            
            _p = nn.functional.softmax(y, dim=1)
            predict = torch.argmax(_p, axis=1)

            total += X_test.size(0)
            correct += (predict == labels).sum()
        
            if smt_path != None:
                if _smt.shape[1] != 1:
                    _smt = torch.flatten(_smt, 1, 2)
                else:
                    _smt = _smt
                    print(model_path + ": ", _smt.shape)

            if index == 0:
                ret = predict.cpu().data
                if smt_path != None:
                    semantic = torch.cat((_smt.cpu().data, labels.cpu().data), axis=1)
            else:
                ret = torch.cat((ret, predict.cpu().data), axis=0)
                if smt_path != None:
                    tmp = torch.cat((_smt.cpu().data, labels.cpu().data), axis=1)
                    semantic = torch.cat((semantic, tmp), axis=0)

            index = index + 1

        if smt_path != None:
            semantic = pd.DataFrame(semantic.numpy())
            semantic.to_csv(smt_path, index=0)

        print('Test Accuracy of the model on the %s set: %4f %%' 
                % (set_type, 100.0 * correct / total))
        return ret
    