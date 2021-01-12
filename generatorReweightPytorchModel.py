import torch
import numpy as np

class generatorReweightPytorchModel(torch.nn.Module) :
    def __init__(self, device) :
        super(generatorReweightPytorchModel, self).__init__()

        self._device = device
        
        self._nn = torch.nn.Sequential(
            # 6 inputs going into first hidden layer with 64 nodes, ReLU activation
            torch.nn.BatchNorm1d(23),
            torch.nn.Linear(23, 64), torch.nn.ReLU(),
            # into second hidden layer with 64 nodes, ReLU activation
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,64), torch.nn.ReLU(),
            # into second hidden layer with 64 nodes, ReLU activation
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,64), torch.nn.ReLU(),
            # into second hidden layer with 64 nodes, ReLU activation
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,64), torch.nn.ReLU(),
            # Three outputs
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,1)
            )
        
    def forward (self, x) :
        return self._nn(x)

def forward(blob, train=True) :
    with torch.set_grad_enabled(train) :
        data = torch.as_tensor(blob.data).type(torch.FloatTensor).to(blob.net._device)
        prediction = blob.net(data).view(-1)
        
        # Training
        los, acc = -1, -1
        if blob.label is not None :
            label = torch.as_tensor(blob.label).type(torch.FloatTensor).to(blob.net._device)
#            print("DATA SUM {0} MEAN {1} MIN {2} MAX {3}".format(data.sum(), data.mean(), data.min(), data.max()))
            if data.sum() != data.sum() :
                for i in range(len(data)) :
                    print(data[i])
                exit(2)
#            print(prediction)
#            print("MIN {0} MAX {1}".format(min(prediction), max(prediction)))
#            sigpred = torch.sigmoid(prediction)
#            print(sigpred)
#            print("MIN {0} MAX {1}".format(min(sigpred), max(sigpred)))
#            print(label)
#            print("MIN {0} MAX {1}".format(min(label), max(label)))
            loss = blob.criterion(prediction, label)
#            print(loss)
#            print("MIN {0} MAX {1}".format(min(loss), max(loss)))
#            print("LOSS MEAN {0}".format(loss.mean()))
#            print("LOSS SUM {0}".format(loss.sum()))
        blob.loss = loss.mean()
        
        return {'prediction' : prediction.cpu().detach().numpy(),
                'loss' : blob.loss.cpu().detach().item()}

def backward(blob) :
    blob.optimizer.zero_grad()
    blob.loss.backward()
    blob.optimizer.step()

def FillLabel(blob, data) :
    blob.label = data['label']

def FillData(blob, data) :
    blob.data = data['features']
