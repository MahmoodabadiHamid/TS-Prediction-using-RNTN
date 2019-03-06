import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim




class RNTN(nn.Module):
    def __init__(self):
        super(RNTN, self).__init__()
        self.L1 = nn.Linear(10,10)
        self.L2 = nn.Linear(10,10)
        self.L3 = nn.Linear(10,10)
        self.L4 = nn.Linear(10,10)
        self.L5 = nn.Linear(10,10)
        self.out = torch.zeros(1,5)
        
    def forward(self, x):
        x =  x.resize(10)
        self.out[0,0] = torch.mv(self.L1(x).resize(1,10) , x)
        self.out[0,1] = torch.mv(self.L2(x.resize(1,10)) , x)
        self.out[0,2] = torch.mv(self.L3(x.resize(1,10)) , x)
        self.out[0,3] = torch.mv(self.L4(x.resize(1,10)) , x)
        self.out[0,4] = torch.mv(self.L5(x.resize(1,10)) , x)        
        
        return F.softmax(self.out, dim=1)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.L = nn.Linear(5,10)
            
    def forward(self, x):
        return self.L(x)


 
if __name__ == '__main__':
    import RNTN
    RNTN.main()