import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.L = nn.Linear(10,5)
            
    def forward(self, x):
        x =  x.resize(10)
        return self.L(x)


class RNTN(nn.Module):
    def __init__(self):
        super(RNTN, self).__init__()
        self.vector_len = 2
        self.L1 = nn.Linear(self.vector_len * 2 ,self.vector_len * 2 )
        self.L2 = nn.Linear(self.vector_len * 2 ,self.vector_len * 2 )
        self.output_vector = torch.zeros(1,self.vector_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        x =  x.resize(self.vector_len * 2 )
        self.output_vector[0,0] = torch.mv(self.L1(x.resize(1,self.vector_len * 2 )) , x)
        self.output_vector[0,1] = torch.mv(self.L2(x.resize(1,self.vector_len * 2 )) , x)
        return torch.tanh(self.output_vector)


 
if __name__ == '__main__':
    import RNTN
    RNTN.main()
