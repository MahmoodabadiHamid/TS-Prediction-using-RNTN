from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import Classes 
import numpy  as np
from torch.autograd import Variable

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RNTN = Classes.RNTN().to(device)
    RNN  = Classes.RNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(RNTN.parameters(), lr=0.001, momentum=0.9)
    
    
    
    
    x = torch.randn(10, 1)
    y = torch.tensor([[0.9, 0.0, 0.1, 0.0, 0.0]])

    
    
    
    
    
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    
    
    
    for epoch in range(100):
        # Forward Propagation
        y_pred = RNTN(x)# model(x)
        # Compute and print loss
        #print(y.shape)
        #input(y_pred.shape)
        
        loss = criterion(y_pred, y)
        print('epoch: ', epoch,' loss: ', loss.item())
    
        # Zero the gradients
        optimizer.zero_grad()
        
        # perform a backward pass (backpropagation)
        loss.backward(retain_graph=True)
        
        # Update the parameters
        optimizer.step()

    print('Finished Training')
 
if __name__ == '__main__':
    
    main()
