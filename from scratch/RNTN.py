from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import Classes 
import numpy  as np
from torch.autograd import Variable

def main():
    RNTN = Classes.RNTN()
    RNN  = Classes.RNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(RNTN.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
    
            running_loss = 0.0
        
            # get the inputs
            inputs, labels = torch.randn(10,1), torch.zeros(5,1)
            labels[2] = 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = RNTN(inputs)
            
            
            #values, indices = torch.max(inputs, 0)
            
            maximum=0
            for i in range(1,5):
                if outputs[0, i] > outputs[0, maximum]:
                    maximum=i
            one_hot=np.zeros((5,1))
            one_hot[maximum,0]=1
            
            
            outputs= Variable(torch.from_numpy(one_hot), dtype=torch.float)
            labels = Variable(labels, dtype=torch.float)
            

            
            loss = criterion(outputs, labels)
            
            print(type(outputs))
            input(type(labels))
            
            loss.backward()
            optimizer.step()
            
            print(':)')
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
 
if __name__ == '__main__':
    
    main()
