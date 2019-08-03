import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import Classes 
import numpy  as np
from torch.autograd import Variable
from Tree import * 

def readDataSet():
    dataSet = []
    vector_len = 2
    treeDepth = 5
    df = pd.read_csv("dataSet/02_preprocessing_after_normalizing_values.csv", index_col=0)
    count = 0
    print("creatingTrees")
    for col in df.columns[:20]:
        
        for i in range(int(len(df[col])/vector_len)):
            data = torch.tensor(list(df[col][i*vector_len:i*vector_len+vector_len].values))
            label = 1 if int(bool(df[col][i*vector_len+vector_len] > df[col][i*vector_len+vector_len - 1])) else -1
            #label = 
            dataSet.append(Raw_Data(data, label))
    trees = []
    for i in range(int(len(dataSet)/ treeDepth)):
        trees.append(Tree(dataSet[i*treeDepth: i*treeDepth+treeDepth]))
    print("Number of trees: ", len(trees))
    print('Done!')
    
    return trees


def doRNTN(tree, RNTN):
    if (tree.l.l):# While we do not reach to leaves
        return RNTN(torch.cat((doRNTN(tree.l, RNTN)[0], tree.r.data),0))
    else:
        return RNTN(torch.cat((tree.l.data, tree.r.data), 0))
        

def extractOneHot(data):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    lbl = [dt.root.r.label for dt in data]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(lbl)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

def main():
    data  = readDataSet()
    one_hot_labels = extractOneHot(data)
    splitPoint = int(len(data)*0.7)
    X_train = data[:splitPoint]
    y_train = one_hot_labels[:splitPoint]
    X_test  = data[splitPoint:]
    y_test  = one_hot_labels[splitPoint:]
    
    RNTN = Classes.RNTN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(RNTN.parameters(), lr=0.01)
    trainLossArr = []
    testLossArr = []
    for epoch in range(100):
        trainLoss = 0.0
        testLoss = 0.0
        for i in range(len(X_train)):
            y_pred = F.softmax(doRNTN(X_train[i].root, RNTN)[0].view(-1,2), dim = 1)
            label = torch.tensor([y_train[i]])
            #print(torch.max(label, 1)[1])
            #print(y_pred)
            #input(criterion(y_pred, torch.max(label, 1)[1]))
            trainLoss =  torch.add(trainLoss, criterion(y_pred, torch.max(label, 1)[1]))
            if i%97 == 0:
                optimizer.zero_grad()
                trainLoss.backward(retain_graph=True)
                optimizer.step()
                trainLossArr.append(trainLoss)
                trainLoss = 0.0
                for i in range(len(X_test)):
                    y_pred = F.softmax(doRNTN(X_test[i].root, RNTN)[0].view(-1,2), dim = 1)
                    label = torch.tensor([y_test[i]])
                    testLoss =  torch.add(testLoss, criterion(y_pred, torch.max(label, 1)[1]))
                testLossArr.append(testLoss)
        print('epoch: ', epoch,' trainLoss: ', trainLoss.item())
    import matplotlib.pyplot as plt
    plt.plot(trainLossArr, 'r')
    plt.plot(testLossArr, 'g' )
    plt.show()
    print('Finished Training')
    return RNTN



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainedRNTN = main()



