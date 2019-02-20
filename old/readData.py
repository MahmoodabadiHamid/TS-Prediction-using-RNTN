import pandas as pd
from pandas import DataFrame
from pandas import Series
from matplotlib import pyplot


class Tree:
    def __init__(self,timeSerie):
        """
        Generate the tree by parsing the given Time Series.
        Args:    
        """
        
        self.root = self.parsTree(timeSerie, level=0)
               
    def parsTree(self, timeSerie, level):
        print(len(timeSerie))
        node = Node()
        if (len(timeSerie) == 2):
            print('==2')
            node.l = self.createLeaf(timeSerie.iloc[0],level)
            node.r = self.createLeaf(timeSerie.iloc[1],level)
            
        elif(len(timeSerie) > 2):
            print('>2')
            node.r = self.createLeaf(timeSerie.iloc[-1],level)
            node.l = self.parsTree(timeSerie.iloc[:len(timeSerie)-1,:], level+1)
            
        return node


    def createLeaf(self, timeSerie, level):
        import vocabulary
        leaf = Node()
        print(':')
        #print((timeSerie.shape))
        
        leaf.label = timeSerie['target']
        leaf.data = timeSerie.drop('target')
        leaf.word = vocabulary.vocab.addWord(str(leaf.data[0]))
        leaf.level = level
        return leaf

        '''
        print(timeSerie)
        if (len(timeSerie) == 1):
            #print(timeSerie['target'])
            node.label = timeSerie['target']
            #print((timeSerie))
            #node.data = timeSerie.drop('target')
            
            node.level = level
        elif (len(timeSerie) > 1):
            lTimeSerie = timeSerie.iloc[len(timeSerie)-1:]
            rTimeSerie = timeSerie.iloc[-1]
            node.l = self.parsTree(lTimeSerie, level+1)
            node.r = self.parsTree(rTimeSerie, level+1)
        '''


            
class Node():
    def __init__(self):
        self.l     = None
        self.r     = None
        self.level = None
        self.word  = None
        self.data  = None
        self.label = None
        self.output= None


def createLabel():
    import pandas as pd
    dt = pd.read_csv('sp500_original.csv', header=0)
    print(dt.shape) # must print -> (x,y)
    dt['target'] = dt['Close'].shift(-1) # this line add a feature to dataset
    print(dt.shape) # must print -> (x,y+1)
        
def readData():
    dt = pd.read_csv('trees/sp500.csv', header=0)
    dt = dt.iloc[:4,:].drop('Date', axis=1)
    dt['target'] = dt['Close'].shift(-1).dropna()

    dt = dt.dropna()
    #dt = dt.drop('Date', axis=1) #remove "Date" column from data set

    tree = Tree(dt)
    a = []
    a.append(tree)
    #print(tree.parsTree(dt, tree.root, level = 0))
    return a
    

if __name__ == '__main__':
    a = readData()





#df = DataFrame(data)

#df['target'] = df['Close'].shift(-1)

#df = df.dropna()


# load dataset
#series = Series.from_csv('car-sales.csv', header=0)
# display first few rows
#print(series.head(5))
# line plot of dataset
#data.plot()
#pyplot.show()
