import pandas as pd
class Node():
    def __init__(self):
        self.l     = None
        self.r     = None
        self.level = None
        self.data  = None
        self.label = None
        self.output= None
        self.isLeaf = False
class Raw_Data():
    def __init__(self, data = None, label = None):
        self.data = data
        self.label = label

class Tree:
    def __init__(self,dataSet):
        """
        Generate the tree by parsing the given Time Series.
        ***Time serie len most be odd! ***
        """
        self.treeDepth = 2
        self.root = self.parsTree(dataSet,  level=0)        
    def parsTree(self, dataSet, level):
        node = Node()
        node.level = level
        if (len(dataSet) == 2):
            node.l = self.createLeaf(dataSet[0], level+1)
            node.r = self.createLeaf(dataSet[1], level+1)
        elif(len(dataSet) > 2):
            node.r = self.createLeaf(dataSet[-1], level)
            node.l = self.parsTree(dataSet[:-1], level+1)
        return node
        
    def createLeaf(self, dt, level):
        leaf = Node()
        leaf.label = dt.label
        leaf.data  = dt.data
        leaf.level = level
        leaf.isLeaf = True
        return leaf

