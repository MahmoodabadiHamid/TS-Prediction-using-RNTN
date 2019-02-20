    #!/usr/bin/env python3

"""
Class which contain a sentence on the form of a tree
"""
import re # Regex to parse the file
import numpy as np
import vocabulary
import utils

import os
os.chdir('trees')
import pandas as pd
#data = pd.read_csv('sp500.csv',header = 0)

class Node():
    def __init__(self):
        self.l     = None
        self.r     = None
        self.level = None
        self.word  = None
        self.data  = None
        self.label = -1
        self.output= None




class Tree:
    def __init__(self,timeSerie):
        """
        Generate the tree by parsing the given Time Series.
        Args:    
        """
        
        self.root = self.parsTree(timeSerie, level=0)
    
        
    def parsTree(self, timeSerie, level):
        
        node = Node()
        node.level = level
       
        
        if  (timeSerie.iloc[-1]['target']  > timeSerie.iloc[-2]['target']):
            node.label = 1
        elif(timeSerie.iloc[-1]['target']  < timeSerie.iloc[-2]['target']):
            node.label = 0
        #elif(timeSerie.iloc[-1]['target']  == timeSerie.iloc[-2]['target']):
        #    node.label = 2
        
        
         

        if (len(timeSerie) == 2):
            node.l = self.createLeaf(timeSerie.iloc[0],level+1)
            node.r = self.createLeaf(timeSerie.iloc[1],level+1)
            
        elif(len(timeSerie) > 2):
            node.r = self.createLeaf(timeSerie.iloc[-1],level)
            node.l = self.parsTree(timeSerie.iloc[:len(timeSerie)-1,:], level+1)
            
        return node


    def createLeaf(self, timeSerie, level):
        leaf = Node()

        
        if(timeSerie['target'] > timeSerie['Close']):
            leaf.label = 1
        elif(timeSerie['target'] < timeSerie['Close']):
            leaf.label = 0
        #elif(timeSerie['target'] == timeSerie['Close']):
        #    leaf.label = 2
        
        
        
        leaf.data = timeSerie.drop('target')
        #leaf.word = timeSerie.drop('target')
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


            
        
    def printTree(self):
        """
        Recursivelly print the tree
        """
        if(self.root != None):
            self._printTree(self.root, 0)

    def _printTree(self, node, level):
        if(node != None):
            if node.word is None:
                self._printTree(node.l, level+1)
                self._printTree(node.r, level+1)
            else: # Leaf
                for i in range(level):
                    print('  ', end="")
                print(node.word.string, " ", node.label)

    def _nbOfNodes(self, node):
        if node.word is None:
            return 1 + self._nbOfNodes(node.l) + self._nbOfNodes(node.r)
        return 1
