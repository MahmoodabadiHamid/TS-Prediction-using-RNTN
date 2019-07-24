# Time Series Prediction Using RNTN Method #
In this project, we are going to apply RNTN model to predict stock market time series.
RNTN mdoel proposed by Socher et al. in the paper Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank 
(Link to paper: https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

### Prerequisites ###

1. python 3.x

2. pytorch 
3. Numpy
4. pandas
5. sklearn

### Runnning the tests ###
Run RNTN.py


### ***Functions definition*** ###
## RNTN.py ##
#### readDataSet() ####
This function read the time series dataset from a csv file and reshape it to tree form then return a list of trees

#### doRNTN() ####
This fucntion recursively feed one tree to RNTN model

#### extractOneHot() ####
This function get a list of labeld and turn it to one-hot labels.

#### main() ####
Here is the main function that calls other functions in order to prepare data and train the RNTN network.




## Classes.py ##
#### RNTN(), RNN() ####
In this functions we define two model structures using pytorch package.


## Tree.py ##
#### Tree() ####
This class generates a tree by parsing the given Time Series and assigning its values to the tree's leaves.
Note that time series length must be odd! 

#### Node() ####
In this class we define a abstract node that can be leaf or non-leaf element of a tree.



















Graduate School of Management and Economics, Sharif University of ...
