import simMeasure as sm
import math

class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

def vector_len(v):
    return math.sqrt(sum([x*x for x in v]))

def dot_product(v1, v2):
    assert len(v1) == len(v2)
    return sum([x*y for (x,y) in zip(v1, v2)])

def euclidean_distance_similarity(v1, v2):
    return sqrt(sum(pow(a-b,2) for a, b in zip(v1, v2)))


def manhattan_distance(v1,v2):
    return sum(abs(a-b) for a,b in zip(v1,v2))


def cosine_similarity(v1, v2):
    """
    Returns the cosine of the angle between the two vectors.
    Results range from -1 (very different) to 1 (very similar).
    """
    return dot_product(v1, v2) / (vector_len(v1) * vector_len(v2))

#   words = load_words('data/words.vec')
def sorted_by_similarity(words, base_vector):
    """Returns words sorted by cosine distance to a given vector, most similar first"""
    words_with_distance = [(cosine_similarity(base_vector, w.vector), w) for w in words]

    print(words_with_distance[0])
    # We want cosine similarity to be as large as possible (close to 1)
    #return sorted(words_with_distance, key=lambda t: t[0], reverse=True)

def k_most_similar(corpus, word, k, sim_measure):
    from collections import Counter
    import collections
    dict ={}
    
    v = []
    
    simMeasure = sm.Similarity()
    for i in range(len(corpus)):
        if (corpus[i].text == word):
            v = corpus[i].vector	
            for i in corpus:
                if (sim_measure == 'cos' ):
                    
                    dict[i.text] = cosine_similarity(i.vector, v)
                elif(sim_measure == 'euc' ):
                    
                    dict[i.text] = simMeasure.euclidean_distance(i.vector, v)
                elif (sim_measure == 'man' ):
                    dict[i.text] = simMeasure.manhattan_distance(i.vector, v)
            d = Counter(dict)
            
            for k, v in d.most_common(k):
                print ('%s: %i' % (k, v))

    if(not bool(dict)):
        print('It seems this word does not appear in this corpus')
            
def plot_similarity(corpus, w1, w2):
    import matplotlib.pyplot as plt
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("A test graph")
    word1 = []
    word2 = []
    for d in corpus:
        if(d.text == w1):
            word1 = d.vector
        if(d.text == w2):
            word2 = d.vector
    #word = corpus[10]
    #word2 = corpus[10]
    if (word1 and word2):
        print(cosine_similarity(word1, word2))
        #for i in range(len(y)):
        plt.plot(range(len(word1[0:])), word1[0:], label = w1)
        plt.plot(range(len(word2[0:])), word2[0:], label = w2)
        plt.legend()
        plt.show()

def readWord_and_vector_from_file():
    file_path = ('data/words-short.vec')
    corpus = []
    with open(file_path, encoding="utf8") as f:
        #print(len(f))
        for line in f:
            try:
                line = line.split('\n')
                rawVector = list(line[0].split(' '))
                word = rawVector[0]
                wordVector = [float(i) for i in rawVector[1:-1]]
                if(len(wordVector) == 299):
                    corpus.append(Word(word, wordVector))
                else:
                    print(len(word))
            except:
                print('Error 404!')
        return corpus


def generateRandomVector():
    import random
    lst = [500]
    for i in range(299):
        lst.append(random.randint(int((lst[-1]  - lst[-1]*0.05)),int((lst[-1]  + lst[-1]*0.05)))) # This line generate random number between +-5% of last day
    return lst
        
def creatRandomStockTimeSerie():
    timeSerieBank = []
    for i in range(300):
        word = str('stock ' + str(i))
        wordVector = generateRandomVector()
        timeSerieBank.append(Word(word, wordVector))
    return timeSerieBank
#plot_similarity(' ')    
while False:
    k_most_similar(corpus ,input('word? '), 10, 'man')
















