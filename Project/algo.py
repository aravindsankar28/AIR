# For a short text, given pairwise similarity for the words in it, identify the units.
from gensim.models import word2vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import random

# D- distance matrix, k
def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in xrange(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C


stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Load DMM results

wordMap = {} # id to word
wordMapRev = {} # word to id

with open('Evaluation/DMM_results/wordMap_40.txt') as f:
	lines = f.read().splitlines()
	for line in lines:
		a = int(line.split()[0])
		b = line.split()[1]
		wordMap[a] = b
		wordMapRev[b] = a
print len(wordMap)

Pwz = {} # P(w|z) for each topic. Dict indexed by topic first.

with open('Evaluation/DMM_results/word_topic_probs_40.txt') as f:
	lines = f.read().splitlines()
	count = 0
	for line in lines:
		Pwz[count] = []
		split = line.split()
		for val in split:
			Pwz[count].append(float(val))
		count += 1

Pz = [] # P(z) for each topic.
with open('Evaluation/DMM_results/topic_priors_40.txt') as f:
	lines = f.read().splitlines()
	for line in lines:
		Pz.append(float(line))
Pz_given_w = {}
# Compute P(z|w) for each word w.
for w in wordMap:
	Pz_given_w[w] = []
	for z in Pwz:
		Pz_given_w[w].append(Pz[z]*Pwz[z][w])

docs = []

with open('train_processed_nodups.txt') as f:
	lines = f.read().splitlines()
	for doc in lines:
		doc = doc.split()
		docs.append(doc)

data = np.array([[1,1], 
                [2,2], 
                [10,10]])
D = pairwise_distances(data, metric='euclidean')

M,C =  kMedoids(D, 2)

for label in C:
    for idx in C[label]:
    	print label, idx, data[idx]


model = word2vec.Word2Vec.load_word2vec_format('/Users/aravind/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

expanded_docs = []
for doc in docs:
	doc_new = []
	for word in doc:
		similar= model.most_similar(word)
		for w in similar:
			w = lemmatizer.lemmatize(w)
			if w not in doc_new and w not in stop and len(w)>3 and w in wordMapRev:
				doc_new.append(w)
	print doc_new
	expanded_docs.add(doc_new)





