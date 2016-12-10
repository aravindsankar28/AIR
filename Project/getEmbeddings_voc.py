# Obtain embeddings for all words in vocab

from gensim.models import word2vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import random
import heapq
from scipy.spatial.distance import cosine

# read google model
model = word2vec.Word2Vec.load_word2vec_format('/Users/aravind/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

vocab = []
with open('vocab_snippet.txt') as f:
	lines = f.read().splitlines()
	for l in lines:
		w = l.split()[1]
		vocab.append(w)

f = open('embeddings_snippet.txt','w')
for w in vocab:
	if w not in model:
		continue
	f.write(w+" ")
	f.write(str(model[w][0]))
	for i in range(1, len(model[w])):
		f.write(" "+str(model[w][i]))
	f.write("\n")
f.close()

