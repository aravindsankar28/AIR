import gensim
from gensim.models import word2vec
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

def wordMatrix(line):
	arr = np.array([])
	index = 0
	for word in line:
		print index
		index+=1
		if (len(arr) == 0):
			arr = np.append(arr, model1[word])
		else:
			arr = np.vstack((arr, model1[word]))
	return arr

model1 = word2vec.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
texts = []
i = 0
train = open("train_docs.txt")
lineIndex = 0
line1 = []
for line in train:
	print lineIndex
	line = line.split(" ")
	line_words = []
	for word in line:
		if len(word)>0 and (word in model1):
			if "-" in word:
				words = word.split("-")
				for w in words:
					if len(w)>0:
						line_words.append(w)
			else:
				line_words.append(word)

	similar_words = []
	for word in line_words:
		line1.append(word)
		#similar = model1.most_similar(word)
		#for sim1 in similar:
		#	if sim1[0] not in line_words:
	# 			similar_words.append(sim1[0])
	# for sim1 in similar_words:
	# 	line_words.append(sim1)
	texts.append(line_words)

print texts


# embeddingMatrix_n_300_list = []
# for line in texts:
# 	embeddingMatrix_n_300_list.append(wordMatrix(line))
	

# for i in range(0,len(embeddingMatrix_n_300_list)):
#arr = embeddingMatrix_n_300_list[i]
arr = wordMatrix(line1)
units = {}
pca = PCA(n_components=300)
pca.fit(arr)
#arr_2 = pca.transform(arr)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
numDims = 0
while var1[numDims] < 85:
	numDims+=1
pca = PCA(n_components=numDims)
pca.fit(arr)
arr_reduced = pca.transform(arr)	
dpgmm = mixture.BayesianGaussianMixture(
n_components=5, covariance_type='full', weight_concentration_prior=1e-2,
weight_concentration_prior_type='dirichlet_process',
mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(numDims),
init_params="random", max_iter=100, random_state=2).fit(arr_reduced)
text_line = texts[i]
i = 0
for c in dpgmm.predict(arr_reduced):
	if c not in units:
		units[c] = []
	units[c].append(text_line[i])
	i+=1
for u in units:
	for w in units[u]:
		print w + ' ',
	print
