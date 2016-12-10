import numpy as np
from numpy import *
import math
from sklearn import mixture
import sys
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
import random, operator

embedding_file= "embeddings_snippet.txt"
similarWords_file = "similarWords_snippet_100.txt"
directory = "Evaluation/DMM_results"
numTopics = 5
f0 = directory+"/wordMap_"+str(numTopics)+".txt"
f1 = directory+"/topic_priors_"+str(numTopics)+".txt"
f2 = directory+"/word_topic_probs_"+str(numTopics)+".txt"
docFile = "docs_nodups.txt"
alpha = 0.5

def getRelatedWords(w, word_related, word_emb, Pz_given_w_dmm, wordMap):
	# Find related words based on embeddings and 
	toAdd = {}
	for word in word_related[w]:
		val = alpha * (1-cosine(word_emb[w], word_emb[word])) + (1 - alpha)*(1 -cosine(Pz_given_w_dmm[wordMap[w]], Pz_given_w_dmm[wordMap[word]]))
		if val > (alpha*0.3 +(1- alpha)*0.5):
			# print val, word
			toAdd[word] = val
	x = sorted(toAdd.items(), key = operator.itemgetter(1), reverse = True)
	words = []
	for val in x[0:10]:
		words.append(val[0])
	return words

def sumOverWords(Doc, Pz, Pwz, wordMap):
	# Pwz is P(z|w) for word w
	#Pdz = [] # P(z|d) for each doc. Indexed by doc first.
	temp = []
	for k in xrange(0,len(Pwz)):
		val = 0.0
		for w in Doc:
			if w not in wordMap:
				continue
			val += Pz[k]*Pwz[k][wordMap[w]] * Doc.count(w)*1.0/len(Doc)
		temp.append(val)
	return temp

def groupSplitDMM(st, Pz_given_w_dmm, wordMap):
	groups = {}
	for w in st:
		prob_w = Pz_given_w_dmm[wordMap[w]]
		max_prob = 0
		max_k= -1
		for k in range(0, len(prob_w)):
			if prob_w[k] > max_prob:
				max_prob = prob_w[k]
				max_k = k
		if max_k not in groups:
			groups[max_k] = []
		groups[max_k].append(w)
	return groups


word_emb = {} # Emb for each word
with open(embedding_file) as f :
	lines = f.read().splitlines()
	for line in lines:
		w = line.split()[0]
		emb_str = line.split()[1:len(line.split())]
		emb = []
		for val in emb_str:
			emb.append(float(val))
		word_emb[w] = emb



word_related = {} # 100 related words for each word.
with open(similarWords_file) as f :
	lines = f.read().splitlines()
	for line in lines:
		w = line.split()[0]
		related = line.split()[1:len(line.split())]
		word_related[w] = related

short_texts=[]
i = 0
train = open(docFile)
lineIndex = 0
line1 = []

docs_new = []
for line in train:
	line = line.split("\n")[0]
	line = line.split(" ")
	line_words = []
	for word in line:
		if word in word_related:
			line_words.append(word)
			#line_words.extend(word_related[word])
	short_texts.append(line_words)
	#texts.append(line_words)

# Read the DMM topic model outputs
wordMapRev ={}
wordMap = {}
with open(f0) as f:
	lines = f.read().splitlines()
	for line in lines:
		i = int(line.split()[0])
		word = line.split()[1]
		wordMapRev[i] = word
		wordMap[word] = i

Pz_dmm = [] # P(z) for each topic.
with open(f1) as f:
	lines = f.read().splitlines()
	for line in lines:
		Pz_dmm.append(float(line))

Pwz_dmm = {} # P(w|z) for each topic. Dict indexed by topic first.

with open(f2) as f:
	lines = f.read().splitlines()
	count = 0
	for line in lines:
		Pwz_dmm[count] = []
		split = line.split()
		for val in split:
			Pwz_dmm[count].append(float(val))
		count += 1

Pz_given_w_dmm = {}
# Compute P(z|w) for each word w.
for w in wordMapRev:
	Pz_given_w_dmm[w] = []
	for z in Pwz_dmm:
		Pz_given_w_dmm[w].append(Pz_dmm[z]*Pwz_dmm[z][w])

expanded_short_texts = []
# Need to add related words based on embedding and gpu idea.
for st in short_texts:
	Pdz =sumOverWords(st, Pz_dmm, Pwz_dmm, wordMap)
	Pdz = normalize([Pdz])[0]
	#print st
	expanded_st = []
	for w in st:
		# decide if we promote word.
		
		Pwz = Pz_given_w_dmm[wordMap[w]] # topic dist. for word w
		#Pdz = normalize(Pdz)
		Pwz = normalize([Pwz])[0]
		Sw = 1 - cosine(Pdz, Pwz) # switch
		Sw = Sw**2 
		x = random.random()
		expanded_st.append(w)
		if x < Sw:
			# promote it. 
			newWords = getRelatedWords(w, word_related, word_emb, Pz_given_w_dmm, wordMap)
		 	expanded_st.extend(newWords)

		# We have expanded short text.
	expanded_st = list(set(expanded_st))
	groups = groupSplitDMM(expanded_st, Pz_given_w_dmm, wordMap)
	# Break into units based on DMM.
	for g in groups:
		print ' '.join(groups[g])
	

