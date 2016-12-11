# This script assumes the output of topic model as the input
# and evaluates the performance of document classification. 

# Input : topic distribution output -> over all words in the corpus, i.e. P(w|z) for all w and z.
# topic distribution global -> P(z) for all topics.
# word map file -> mapping from integer to words.


# Example run : python classifier.py DMM 20 Snippet
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import math
import sys
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

# These files need to be changed for testing different algos.
algo = sys.argv[1]
numTopics = int(sys.argv[2])
corpus = sys.argv[3]

directory = "Result_Files"

numLabels = -1
# Format -> algo_topics_corpus_zzz.txt
path = directory+"/"+algo+"_"+str(numTopics)+"_"+corpus
f0 = path+"_wordMap.txt"
f1 = path+"_topic_priors.txt"
f2 = path+"_word_topic_probs.txt"

# f0 = directory+"/wordMap_"+str(numTopics)+".txt"
# f1 = directory+"/topic_priors_"+str(numTopics)+".txt"
# f2 = directory+"/word_topic_probs_"+str(numTopics)+".txt"

f3 = corpus+"_Data/"+algo+"_docs_nodups.txt"
f4 = corpus+"_Data/topics.txt"
# f3 = "../train_processed_nodups.txt"
# f4 = "../test_processed_nodups.txt"
# f5 = "../train_topics.txt"
# f6 = "../test_topics.txt"



# Return P(d|z) for the set of input docs based on naive bayes
def naiveBayes(Docs):
	Pdz = [] # P(z|d) for each doc. Indexed by doc first.
	for d in range(0, len(Docs)):
		temp = []
		for k in xrange(0,topics):
			val = Pz[k]
			for w in Docs[d]:
				# TODO : for now ignore such words
				if w not in wordMap:
					continue
				val *= Pwz[k][wordMap[w]]
			temp.append(val)
		Pdz.append(temp)
	return Pdz


# Return P(d|z) for the set of input docs based on sum over words in that doc.
def sumOverWords(Docs):
	Pdz = [] # P(z|d) for each doc. Indexed by doc first.
	for d in range(0, len(Docs)):
		temp = []
		for k in xrange(0,topics):
			val = 0.0
			for w in Docs[d]:
				if w not in wordMap:
					continue
				val += Pz[k]*Pwz[k][wordMap[w]] * Docs[d].count(w)*1.0/len(Docs[d])
			temp.append(val)
		Pdz.append(temp)
	return Pdz

# Return biterms for a document
def getBiterms(doc):
	biterms = []
	for i in range(0, len(doc)):
		for j in range(i+1, len(doc)):
			biterm = words[i]+" "+words[j]
			biterms.add(biterm)
	return biterms

# def sumWordsBTM():
# 	Pdz = []
# 	for d in range(0, len(docs)):
# 		biterms = getBiterms[d]

 
def classifySVM(X, Y):
	X_total = X
	Y_total = Y
	X_total = normalize(X_total)
	clf = SVC(kernel='linear', C=1)
	z = cross_val_score(clf, X_total , Y_total, cv=5, scoring='accuracy')
	t = np.array(z)
	print np.mean(t)
	
	svc = SVC(kernel='linear',C=1).fit(X_total, Y_total)
	y_pred = svc.predict(X_total)
	print "full accuracy",accuracy_score(Y_total, y_pred)
	# for i in range(0, len(X)):
	# 	if Y_total[i] != y_pred[i]:
	# 		print i, docs[i], labelMapRev[Y_total[i]], labelMapRev[y_pred[i]]
 
	# X_train = normalize(X_train)
	# svc = LinearSVC(C=1).fit(X_train, Y_train)
	# svc = SVC(kernel='linear',C=1).fit(X, y)
	# y_pred = svc.predict(X_train)
	# print "train accuracy",accuracy_score(Y_train, y_pred)
	# y_pred = svc.predict(normalize(X_test))
	# print "test accuracy",accuracy_score(Y_test, y_pred)

# Compute NMI from clusters and labels based on BTM paper.
def NMI(clusters, labels, n):
	H_pred = 0.0
	H_labels = 0.0
	for i in clusters:
		H_pred += (len(clusters[i])*1.0) *math.log(len(clusters[i])*1.0/n, 2)
	H_pred /= n
	for j in labels:
		H_labels += (len(labels[j])*1.0) *math.log(len(labels[j])*1.0/n, 2)
	H_labels /= n
	Inf = 0.0
	for i in clusters:
		for j in labels:
			intersection = len(set(clusters[i]) & set(labels[j]))
			if intersection != 0:
				Inf += intersection*math.log(1.0*len(clusters[i]) * len(labels[j])/(n*intersection),2)
	Inf /= n
	return Inf*2/(H_pred+H_labels)

# Compute cluster purity from clusters and labels based on BTM paper.
def clusterPurity(clusters, labels, n):
	purity = 0.0
	for i in range(0, len(clusters)):
		max_score = 0
		for j in range(0, len(labels)):
			score = len(set(clusters[i]) & set(labels[j]))
			if score > max_score:
				max_score = score
	purity += max_score
	return purity/n

# Helper function to get groups/clusters from assignments. 
def getGroupsFromAssignments(ass):
	groups = {}
	for d in range(0, len(ass)):
		l = ass[d]
		if l not in groups:
			groups[l] = []
		groups[l].append(d)
	return groups

# Input : X - P(z|d) for each d, Y - labels for each doc. 
def clusterKMeans(X, Y, K):
	kmeans = KMeans(init='k-means++', n_clusters=K)
	kmeans.fit(X)
	pred_labels = kmeans.fit_predict(X)

	clusters = getGroupsFromAssignments(pred_labels)
	labels = getGroupsFromAssignments(Y)

	# print "purity for kmeans clustering ", clusterPurity(clusters, labels, len(pred_labels))
	print "nmi for kmeans clustering ",NMI(clusters, labels, len(pred_labels))

# Input : X - P(z|d) for each d, Y - labels for each doc. 
# Cluster based on assigning each document to best topic
def clusterNaive(X, Y):
	labels = {}
	clusters = {}
	pred_labels = []
	for d in range(0, len(X)):
		topic_probs = X[d]
		max_prob = -1
		max_topic = -1
		for t in range(0,len(topic_probs)):
			p = topic_probs[t]
			if p > max_prob:
				max_prob = p
				max_topic = t
		pred_labels.append(max_topic)

	clusters = getGroupsFromAssignments(pred_labels)
	labels = getGroupsFromAssignments(Y)
	print len(clusters), len(labels)
	# print "nmi metric naive clustering", normalized_mutual_info_score(np.array(Y), np.array(pred_labels))
	# print "purity for naive clustering ", clusterPurity(clusters, labels, len(pred_labels))
	print "nmi for naive clustering ",NMI(clusters, labels, len(pred_labels))
	

topics = 0
wordMap = {} # From word to word index
wordMapRev = {} # From word index to word
with open(f0) as f:
	lines = f.read().splitlines()
	for line in lines:
		i = int(line.split()[0])
		word = line.split()[1]
		wordMapRev[i] = word
		wordMap[word] = i

Pz = [] # P(z) for each topic.
with open(f1) as f:
	lines = f.read().splitlines()
	for line in lines:
		Pz.append(float(line))

Pwz = {} # P(w|z) for each topic. Dict indexed by topic first.

with open(f2) as f:
	lines = f.read().splitlines()
	count = 0
	for line in lines:
		Pwz[count] = []
		split = line.split()
		for val in split:
			Pwz[count].append(float(val))
		count += 1

topics = len(Pwz)

# Now, read docs and get P(z = k|d) for each doc - basically find the features for the document
docs = []

with open(f3) as f:
	lines = f.read().splitlines()
	for doc in lines:
		doc = doc.split()
		docs.append(doc)



#Pdz = naiveBayes(docs)

Pdz = sumOverWords(docs)

# read labels
labels = []
labelMap = {} # From topic to label
labelMapRev = {}
count = 0

with open(f4) as f:
	lines = f.read().splitlines()
	for line in lines:
		if line not in labelMap:
			labelMap[line] = count
			labelMapRev[count] = line
			count += 1
		labels.append(labelMap[line])

numLabels = len(labelMap)
# Evaluate document classification and clustering.
classifySVM(Pdz, labels)
clusterNaive(Pdz, labels)
#clusterKMeans(Pdz,labels,numLabels)
