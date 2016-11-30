import sys
import numpy as np
from random import randint

N = 5
alpha = 50.0/N
beta = 0.01
vocab = []
units = []
topics = []
assignments = []
nz_units = []
nz_words = []
nwz = {}
nzw = {}
numIter = 10

def initUnits(unitfile):
	unitFile = open(unitfile)
	for line in unitFile:
		unit = []
		line = line.split("\n")[0]
		line = line.split(" ")
		for w in line:
			if w not in vocab:
				vocab.append(w)
				nwz[w] = [0]*N
			unit.append(w)
		units.append(unit)
	return 0


def initTopics(vocab):
	alpha_array = [alpha]*len(vocab)
	for t in range(0,N):
		topic = np.random.dirichlet(alpha_array)
		topics.append(topic)
		nz_units.append(0)
		nz_words.append(0)
		nzw[t] = [0]*len(vocab)
	return 0

def initAssignments():
	for unit in units:
		topic = randint(0,N-1)
		nz_units[topic] += 1
		nz_words[topic] += len(unit)
		assignments.append(topic)
		for w in unit:
			nwz[w][topic] += 1
			nzw[topic][vocab.index(w)] += 1
	return 0


def gibbsIteration():
	array_topics = []
	for t in range(0,N):
		array_topics.append(t)
	for unit in units:
		current_topic = assignments[units.index(unit)]
		probs = [0.0]*N
		sum_probs = 0.0
		for t in range(0,N):
			probs[t] = (nz_units[t]*1.0) + alpha
			for w in unit:
				probs[t] *= (nwz[w][t]*1.0 + beta)/(nz_words[t] + (len(vocab)*beta))
			sum_probs += probs[t]
		probs_normalized = []
		for p in probs:
			probs_normalized.append(p/sum_probs)
		#print probs_normalized
		new_topic = np.random.choice(array_topics,1,p=list(probs_normalized))[0] 

		for w in unit:
			nwz[w][new_topic] += 1
			nzw[new_topic][vocab.index(w)] += 1
			nwz[w][current_topic] -= 1
			nzw[current_topic][vocab.index(w)] -= 1
		nz_units[current_topic] -= 1
		nz_units[new_topic] += 1
		nz_words[current_topic] -= len(unit)
		nz_words[new_topic] += len(unit)
		assignments[units.index(unit)] = new_topic



initUnits(sys.argv[1])
initTopics(vocab)
print len(vocab)
initAssignments()
for iterations in range(0,numIter):
	print iterations
	gibbsIteration()
for t in range(0,N):
	nzw[t].sort()
	nzw[t].reverse()
	for i in range(10):
		print nzw[t][i]," ",
	print "\n"

