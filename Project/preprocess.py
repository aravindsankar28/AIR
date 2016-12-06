import sys
import re
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

wordDocumentFrequency = {}

name = sys.argv[1]
docs = []
with open(name) as f:
	lines = f.read().splitlines()
	for line in lines:
		pattern = re.compile('([^\s\w]|_)+')
		strippedList = pattern.sub('', line)
		wordList = strippedList.split()
		wordListNew = []
		for word in wordList:
			word = word.lower()
			if len(word) >3 and word not in stop:
				wordListNew.append(word)
		docs.append(wordListNew)
		for w in wordListNew:
			if w not in wordDocumentFrequency:
				wordDocumentFrequency[w] = 0
			wordDocumentFrequency[w] += 1
		#print ' '.join(wordListNew)


for d in docs:
	wordListNew = []
	for w in d:
		if wordDocumentFrequency[w] >=3 and w not in wordListNew:
			wordListNew.append(w)
	print ' '.join(wordListNew)