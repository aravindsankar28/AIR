
# Get most similar words for each word based on embeddings 
from scipy.spatial.distance import cosine
from heapq import heappush, heappop, heapreplace
import operator

word_emb = {}
with open('embeddings_tmn.txt') as f:
	lines = f.read().splitlines()
	for l in lines:
		w = l.split()[0]
		emb_str = l.split()[1:len(l.split())]
		emb = []
		for val in emb_str:
			emb.append(float(val))
		word_emb[w] = emb


h = []
K = 100

f = open('similarWords_tmn_100.txt','w')
i =0 

for w in word_emb:
	print i
	i += 1
	d = {}
	for w_1 in word_emb:
		if w_1 != w:
			val = cosine(word_emb[w],word_emb[w_1])	
			d[w_1] = val
	s = sorted(d.items(),key=operator.itemgetter(1))
	f.write(w)
	for j in range(0,K):
		f.write(" "+s[j][0])
	f.write("\n")	
f.close()

# for w in word_emb:
# 	print i
# 	i += 1
# 	h = []
# 	for w_1 in word_emb:
# 		if w_1 != w:
# 			val = 1 -cosine(word_emb[w],word_emb[w_1])
# 			if len(h) < K:
# 				heappush(h, (val, w_1))
# 			elif val > h[0][0]:
# 				heapreplace(h, (val, w_1))
# 	f.write(w)
# 	for x in h:
# 		f.write(" "+x[1])
# 	f.write("\n")
# f.close()