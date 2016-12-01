import sys

filename = sys.argv[1]

biterms = set()

with open(filename) as f:
	lines = f.read().splitlines()
	for doc in lines:
		words = doc.split()
		for i in range(0, len(words)):
			for j in range(i+1, len(words)):
				if words[i] <= words[j]:
					biterm = words[i]+" "+words[j]
					biterms.add(biterm)
for biterm in biterms:
	print biterm