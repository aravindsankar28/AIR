from gensim.models import word2vec

model_org = word2vec.Word2Vec.load_word2vec_format('/Users/aravind/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
textfile = open("../train_docs_new.txt")
textfile2 = open("../train_docs_noSpecialChar_noUselessWord","w")
vocab = {}
i = 0
for line in textfile:
	line1 = ""
	i+=1
	print i
	line = line.split("\n")[0]
	line = line.split(" ")
	skip = 0
	for word in line:
		if word in model_org:
			if word not in vocab:
				vocab[word]  = 0
			if skip == 0:
				line1 = line1 + word
				skip = 1
			else:
				line1 = line1 + " " + word
	textfile2.write(line1+"\n")
		
textfile2.close()
emb = open("embeddings.txt","w")
skip = 0
i = 0
for word in vocab:
	i+=1
	print i
	if skip == 0:
		emb.write(word)
		skip = 1
	else:
		emb.write(" "+word)
emb.write("\n")
for word in vocab:
	skip = 0
	for element in model_org[word]:
		if skip == 0:
			emb.write(str(element))
			skip = 1
		else:
			emb.write(" "+str(element))
	emb.write("\n")
emb.close()
