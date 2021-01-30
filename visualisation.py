import matplotlib                                                                                                         
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import math
# import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import gensim
import matplotlib as mpl
import logging

tsne = TSNE(random_state=1, n_iter=1000, metric="cosine")

def lemmatize(w):
	lemma = w.split('%')[0]
	return lemma

def load_senseMatrices_npz(path):
	logging.info("Loading Pre-trained Sense Matrices ...")
	A = np.load(path, allow_pickle=True)    # A is loaded a 0d array
	A = np.atleast_1d(A.f.arr_0)            # convert it to a 1d array with 1 element
	A = A[0]                                # a dictionary, key is sense id and value is sense matrix 
	logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(A))
	return A


def load_senseMatrices_npz(path):
	logging.info("Loading Pre-trained B Matrices ...")
	B = np.load(path, allow_pickle=True)    # A is loaded a 0d array
	B = np.atleast_1d(B.f.arr_0)            # convert it to a 1d array with 1 element
	B = B[0]                                # a dictionary, key is sense id and value is sense matrix 
	return B


def load_senseMatrices_npz(path):
	logging.info("Loading Pre-trained C Matrices ...")
	C = np.load(path, allow_pickle=True)    # A is loaded a 0d array
	C = np.atleast_1d(C.f.arr_0)            # convert it to a 1d array with 1 element
	C = C[0]                                # a dictionary, key is sense id and value is sense matrix 
	return C


"""Get embeddings from files"""
def load_glove_embeddings(fn):
	embeddings = {}
	with open(fn, 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			word = splitLine[0]
			vec = np.array(splitLine[1:], dtype='float32')
			embeddings[word] = vec
	return embeddings


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def tsne_plot(sense_embeddings):
	"Creates and TSNE model and plots it"
	labels = []
	tokens = []

	for sense in sense_embeddings.keys():
		tokens.append(sense_embeddings[sense])
		labels.append(sense)



	# tokens = np.array(tokens)
	# tsne_model = TSNE(n_components=2, init='pca', random_state=64)
	# rs = sklearn.utils.check_random_state(None)
	# print('random state:', rs)
	tsne_model = TSNE(n_components=2, init='pca', perplexity=3, n_iter=1500, metric='cosine')
	
	# print('tokens', tokens)
	new_values = tsne_model.fit_transform(tokens)
	print('new_values.shape', new_values.shape)


	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])
		
	plt.figure(figsize=(5, 5)) 

	# colors = ['r','b']
	# Label_Com = ['Component 1','Component 2']

	for i in range(len(x)):
		if i < 8 and i != 1:
			plt.scatter(x[i], y[i], c='r')
		else:
			plt.scatter(x[i], y[i], c='b')
		plt.annotate(labels[i],
					 xy=(x[i], y[i]),
					 xytext=(5, 2),
					 textcoords='offset points',
					 ha='right',
					 va='bottom')
	# plt.show()

	path = 'data/figure/bank11_sense_gelu_200.png'

	savefig(path, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

if __name__ == "__main__":


	sv_path = 'data/vectors/senseMatrix.semcor_diagonal_gelu_large_300_200.npz'
	# b_path = 'data/vectors/paramB.semcor_diagonal_gelu_BiCi_noval_1024_300_50.npz'
	# c_path = 'data/vectors/paramC.semcor_diagonal_gelu_BiCi_noval_1024_300_50.npz'

	glove_embedding_path = 'external/glove/glove.840B.300d.txt'

	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")

	words = ['money', 'fund', 'cash', 'mortgagee', 'depository', 'lender', 'cay', 'beach', 'coast', 'shore', 'riverside']
	# words = ['lump', 'torch', 'flashlight', 'headlight', 'candle', 'sidelight', 'portable', 'transportable', 'weightless', 'airy']
	# words = ['level', 'horizontal', 'even', 'smooth', 'consistent', 'plumb', 'apartment', 'house', 'penthouse', 'accommodation']
	# words = ['circle', 'circlet', 'band', 'round', 'loop', 'circuit', 'phone', 'call', 'dial', 'buzz', 'bell']
	# words = ['instruct', 'teach', 'coach', 'tutor', 'educate', 'develop', 'procession', 'succession', 'cavalcade', 'caravan', 'convoy', 'railway']
	# t_word = 'bank'
	# selected_senses = ['train%2:31:01::', 'train%1:06:00::']
	selected_senses = ['bank%1:17:00::', 'bank%1:14:00::']
	
	# selected_senses = []

	relu = nn.ReLU(inplace=True)

	A = load_senseMatrices_npz(sv_path)
	# B = load_senseMatrices_npz(b_path)
	# C = load_senseMatrices_npz(c_path)
	sense_embeddings = {}
	# for sense in A.keys():
	for sense in selected_senses:
		# print('sense', sense)
		curr_lemma = lemmatize(sense)
		# if curr_lemma not in glove_embeddings:
			# continue
		# if curr_lemma == t_word:

		A_matrix = A[sense]
		# 	# B_matrix = B[sense]
		# 	# C_matrix = C[sense]
		vec_g = glove_embeddings[curr_lemma]

		# senseVec = A_matrix * vec_g
		senseVec = gelu(torch.from_numpy(A_matrix * vec_g))
		senseVec = senseVec.cpu().detach().numpy()
		# 	# senseVec = gelu(torch.from_numpy(A_matrix * vec_g))
		# 	# senseVec = gelu(torch.from_numpy(C_matrix) * gelu(torch.from_numpy(B_matrix) * gelu(torch.from_numpy(A_matrix * vec_g))))
		# 	# sense_embeddings[sense] = senseVec.cpu().detach().numpy()
		if sense == 'bank%1:14:00::':
			label = 'bank_01'
		else:
			label = 'bank_02'
		sense_embeddings[label] = senseVec
		# 	# selected_senses.append(sense)

	for w in words:
		sense_embeddings[w] = glove_embeddings[w]

	# print('sense_embeddings', sense_embeddings)
	# print('selected_senses', selected_senses)
	tsne_plot(sense_embeddings)

	# import nltk
	# nltk.download('wordnet')
	# from nltk.corpus import wordnet   #Import wordnet from the NLTK
	# synset = wordnet.synsets("bank")
	# print('Word and Type : ' + synset[0].name())
	# print('Synonym of Travel is: ' + synset[0].lemmas()[0].name())
	# print('The meaning of the word : ' + synset[0].definition())
	# print('Example of Travel : ' + str(synset[0].examples()))

