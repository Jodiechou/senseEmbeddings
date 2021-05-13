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


def load_ares_txt(path):
		sense_vecs = {}
		with open(path, 'r') as sfile:
			for idx, line in enumerate(sfile):
				if idx == 0:
					continue
				splitLine = line.split(' ')
				label = splitLine[0]
				vec = np.array(splitLine[1:], dtype='float32')
				dim = vec.shape[0]
				# print('self.dim', self.dim)
				sense_vecs[label] = vec
		return sense_vecs


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def relu(x):
   return np.maximum(0, x)


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
	# print('new_values.shape', new_values.shape)


	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])
		
	plt.figure(figsize=(5, 5)) 

	# colors = ['r','b']
	# Label_Com = ['Component 1','Component 2']

	for i in range(len(x)):
		if i < 7 and i != 1:
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

	# path = 'data/figure/new_bank_sense_linear_useSense_50.png'
	path = 'data/figure/new_bank_sense_glove_50.png'

	savefig(path, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

if __name__ == "__main__":


	sv_path = 'data/vectors/senseMatrix.semcor_diagonal_linear_large_bertlast4layers_multiword_300_50.npz'
	# b_path = 'data/vectors/paramB.semcor_diagonal_gelu_BiCi_noval_1024_300_50.npz'
	# c_path = 'data/vectors/paramC.semcor_diagonal_gelu_BiCi_noval_1024_300_50.npz'
	ares_embedding_path = 'external/ares/ares_bert_large.txt'

	glove_embedding_path = 'external/glove/glove.840B.300d.txt'

	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")

	# words = ['money', 'currency', 'dollar', 'credit', 'expense', 'payment', 'sea', 'beach', 'hill', 'shore', 'mountain']
	# words = ['lump', 'torch', 'flashlight', 'headlight', 'candle', 'sidelight', 'portable', 'transportable', 'weightless', 'airy']
	# words = ['level', 'horizontal', 'even', 'smooth', 'consistent', 'plumb', 'apartment', 'house', 'penthouse', 'accommodation']
	# words = ['circle', 'circlet', 'band', 'round', 'loop', 'circuit', 'phone', 'call', 'dial', 'buzz', 'bell']
	# words = ['instruct', 'teach', 'coach', 'tutor', 'educate', 'develop', 'cavalcade', 'caravan', 'convoy', 'railway']
	# selected_senses = ['train%2:31:01::', 'train%1:06:00::']
	# senses = ['instruct%2:32:00::', 'teach%2:32:00::', 'ccoach%2:32:00::', 'tutor%2:32:00::', 'educate%2:41:00::', 'develop%2:31:00::', 
	# '', '', '', '', 'railway%1:06:01::']
	# senses = ['organization%1:14:00::', 'college%1:14:00::', 'university%1:14:00::', 'high_school%1:14:00::', 'school%1:14:00::', 'institution%1:14:00::', 
	# 'sea%1:17:00::', 'beach%1:17:00::', 'hill%1:17:00::', 'shore%1:17:00::', 'mountain%1:17:00::']
	senses = ['appropriation%1:21:00::', 'currency%1:21:00::', 'savings%1:21:00::', 'expense%1:21:00::', 'payment%1:21:00::', 
	'coast%1:17:00::', 'beach%1:17:00::', 'island%1:17:00::', 'canyon%1:17:00::', 'riverbank%1:17:00::']
	selected_senses = ['bank%1:14:00::', 'bank%1:17:00::']
	
	# selected_senses = []

	# relu = nn.ReLU(inplace=True)

	A = load_senseMatrices_npz(sv_path)
	# ares_embeddings = load_ares_txt(ares_embedding_path)
	# print('len of ares_embeddings', len(ares_embeddings))
	# B = load_senseMatrices_npz(b_path)
	# C = load_senseMatrices_npz(c_path)
	sense_embeddings = {}
	for sense in selected_senses:
		curr_lemma = lemmatize(sense)

		if sense == 'bank%1:14:00::':
			label = 'bank_01'
		else:
			label = 'bank_02'

		# A_matrix = A[sense]
		vec_g = glove_embeddings[curr_lemma]
		# senseVec = A_matrix * vec_g

		# senseVec = gelu(A_matrix * vec_g)

		# ares_vec = ares_embeddings[sense]
		# sense_embeddings[label] = ares_vec

		# cont_vec = np.concatenate((ares_vec, senseVec), axis=0)
		sense_embeddings[label] = vec_g

	for s in senses:
		lemma = lemmatize(s)
		# A_matrix_temp = A[s]
		vec_g = glove_embeddings[lemma]
		# vec = A_matrix_temp * vec_g
		# cont_vec = np.concatenate((ares_embeddings[s], vec), axis=0)
		sense_embeddings[lemma] = vec_g

		# sense_embeddings[lemma] = ares_embeddings[s]
		# sense_embeddings[lemma] = cont_vec



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

