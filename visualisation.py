import matplotlib                                                                                                         
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import math
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
	tsne_model = TSNE(n_components=2, init='pca', perplexity=3, n_iter=1500, metric='cosine')
	new_values = tsne_model.fit_transform(tokens)

	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])
	plt.figure(figsize=(5, 5)) 

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
	path = 'data/figure/new_bank_sense_cdes.png'

	savefig(path, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

if __name__ == "__main__":


	sv_path = 'data/vectors/senseMatrix.semcor_diagonal_linear_large_bertlast4layers_multiword_300_50.npz'
	ares_embedding_path = 'external/ares/ares_bert_large.txt'
	glove_embedding_path = 'external/glove/glove.840B.300d.txt'
	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")

	senses = ['appropriation%1:21:00::', 'currency%1:21:00::', 'savings%1:21:00::', 'expense%1:21:00::', 'payment%1:21:00::', 
	'coast%1:17:00::', 'beach%1:17:00::', 'island%1:17:00::', 'canyon%1:17:00::', 'riverbank%1:17:00::']
	selected_senses = ['bank%1:14:00::', 'bank%1:17:00::']
	A = load_senseMatrices_npz(sv_path)
	ares_embeddings = load_ares_txt(ares_embedding_path)

	sense_embeddings = {}
	for sense in selected_senses:
		curr_lemma = lemmatize(sense)

		if sense == 'bank%1:14:00::':
			label = 'bank_01'
		else:
			label = 'bank_02'

		### For GloVe--------
		# if sense == 'bank%1:14:00::':
		# 	label = 'bank'
		###-----------

		A_matrix = A[sense]
		vec_g = glove_embeddings[curr_lemma]
		senseVec = A_matrix * vec_g
		ares_vec = ares_embeddings[sense]
		sense_embeddings[label] = ares_vec
		cont_vec = np.concatenate((ares_vec, senseVec), axis=0)
		sense_embeddings[label] = cont_vec

	for s in senses:
		lemma = lemmatize(s)
		A_matrix_temp = A[s]
		vec_g = glove_embeddings[lemma]
		vec = A_matrix_temp * vec_g
		cont_vec = np.concatenate((ares_embeddings[s], vec), axis=0)
		# sense_embeddings[lemma] = vec_g
		sense_embeddings[lemma] = cont_vec
		# sense_embeddings[lemma] = vec_g

	tsne_plot(sense_embeddings)

