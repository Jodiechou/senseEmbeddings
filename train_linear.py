import numpy as np 
from random import shuffle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

#from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

import logging
from datetime import datetime


# Get embeddings from files
def load_glove_embeddings():
	logging.info("Loading Glove Embeddings........")
	embed_g = []
	with open('data/vectors/glove_embeddings.32.test.txt', 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			vocab = splitLine[ :-299]
			vec_g = np.array(splitLine[-300:], dtype='float32')
			embed_g.append(vec_g)
	logging.info("Done. Loaded %d words from GloVe embeddings" % len(embed_g))
	return embed_g, vocab


def load_bert_embeddings():
	logging.info("Loading BERT Embeddings........")
	embed_c = []
	with open('data/vectors/bert_embeddings.32.test.txt', 'r') as cfile:
		for line in cfile:
			splitLine = line.split(' ')
			vec_c = np.array(splitLine[-1024:], dtype='float32')
			embed_c.append(vec_c)
	logging.info("Done. Loaded %d words from BERT embeddings" % len(embed_c))
	return embed_c


def lossFunctions(z):
	loss = torch.dot(z, z)
	return loss

def forward(input1, input2, w, matrix_A_temp):
	#print("input1", input1)
	#print("input2", input2.item())
	z = torch.matmul(input1, w)-torch.matmul(input2, matrix_A_temp)
	#print("z before", z)
	z = z.view(-1)
	#print("z", z)
	return z



if __name__ == '__main__':

	# Initialising inputs (contextualised embeddings and word embeddings), and the matrics W and A_i 
	# input_c and input_g should only contain the vectors instead of word+vectors
	# train_counter = []
	# valid_counter = []
	# train_loss =[]
	# valid_loss = []
	matrix_A = []
	weight = []

	# min_valid_loss = np.inf
	# iteration_number_train = 0
	# iteration_number_valid = 0
	#EPOCH = 20
	#batch_size = 200 
	#validation_split = 0.2
	shuffle_dataset = True
	lr = 0.00001

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	vocab = load_glove_embeddings()[1]

	embed_g = load_glove_embeddings()[0]
	embed_c = load_bert_embeddings()
	embed_g = np.array(embed_g)
	embed_c = np.array(embed_c)

	print("Shape of embed_g: ", embed_g.shape)
	print("Shape of embed_c: ", embed_c.shape)


	embed_g_shuf = []
	embed_c_shuf = []
	index_shuf = list(range(len(embed_g)))
	shuffle(index_shuf)
	for i in index_shuf:
		embed_g_shuf.append(embed_g[i])
		embed_c_shuf.append(embed_c[i])

	embed_g_shuf = np.array(embed_g_shuf)
	embed_c_shuf = np.array(embed_c_shuf)


	# create training and testing vars
	#g_train, g_test, c_train, c_test = train_test_split(embed_g_shuf, embed_c_shuf, test_size=0.2)

	g_train = torch.from_numpy(embed_g_shuf)
	c_train = torch.from_numpy(embed_c_shuf)

	# g_test = torch.from_numpy(g_test)
	# c_test = torch.from_numpy(c_test)

	a = [Variable(torch.randn(300, 300), requires_grad=True) for _ in range(len(g_train))]
	w = Variable(torch.randn(1024, 300), requires_grad=True)



	print("Training")
	for i in range(len(g_train)):
		matrix_A_temp = a[i]
		matrix_A_temp = Variable(torch.Tensor(matrix_A_temp), requires_grad=True)

		for epoch in range(50):
			# Training
			#total_train_loss = []
			#print('embed_g[i]', embed_g[i])
			
			contEmbed = c_train[i].view(-1, 1024)
			wordEmbed = g_train[i].view(-1, 300)
			#print("wordEmbed", wordEmbed)
			z = forward(contEmbed, wordEmbed, w, matrix_A_temp)
			#print('z', z)
			loss = lossFunctions(z)
			loss.backward()
			#print('\tgrad: ', w.grad.data[0], matrix_A_temp.grad.data[0])
			w.data = w.data - lr * w.grad.data
			matrix_A_temp.data = matrix_A_temp.data - lr * matrix_A_temp.grad.data

			w.grad.data.zero_()
			matrix_A_temp.grad.data.zero_()


			print("progress: %d , epoch: %d, loss: %f " %(i, epoch, loss.item()))
			#matrix_A_temp = 
		print('matrix_A size', matrix_A_temp.size())

		matrix_A.append(matrix_A_temp.data[0])
		#print('matrix_A size', matrix_A.size())
		# matrix_A = np.array(matrix_A)
		# print(matrix_A[0].shape())
		#weight.append(w)




	# print("Testing")
	# for i in range(len(g_test)):
	# 	matrix_A_test = matrix_A[i]
	# 	matrix_A_test = Variable(torch.Tensor(matrix_A_test))
	# 	weight = np.array(weight)
	# 	w_test = torch.from_numpy(weight)
	# 	w_test = Variable(torch.Tensor(w_test))

	# 	for epoch in range(25):
	# 		# Training
	# 		#total_train_loss = []
	# 		#print('embed_g[i]', embed_g[i])
			
	# 		cont_test = c_test[i].view(-1, 1024)
	# 		word_test = g_test[i].view(-1, 300)
	# 		#print("wordEmbed", wordEmbed)
	# 		z = forward(cont_test, word_test, w, matrix_A_test)
	# 		#print('z', z)
	# 		loss = lossFunctions(z)
	# 		#loss.backward()
	# 		#print('\tgrad: ', w.grad.data[0], matrix_A_temp.grad.data[0])
	# 		#w.data = w.data - lr * w.grad.data
	# 		#matrix_A_test.data = matrix_A_test.data - lr * matrix_A_test.grad.data

	# 		#w.grad.data.zero_()
	# 		#matrix_A_test.grad.data.zero_()


	# 		print("progress: %d , epoch: %d, loss: %f " %(i, epoch, loss.item()))

	senses = []

	with open('data/vectors/senses.32.test.txt', 'r') as sense_f:
		for line in sense_f:
			word = line.split(' ')[0]
			sense = line.split(' ')[1]
			senses.append(sense)
			

	print("senses", senses)

	print("senses", senses)
	print("matrix_A", matrix_A)



	out_path = 'data/vectors/sense_embeddings.32.txt'

	with open(out_path, 'w') as f:
		for sense, embed in zip(senses, matrix_A):
			embed_str = ' '.join([str(v) for v in embed])
			f.write('%s %s\n' % (sense, embed_str))
	logging.info('Written %s' % out_path)

