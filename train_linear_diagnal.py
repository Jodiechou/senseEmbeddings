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

import matplotlib
import matplotlib.pyplot as plt

import logging



logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


# Get embeddings from files
def load_glove_embeddings():
	embed_g = []
	senses = []
	with open('data/vectors/glove_embeddings.semcor.300.txt', 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			sense = splitLine[0]
			vec_g = np.array(splitLine[1:], dtype='float32')
			senses.append(sense)
			embed_g.append(vec_g)
	return embed_g, senses


def load_bert_embeddings():	
	embed_c = []
	with open('data/vectors/bert_embeddings.semcor.1024.txt', 'r') as cfile:
		for line in cfile:
			splitLine = line.split(' ')
			vec_c = np.array(splitLine[1:], dtype='float32')
			embed_c.append(vec_c)
	return embed_c


def lossFunctions(z):
	loss = torch.dot(z, z)
	return loss

def forward(input1, input2, w, matrix_A_temp):
	z = torch.matmul(input1, w)-torch.matmul(input2, matrix_A_temp)
	z = z.view(-1)
	return z



if __name__ == '__main__':


	
	mat_A = {}
	senEmbeds = {}

	lr = 0.1

	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info("Loading Glove Embeddings........")
	senses = load_glove_embeddings()[1]
	embed_g = load_glove_embeddings()[0]
	logging.info("Done. Loaded %d words from GloVe embeddings" % len(embed_g))

	logging.info("Loading BERT Embeddings........")
	embed_c = load_bert_embeddings()
	logging.info("Done. Loaded %d words from BERT embeddings" % len(embed_c))

	embed_g = np.array(embed_g)
	embed_c = np.array(embed_c)
	print("Shape of embed_g: ", embed_g.shape)
	print("Shape of embed_c: ", embed_c.shape)

	matrix_A = np.zeros((len(senses), 300, 300))
	embeds = np.zeros((len(senses), 300))


	embed_g_shuf = []
	embed_c_shuf = []
	index_shuf = list(range(len(embed_g)))
	shuffle(index_shuf)
	for i in index_shuf:
		embed_g_shuf.append(embed_g[i])
		embed_c_shuf.append(embed_c[i])

	embed_g_shuf = np.array(embed_g_shuf)
	embed_c_shuf = np.array(embed_c_shuf)



	g_train = torch.from_numpy(embed_g_shuf)
	c_train = torch.from_numpy(embed_c_shuf)


	dtype = torch.FloatTensor


	##### Learned vector dimension is 300
	##### i is the sense id
	#a = [Variable(torch.randn(300, 300), requires_grad=True) for _ in range(len(g_train))]
	#a = []
	w = Variable(torch.randn(1024, 300).type(dtype), requires_grad=True)
	matrix_A_temp = np.identity(300)

	#print('matrix_A_temp', matrix_A_temp)
	#print('shape of matrix_A_temp', matrix_A_temp.shape)
	matrix_A_temp = torch.from_numpy(matrix_A_temp).type(dtype)
	matrix_A_temp = Variable(matrix_A_temp, requires_grad=True)
	

	#matrix_A_temp = Variable(torch.randn(300, 300), requires_grad=True)
	optimizer = optim.Adam((matrix_A_temp, w), lr)


	# a = [Variable(torch.randn(300, 1024), requires_grad=True) for _ in range(len(g_train))]
	# w = Variable(torch.randn(1024, 1024), requires_grad=True)

	loss_count = 0
	loss_sum = 0



	logging.info("------------------Training-------------------")
	for epoch in range(20):

			for i in range(len(senses)):
				#matrix_A_temp = a[i]
				

				contEmbed = c_train[i].view(-1, 1024)
				wordEmbed = g_train[i].view(-1, 300)
				
				z = forward(contEmbed, wordEmbed, w, matrix_A_temp)
				optimizer.zero_grad()
				loss = lossFunctions(z)
				loss_sum += loss.item() 
				loss_count += 1
				loss.backward()
				optimizer.step()


				embed = torch.matmul(g_train[i], matrix_A_temp)
			
				#embeds.append(embed.detach().numpy())
				embeds[i] = embed.detach().numpy()
				#matrix_A.append(matrix_A_temp.detach().numpy())
				matrix_A[i] = matrix_A_temp.detach().numpy()
				



				#print("progress: %d , epoch: %d, loss: %f " %(i, epoch, loss.item()))

			average_loss = 	loss_sum / loss_count	
			print("epoch: %d, loss: %f " %(epoch, average_loss))

			

		
	embeds = np.array(embeds)
	matrix_A = np.array(matrix_A)
	print('matrix_a shape', matrix_A.shape)
	weight = w.detach().numpy()

	# print('embeddings', embeds[0])
	# print('matirx A', matrix_A[0])
	
	print('shape of sense embedding', embeds.shape)
	print('shape of each matrix A', matrix_A[0].shape)
	print('shape of weight matrix w', weight.shape)		


	# Build the structure of sense embeddings and A-matrix
	for n in range(len(senses)):
		mat_A[senses[n]] = matrix_A[n]
		senEmbeds[senses[n]] = embeds[n]


	logging.info("Total number of senses %d " %(len(senEmbeds)))



	matirx_path = 'data/vectors/senseMatrix.semcor.300.300.txt'
	with open(matirx_path, 'w') as f:
		for sen_str, matrix in mat_A.items():
			matrix_str = ' '.join([str(v) for v in matrix])
			f.write('%s %s\n' % (sen_str, matrix_str))
	logging.info('Written %s' % matirx_path)


	# matirx_path = 'data/vectors/senseMatrix.semcor.300.300.npz'
	# np.save(senseEmbed_path, matrix_A)
	# logging.info('Written %s' % matirx_path)



	senseEmbed_path = 'data/vectors/senseEmbed.semcor.300.txt'
	with open(senseEmbed_path, 'w') as sf:
		for sense_str, emb in senEmbeds.items():
			embed_str = ' '.join([str(v) for v in emb])
			sf.write('%s %s\n' % (sense_str, embed_str))
	logging.info('Written %s' % senseEmbed_path)


	# senseEmbed_path = 'data/vectors/senseEmbed.semcor.300.npz'
	# np.save(senseEmbed_path, matrix_A)
	# logging.info('Written %s' % senseEmbed_path)



	# # Save the model parameter in a txt file
	# w_path = 'data/vectors/weight.1024.300.txt'
	# np.savetxt(w_path, weight)
	# logging.info('Written %s' % w_path)


	# Save the model parameter in a npy file
	w1_path = 'data/vectors/weight.semcor.1024.300.npy'
	np.save(w1_path, weight)
	logging.info('Written %s' % w1_path)
	# print(np.load(w1_path).shape)





