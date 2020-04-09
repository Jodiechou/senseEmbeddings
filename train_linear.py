import numpy as np 
from random import shuffle
from torch.autograd import Variable
import torch.optim as optim
import torch
import logging
import argparse

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def shuffler(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


def write_to_file(path, mat):
	with open(path, 'w') as f:
		for sen_str, matrix in mat.items():
			matrix_str = ' '.join([str(v) for v in matrix])
			f.write('%s %s\n' % (sen_str, matrix_str))
	logging.info('Written %s' % path)


def save_pickle_dict(path, mat):
	pass


def get_args(
		num_epochs = 200,
		emb_dim = 300,
		diag = False
			 ):

	parser = argparse.ArgumentParser(description='BERT Word Sense Embeddings')
	parser.add_argument('--embedding_path', default='data/vectors/glove_embeddings.semcor_{}.txt'.format(emb_dim), type=str)
	parser.add_argument('--sense_embedding_path', default='data/vectors/bert_embeddings.semcor_1024.txt', type=str)
	parser.add_argument('--sense_matrix_path', type=str, default='data/vectors/senseMatrix.semcor_{}_{}.txt'.format(emb_dim, emb_dim))
	parser.add_argument('--save_sense_emb_path', default='data/vectors/senseEmbed.semcor_{}.txt'.format(emb_dim))
	parser.add_argument('--save_sense_matrix_path', default='data/vectors/senseEmbed.semcor_{}.npz'.format(emb_dim))
	parser.add_argument('--save_weight_path', default='data/vectors/weight.semcor_1024_{}.npy'.format(emb_dim))
	parser.add_argument('--num_epochs', default=num_epochs, type=int)
	parser.add_argument('--bsize', default=32, type=int)
	parser.add_argument('--loss', default='standard', type=str, choices=['standard'])
	parser.add_argument('--emb_dim', default=emb_dim, type=int)
	parser.add_argument('--diagonalize', default=diag, type=bool)
	parser.add_argument('--device', default='cuda', type=str)
	args = parser.parse_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because Jodie doesn't have a GPU !!")
		args.device = 'cpu'

	return args


# Get embeddings from files
def load_glove_embeddings(fn):
	embed_g = []
	senses = []
	symbols = [",", "+", "'", ':', '%']

	with open(fn, 'r') as gfile:
		for line in gfile:

			splitLine = line.split(' ')
			sense_word = ''
			end_idx = 0

			for chars in splitLine:
				if any(c.isalpha() for c in chars) or chars in symbols or len(chars) < 4:
					if end_idx > 0:
						sense_word += " " + chars
					else:
						sense_word += chars
					end_idx += 1
				else:
					break

			try:
				assert len(splitLine[end_idx:]) ==  300
				sense_vec = [float(i) for i in splitLine[end_idx:]]
			except:
				sense_word = splitLine[0].split('%')[0]
				sense_vec = [float(i) for i in splitLine[1:]]

			senses.append(sense_word)
			embed_g.append(sense_vec)
	return embed_g, senses


def load_bert_embeddings(fn):
	embed_c = []
	symbols = [",", "+", "'", ':', '%']

	with open(fn, 'r') as cfile:
		for line in cfile:
			splitLine = line.split(' ')
			sense_word = ''
			end_idx = 0
			for chars in splitLine:
				if any(c.isalpha() for c in chars) or chars in symbols or len(chars) < 4:
					if end_idx > 0:
						sense_word += " " + chars
					else:
						sense_word += chars
					end_idx += 1
				else:
					break

			bert_vec = [float(i) for i in  splitLine[end_idx:]]
			embed_c.append(bert_vec)
	return embed_c


def lossFunctions(z):
	loss = torch.dot(z, z)
	return loss


def forward(input1, input2, w, matrix_A_temp):
	z = torch.matmul(input1, w) - torch.matmul(input2, matrix_A_temp)
	z = z.view(-1)
	return z


if __name__ == '__main__':

	args = get_args()
	device = torch.device(args.device)

	mat_A = {}
	senEmbeds = {}

	lr = 1e-4

	logging.info("Loading Glove Embeddings........")
	senses = load_glove_embeddings(args.embedding_path)[1]
	embed_g = load_glove_embeddings(args.embedding_path)[0]
	
	logging.info("Done. Loaded %d words from GloVe embeddings" % len(embed_g))
	logging.info("Loading BERT Embeddings........")
	embed_c = load_bert_embeddings(args.sense_embedding_path)
	logging.info("Done. Loaded %d words from BERT embeddings" % len(embed_c))

	embed_g = np.array(embed_g, dtype='float32')
	embed_c = np.array(embed_c, dtype='float32')
	embed_g_shuf, embed_c_shuf = shuffler(embed_g, embed_c)
	print("Shape of embed_g: ", embed_g.shape)
	print("Shape of embed_c: ", embed_c.shape)

	matrix_A = np.zeros((len(senses), args.emb_dim, args.emb_dim), dtype='float32')
	embeds = np.zeros((len(senses), args.emb_dim), dtype='float32')


	g_train = torch.from_numpy(embed_g_shuf)
	c_train = torch.from_numpy(embed_c_shuf)

	w = torch.randn(1024, args.emb_dim, requires_grad=True,  device='cuda')
	matrix_A_temp = torch.randn(args.emb_dim, args.emb_dim, requires_grad=True, device='cuda')
	optimizer = optim.Adam((matrix_A_temp, w), lr)

	num_senses = len(senses)
	num_batches = num_senses // args.bsize

	logging.info("------------------Training-------------------")
	for epoch in range(args.num_epochs):
		loss_sum = 0
		for batch_num, i in enumerate(range(args.bsize, num_senses, args.bsize)):

			if batch_num == num_batches: 
				i = num_senses


			contEmbed = c_train[i-args.bsize:i].view(-1, 1024).to(device)
			g_train_vec = g_train[i-args.bsize:i].to(device)
			wordEmbed = g_train_vec.view(-1, args.emb_dim)

			z = forward(contEmbed, wordEmbed, w, matrix_A_temp)
			optimizer.zero_grad()
			loss = lossFunctions(z)
			loss_sum += loss.item()
			loss.backward()
			optimizer.step()

			embed = torch.matmul(wordEmbed, matrix_A_temp)
			embeds[i-args.bsize:i] = embed.cpu().detach().numpy()
			matrix_A[i-args.bsize:i] = matrix_A_temp.cpu().detach().numpy()

		average_loss = 	loss_sum / num_batches
		print("epoch: %d, loss: %f " %(epoch, loss_sum))

	embeds = np.array(embeds, dtype='float32')
	matrix_A = np.array(matrix_A, dtype='float32')
	print('matrix_a shape', matrix_A.shape)
	weight = w.cpu().detach().numpy()


	
	print('shape of sense embedding', embeds.shape)
	print('shape of each matrix A', matrix_A[0].shape)
	print('shape of weight matrix w', weight.shape)		

	# Build the structure of sense embeddings and A-matrix
	for n in range(len(senses)):
		mat_A[senses[n]] = matrix_A[n]
		senEmbeds[senses[n]] = embeds[n]

	logging.info("Total number of senses %d " %(len(senEmbeds)))

	write_to_file(args.sense_matrix_path, mat_A)
	write_to_file(args.save_sense_emb_path, senEmbeds)
	

	# np.save(args.save_sense_matrix_path.replace('txt', 'npz'), matrix_A)
	np.savez(args.save_sense_matrix_path, matrix_A)
	logging.info('Written %s' % args.save_sense_matrix_path)

	# Save the model parameter in a npy file
	np.save(args.save_weight_path, weight) # or txt file np.savetxt(w_path, weight)
	logging.info('Written %s' % args.save_weight_path)
	# print(np.load(w1_path).shape)





