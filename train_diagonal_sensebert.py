import tensorflow as tf
import sensebert
from sensebert import SenseBert
import numpy as np

import torch.optim as optim
import torch
import torch.nn as nn
import random

import logging
import argparse
import lxml.etree
import os
# from transformers import BertTokenizer, BertModel, BertForMaskedLM

# with tf.Session() as session:
#     sensebert_model = SenseBert("sensebert-base-uncased", session=session)  # or sensebert-large-uncased
#     input_ids, input_mask = sensebert_model.tokenize(["I went to the store to buy some groceries.", "The store was closed."])
#     model_outputs = sensebert_model.run(input_ids, input_mask)

# contextualized_embeddings, mlm_logits, supersense_logits = model_outputs  # these are NumPy arrays
# print('contextualized_embeddings', contextualized_embeddings)
# print('shape of contextualised embeddings:', contextualized_embeddings.shape)
# print('shape of contextualised embeddings:', contextualized_embeddings[0].shape)
# print('shape of contextualised embeddings[1][1]:', contextualized_embeddings[1][1].shape)
# print('length of contextualised embeddings:', len(contextualized_embeddings[1]))
# print(sensebert_model.tokenizer.convert_ids_to_senses([np.argmax(supersense_logits[1][3])]))
# print(sensebert_model.tokenizer.convert_ids_to_senses([np.argmax(supersense_logits[0][5])]))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def load_instances(train_path, keys_path):	
	"""Parse XML of split set and return list of instances (dict)."""
	train_instances = []
	sense_mapping = get_sense_mapping(keys_path)
	# tree = ET.parse(train_path)
	# for text in tree.getroot():
	text = read_xml_sents(train_path)
	for sent_idx, sentence in enumerate(text):
		inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': [], 'id': []}
		for e in sentence:
			inst['tokens_mw'].append(e.text)
			inst['lemmas'].append(e.get('lemma'))
			inst['id'].append(e.get('id'))
			inst['pos'].append(e.get('pos'))
			if 'id' in e.attrib.keys():
				inst['senses'].append(sense_mapping[e.get('id')])
			else:
				inst['senses'].append(None)

		inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

		"""handling multi-word expressions, mapping allows matching tokens with mw features"""
		idx_map_abs = []
		idx_map_rel = [(i, list(range(len(t.split()))))
						for i, t in enumerate(inst['tokens_mw'])]
		token_counter = 0
		"""converting relative token positions to absolute"""
		for idx_group, idx_tokens in idx_map_rel:  
			idx_tokens = [i+token_counter for i in idx_tokens]
			token_counter += len(idx_tokens)
			idx_map_abs.append([idx_group, idx_tokens])
		inst['tokenized_sentence'] = ' '.join(inst['tokens'])
		inst['idx_map_abs'] = idx_map_abs
		inst['idx'] = sent_idx
		train_instances.append(inst)

	return train_instances


def chunks(l, n):
	"""Yield successive n-sized chunks from given list."""
	for i in range(0, len(l), n):
		yield l[i:min(i + n, len(l))]


def get_sense_mapping(keys_path):
	sensekey_mapping = {}
	sense2id = {}
	with open(keys_path) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			sensekey_mapping[id_] = keys

	return sensekey_mapping


def read_xml_sents(xml_path):
	with open(xml_path) as f:
		for line in f:
			line = line.strip()
			if line.startswith('<sentence '):
				sent_elems = [line]
			elif line.startswith('<wf ') or line.startswith('<instance '):
				sent_elems.append(line)
			elif line.startswith('</sentence>'):
				sent_elems.append(line)
				yield lxml.etree.fromstring(''.join(sent_elems))


# def load_senseMatrices_npz(path):
# 	logging.info("Loading Pre-trained Sense Matrices ...")
# 	A = np.load(path, allow_pickle=True)	# A is loaded a 0d array
# 	A = np.atleast_1d(A.f.arr_0)			# convert it to a 1d array with 1 element
# 	A = A[0]								# a dictionary, key is sense id and value is sense matrix 
# 	logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(A))
# 	return A


# def load_weight(path):
# 	logging.info("Loading Model Parameters W ...")
# 	weight = np.load(path)
# 	weight = weight.f.arr_0
# 	logging.info('Loaded Model Parameters W')
# 	return weight


def write_to_file(path, mat):
	with open(path, 'w') as f:
		for sen_str, matrix in mat.items():
			matrix_str = ' '.join([str(v) for v in matrix])
			f.write('%s %s\n' % (sen_str, matrix_str))
	logging.info('Written %s' % path)


def save_pickle_dict(path, mat):
	pass


def get_args(
		num_epochs = 20,
		emb_dim = 300,
		batch_size = 64,
		diag = False,
		lr = 1e-4
			 ):

	parser = argparse.ArgumentParser(description='Word Sense Mapping')
	##parser.add_argument('--glove_embedding_path', default='external/glove/glove.500k.768.txt')
	##parser.add_argument('--glove_embedding_path', default='external/glove/glove.1024.txt')
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	# parser.add_argument('-load_sv_path', help='Path to sense vectors', default='data/vectors/senseMatrix.semcor_diagonal_relu_base_{}_50.npz'.format(emb_dim))
	# parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_diagonal_relu_768_{}_50.npz'.format(emb_dim))
	parser.add_argument('--num_epochs', default=num_epochs, type=int)
	parser.add_argument('--loss', default='standard', type=str, choices=['standard'])
	parser.add_argument('--emb_dim', default=emb_dim, type=int)
	parser.add_argument('--diagonalize', default=diag, type=bool)
	parser.add_argument('--device', default='cuda', type=str)
	parser.add_argument('--bert', default='large', type=str)
	parser.add_argument('--wsd_fw_path', help='Path to Semcor', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('--dataset', default='semcor', help='Name of dataset', required=False,
						choices=['semcor', 'semcor_omsti'])
	parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size', required=False)
	parser.add_argument('--lr', type=float, default=lr, help='Learning rate', required=False)
	parser.add_argument('--merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
						choices=['mean', 'first', 'sum'])
		
	args = parser.parse_args()

	return args


# Get embeddings from files
def load_glove_embeddings(fn):
	embeddings = {}
	with open(fn, 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			word = splitLine[0]
			vec = np.array(splitLine[1:], dtype='float32')
			vec = torch.from_numpy(vec)
			embeddings[word] = vec
	return embeddings

def get_sensebert_embeddings(sent):

	# with tf.Session() as session:
	# 	# sensebert_model = SenseBert("sensebert-large-uncased", session=session)    # or sensebert-large-uncased
	# 	sensebert_model = SenseBert(model, tokenizer, session=session)
	# 	input_ids, input_mask = sensebert_model.tokenize(sent)
	# 	# print('sensebert_model.tokenize(sent)', sensebert_model.tokenize(sent))
	# 	model_outputs = sensebert_model.run(input_ids, input_mask)

	# sensebert_model = SenseBert("sensebert-large-uncased", session=session)    # or sensebert-large-uncased

	with tf.device('/gpu:1'):
		sensebert_model = SenseBert(model, tokenizer, session=session)
		input_ids, input_mask = sensebert_model.tokenize(sent)
		# print('sensebert_model.tokenize(sent)', sensebert_model.tokenize(sent))
		model_outputs = sensebert_model.run(input_ids, input_mask)

	context_embeddings, mlm_logits, supersense_logits = model_outputs
	sensebert_embeddings = context_embeddings
	sensebert_embeddings = torch.from_numpy(sensebert_embeddings)

	return sensebert_embeddings




if __name__ == '__main__':

	args = get_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because Jodie doesn't have a GPU !!")
		args.device = 'cpu'

	if args.dataset == 'semcor':
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	elif args.dataset == 'semcor_omsti':
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'

	device = torch.device(args.device)

	sense2idx, sense_matrix = {}, {}
	idx, index = 0, 0

	with tf.device('/gpu:1'):
		session = tf.Session()
		model = sensebert._load_model("sensebert-large-uncased", session=session)
		tokenizer = sensebert.load_tokenizer("sensebert-large-uncased")

	# if args.bert == 'large':
	# 	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	# 	model = BertModel.from_pretrained('bert-large-cased')
	# 	save_sense_matrix_path = 'data/vectors/senseMatrix.semcor_diagonal_linear_sensebert_large_{}_1.npz'.format(args.emb_dim)
	# 	save_weight_path = 'data/vectors/weight.semcor_diagonal_linear_sensebert_1024_{}_1.npz'.format(args.emb_dim)
	# elif args.bert == 'base':
	# 	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	# 	model = BertModel.from_pretrained('bert-base-cased')
	# 	save_sense_matrix_path = 'data/vectors/senseMatrix.semcor_diagonal_linear_sensebert_large_{}_20.npz'.format(args.emb_dim)
	# 	save_weight_path = 'data/vectors/weight.semcor_diagonal_linear_sensebert_768_{}_20.npz'.format(args.emb_dim)


	# sensebert_model = SenseBert("sensebert-large-uncased")  # or sensebert-large-uncased
	# tokenizer = sensebert_model.tokenize()

	save_sense_matrix_path = 'data/vectors/senseMatrix.semcor_diagonal_relu_sensebert_large_{}_20.npz'.format(args.emb_dim)
	save_weight_path = 'data/vectors/weight.semcor_diagonal_relu_sensebert_1024_{}_20.npz'.format(args.emb_dim)

	# model.eval()
	
	logging.info("Loading Data........")
	instances = load_instances(train_path, keys_path)
	instances_len = len(instances)
	logging.info("Done. Loaded %d instances from dataset" % instances_len)

	'''
	split the training set for training and validation
	'''
	random.seed(52)
	random.shuffle(instances)
	index = int(instances_len*0.8)
	train_set = instances[:index]
	valid_set = instances[index:]
	# print('train_set', len(train_set))
	# print('val_set', len(val_set))

	# sentences = [e['tokenized_sentence'] for e in instances]
	# sensebert = get_sensebert_embeddings(sentences)

	# train_sents_embed = sensebert[:index]
	# valid_sents_embed = sensebert[index:]


	"""
	build sense2idx dictionary
	use dictionary to filter sense with the same id
	"""
	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(args.glove_embedding_path)
	logging.info("Done. Loaded %d words from GloVe embeddings" % len(glove_embeddings))

	for sent_instance in instances:
		for i in range(len(sent_instance['senses'])):
			if sent_instance['senses'][i] is None:
				continue

			# filter out of vocabulary words 
			for sense in sent_instance['senses'][i]:
				if sense in sense2idx:
					continue

				sense2idx[sense] = idx
				idx += 1

	
	num_senses = len(sense2idx)

	A = []
	for i in range(0, num_senses):
		A.append(torch.randn(args.emb_dim, dtype=torch.float32, device=device, requires_grad=False))

	W = [torch.randn(args.emb_dim, 1024, dtype=torch.float32, device=device, requires_grad=True)]
	
	optimizer = optim.Adam(W+A, lr=args.lr)
	# ##optimizer = optim.Adam(A, lr=args.lr)

	# ## Train from the saved model
	# A = []
	# loaded_A = load_senseMatrices_npz(args.load_sv_path)
	# for embed in loaded_A.values():
	# 	embed = torch.from_numpy(embed).to(device)
	# 	A.append(embed)

	# loaded_W = load_weight(args.load_weight_path)
	# loaded_W = torch.from_numpy(loaded_W).to(device)
	# loaded_W.requires_grad = True
	# W = [loaded_W]

	# optimizer = optim.Adam(W+A, lr=args.lr)
	# ##

	active = nn.ReLU(inplace=True)


	logging.info("------------------Training-------------------")

	for epoch in range(args.num_epochs):
		min_valid_loss = float('inf')
		cum_loss = 0  
		count = 0
		valid_count = 0
		valid_cum_loss = 0

		# if epoch > 0:
		# 	adjust_learning_rate(optimizer,epoch)

		for batch_idx, batch in enumerate(chunks(train_set, args.batch_size)):

			batch_sents = [e['tokenized_sentence'] for e in batch]
			batch_sensebert = get_sensebert_embeddings(batch_sents)

			# print('batch_sents', batch_sents)
			# print('shape of batch_sensebert', batch_sensebert.shape)

			"""
			set all of the A matrices to requires_grad = False
			optimizer.param_groups is a list which contains one dictionary
			optimizer.param_groups[0]['params'] returns a list of trainable parameters 
			""" 

			# for param_group in optimizer.param_groups[0]['params'][1:]:
			# 	param_group.requires_grad = False

			# print('batch', batch)
			for a in A:
				a.requires_grad = False

			optimizer.zero_grad()
			loss = torch.zeros(1, dtype=torch.float32).to(device)
			count += 1

			"""process contextual embeddings in sentences batches of size args.batch_size"""
			for sent_info, sent_sensebert in zip(batch, batch_sensebert): ## iterate through every sentence 
				idx_map_abs = sent_info['idx_map_abs']
				# print('shape of sent_sensebert', sent_sensebert.shape)
				# print('shape of sent_sensebert[1]', sent_sensebert[1].shape)

				# print("sent_info['tokenized_sentence']", sent_info['tokenized_sentence'])

				# sensebert_embeddings = get_sensebert_embeddings(sent_info['tokenized_sentence'])
				# print('sensebert_embeddings', len(sensebert_embeddings))
				# print('shape of sensebert_embeddings[1]', sensebert_embeddings.shape)
				
				for mw_idx, tok_idxs in idx_map_abs:
					if sent_info['senses'][mw_idx] is None:
						continue

					for sense in sent_info['senses'][mw_idx]:
						if sense not in sense2idx:
							continue

						index = sense2idx[sense]
						multi_words = []

						"""
						for the case of taking multiple words as a instance
						for example, obtaining the embedding for 'too much' instead of two embeddings for 'too' and 'much'
						we use mean to compute the averaged vec for a multiple words expression
						"""
						vec_c = torch.mean(torch.stack([sent_sensebert[i+1] for i in tok_idxs]), dim=0).to(device) ## tok_idxs starts from 0
						# print('shape of vec_c', vec_c.shape)
						vec_c = vec_c.reshape(1024, 1)

						for j in tok_idxs:
							token_word = sent_info['tokens'][j]
							
							if token_word in glove_embeddings.keys():
								multi_words.append(token_word)
								
						if len(multi_words) == 0:
							continue

						vec_g = torch.mean(torch.stack([glove_embeddings[w] for w in multi_words]), dim=0).to(device)

						context_vec = torch.mm(W[0], vec_c).squeeze(1)
						sense_vec = A[index] * vec_g
						sense_vec = active(sense_vec)
						loss += (context_vec - sense_vec).norm() ** 2
						#loss += (vec_c - A[index] * vec_g).norm() ** 2  ### no W

						# distance = context_vec - sense_vec
						# distance = active(distance)
						# loss += (distance).norm() ** 2
						
						
						"""set the A matrices that are in a current batch to require gradient"""
						# optimizer.param_groups[0]['params'][1+index].requires_grad = True
						A[index].requires_grad = True
						

			cum_loss += loss.item()
			loss.backward()

			optimizer.step()

		cum_loss /= count


		'''Validation'''
		for valid_batch_idx, valid_batch in enumerate(chunks(valid_set, args.batch_size)):

			valid_batch_sents = [e['tokenized_sentence'] for e in valid_batch]
			valid_batch_sensebert = get_sensebert_embeddings(valid_batch_sents)


			valid_loss = torch.zeros(1, dtype=torch.float32).to(device)
			valid_count += 1

			"""process contextual embeddings in sentences batches of size args.batch_size"""
			# for valid_sent_info in valid_batch:
			for valid_sent_info, valid_sent_sensebert in zip(valid_batch, valid_batch_sensebert): 
				valid_idx_map_abs = valid_sent_info['idx_map_abs']

				# valid_sent_bert = get_bert_embedding(valid_sent_info['tokenized_sentence'])

				for valid_mw_idx, valid_tok_idxs in valid_idx_map_abs:
					if valid_sent_info['senses'][valid_mw_idx] is None:
						continue

					for valid_sense in valid_sent_info['senses'][valid_mw_idx]:
						if valid_sense not in sense2idx:
							continue

						valid_index = sense2idx[valid_sense]
						valid_multi_words = []
						
						"""
						for the case of taking multiple words as a instance
						for example, obtaining the embedding for 'too much' instead of two embeddings for 'too' and 'much'
						we use mean to compute the averaged vec for a multiple words expression
						"""
						valid_vec_c = torch.mean(torch.stack([valid_sent_sensebert[i+1] for i in valid_tok_idxs]), dim=0).to(device)
						valid_vec_c = valid_vec_c.reshape(1024, 1)

						for k in valid_tok_idxs:
							valid_token_word = valid_sent_info['tokens'][k]
							
							if valid_token_word in glove_embeddings.keys():
								valid_multi_words.append(valid_token_word)

						if len(valid_multi_words) == 0:	
							continue
							
						valid_vec_g = torch.mean(torch.stack([glove_embeddings[w] for w in valid_multi_words]), dim=0).to(device)

						valid_context_vec = torch.mm(W[0], valid_vec_c).squeeze(1)
						valid_sense_vec = A[valid_index] * valid_vec_g

						valid_sense_vec = active(valid_sense_vec)
						valid_loss += (valid_context_vec - valid_sense_vec).norm() ** 2
						#loss += (vec_c - A[index] * vec_g).norm() ** 2  ### no W

						# valid_distance = valid_context_vec - valid_sense_vec
						# valid_distance = active(valid_distance)
						# valid_loss += (valid_distance).norm() ** 2
						
						

			valid_cum_loss += valid_loss.item()
		valid_cum_loss /= valid_count

		print("epoch: %d, train_loss: %f, valid_loss: %f" %(epoch, cum_loss, valid_cum_loss))

		if valid_cum_loss < min_valid_loss:
			min_valid_loss = valid_cum_loss
			best_epoch = epoch

			"""save the trained parameters W and A matrices"""
			weight = W[0].cpu().detach().numpy()
			matrix_A = [A[i].cpu().detach().numpy() for i in range(num_senses)]

			"""build the structures of W and A matrices"""
			for n in range(len(sense2idx)):
				sense_matrix[list(sense2idx.keys())[n]] = matrix_A[n]

			np.savez(save_sense_matrix_path, sense_matrix)
			np.savez(save_weight_path, weight)
			print('best epoch: %d, minimun validation loss: %f' %(best_epoch, min_valid_loss))

			early_stop_count = 0

		else:
			early_stop_count += 1
			if early_stop_count > 10:
				break

	print('shape of each matrix A:', matrix_A[0].shape)
	print('shape of weight matrix w:', weight.shape)
	print('Total number of senses: %d ' % len(sense_matrix))
	print('Written %s' % save_sense_matrix_path)	
	print('Written %s' % save_weight_path)



