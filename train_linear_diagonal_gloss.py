import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np 
import torch.optim as optim
import torch
import random

import logging
import argparse
import lxml.etree
from transformers import BertTokenizer, BertModel, BertForMaskedLM




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


def write_to_file(path, mat):
	with open(path, 'w') as f:
		for sen_str, matrix in mat.items():
			matrix_str = ' '.join([str(v) for v in matrix])
			f.write('%s %s\n' % (sen_str, matrix_str))
	logging.info('Written %s' % path)


def save_pickle_dict(path, mat):
	pass


def load_gloss_embeddings(path):
	# logging.info("Loading Pre-trained Sense Matrices ...")
	loader = np.load(path, allow_pickle=True)    # gloss_embeddings is loaded a 0d array
	loader = np.atleast_1d(loader.f.arr_0)       # convert it to a 1d array with 1 element
	gloss_embeddings = loader[0]				 # a dictionary, key is sense id and value is embeddings
	logging.info("Loaded %d gloss embeddings" % len(gloss_embeddings))
	return gloss_embeddings


def get_args(
		num_epochs = 50,
		emb_dim = 300,
		batch_size = 64,
		diag = False,
		lr = 1e-4
			 ):

	parser = argparse.ArgumentParser(description='Word Sense Mapping')
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	parser.add_argument('--gloss_embedding_path', default='data/vectors/gloss_embeddings.npz')
	# parser.add_argument('-load_sv_path', help='Path to sense vectors', default='data/old/senseMatrix.semcor_diagonal_valid_large_{}_50.npz'.format(emb_dim))
	# parser.add_argument('--load_weight_path', default='data/old/weight.semcor_diagonal_valid_1024_{}_50.npz'.format(emb_dim))
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


def get_bert_embedding(sent):
	"""
	input: a sentence
	output: word embeddigns for the words apprearing in the sentence
	"""
	tokenized_text = tokenizer.tokenize("[CLS] {0} [SEP]".format(sent))
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [0 for i in range(len(indexed_tokens))]
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])
	tokens_tensor = tokens_tensor.to(device)
	segments_tensors = segments_tensors.to(device)
	model.to(device)
	with torch.no_grad():
		outputs = model(tokens_tensor, token_type_ids=segments_tensors)
	"""[1:-1] is used to get rid of CLS] and [SEP]"""
	res = list(zip(tokenized_text[1:-1], outputs[0].cpu().detach().numpy()[0][1:-1])) 
	
	"""merge subtokens"""
	sent_tokens_vecs = []
	for token in sent.split():
		token_vecs = []
		sub = []
		for subtoken in tokenizer.tokenize(token):
			encoded_token, encoded_vec = res.pop(0)
			sub.append(encoded_token)
			token_vecs.append(encoded_vec)
			merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0) 
			merged_vec = torch.from_numpy(merged_vec)
			# if args.bert == 'large':
			# 	merged_vec = torch.from_numpy(merged_vec.reshape(1024, 1))
			# elif args.bert == 'base':
			# 	merged_vec = torch.from_numpy(merged_vec.reshape(768, 1))
		sent_tokens_vecs.append((token, merged_vec))
	return sent_tokens_vecs


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

	if args.bert == 'large':
		tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
		model = BertModel.from_pretrained('bert-large-cased')
		save_sense_matrix_path = 'data/vectors/senseMatrix.semcor_gloss_diagonal_linear_large_{}_50.npz'.format(args.emb_dim)
		save_weight_path = 'data/vectors/weight.semcor_gloss_diagonal_linear_2048_{}_50.npz'.format(args.emb_dim)
	elif args.bert == 'base':
		tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
		model = BertModel.from_pretrained('bert-base-cased')
		save_sense_matrix_path = 'data/vectors/senseMatrix.semcor_diagonal_valid_base_{}_20.npz'.format(args.emb_dim)
		save_weight_path = 'data/vectors/weight.semcor_diagonal_valid_base_768_{}_20.npz'.format(args.emb_dim)

	model.eval()
	
	logging.info("Loading Data........")
	instances = load_instances(train_path, keys_path)
	instances_len = len(instances)
	logging.info("Done. Loaded %d instances from dataset" % instances_len)

	'''
	split the training set for training and validation
	'''
	random.seed(65)
	random.shuffle(instances)
	index = int(instances_len*0.8)
	train_set = instances[:index]
	valid_set = instances[index:]


	"""
	build sense2idx dictionary
	use dictionary to filter sense with the same id
	"""
	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(args.glove_embedding_path)
	logging.info("Done. Loaded %d words from GloVe embeddings" % len(glove_embeddings))

	gloss_vecs = load_gloss_embeddings(args.gloss_embedding_path)
	# print('gloss_vecs', gloss_vecs)
	# print('len of gloss_vecs', len(gloss_vecs))

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

	if args.bert == 'large':
		W = [torch.randn(args.emb_dim, 2048, dtype=torch.float32, device=device, requires_grad=True)]
	elif args.bert == 'base':
		W = [torch.randn(args.emb_dim, 768, dtype=torch.float32, device=device, requires_grad=True)]


	optimizer = optim.Adam(W+A, lr=args.lr)
	##optimizer = optim.Adam(A, lr=args.lr)


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

	

	logging.info("------------------Training-------------------")

	for epoch in range(args.num_epochs):
		min_valid_loss = float('inf')
		cum_loss = 0  
		count = 0
		valid_count = 0
		valid_cum_loss = 0

		for batch_idx, batch in enumerate(chunks(train_set, args.batch_size)):

			"""
			set all of the A matrices to requires_grad = False
			optimizer.param_groups is a list which contains one dictionary
			optimizer.param_groups[0]['params'] returns a list of trainable parameters 
			""" 

			# for param_group in optimizer.param_groups[0]['params'][1:]:
			# 	param_group.requires_grad = False
			for a in A:
				a.requires_grad = False

			optimizer.zero_grad()
			loss = torch.zeros(1, dtype=torch.float32).to(device)
			count += 1

			"""process contextual embeddings in sentences batches of size args.batch_size"""
			for sent_info in batch:
				idx_map_abs = sent_info['idx_map_abs']

				sent_bert = get_bert_embedding(sent_info['tokenized_sentence'])

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
						vec_c = torch.mean(torch.stack([sent_bert[i][1] for i in tok_idxs]), dim=0).to(device)
						gloss_vec = torch.from_numpy(gloss_vecs[sense]).to(device)
						cont_vec_c = torch.cat((vec_c, gloss_vec), 0)
						cont_vec_c = cont_vec_c.reshape(2048, 1)
						# print('shape of cont_vec_c', cont_vec_c.shape)

						for j in tok_idxs:
							token_word = sent_info['tokens'][j]
							
							if token_word in glove_embeddings.keys():
								multi_words.append(token_word)

						if len(multi_words) == 0:
							continue

						vec_g = torch.mean(torch.stack([glove_embeddings[w] for w in multi_words]), dim=0).to(device)
						
						loss += ((torch.mm(W[0], cont_vec_c).squeeze(1)) - A[index] * vec_g).norm() ** 2
						##loss += (vec_c - A[index] * vec_g).norm() ** 2  ### no W
						
						"""set the A matrices that are in a current batch to require gradient"""
						# optimizer.param_groups[0]['params'][1+index].requires_grad = True
						A[index].requires_grad = True
						

			cum_loss += loss.item()
			loss.backward()

			optimizer.step()

		cum_loss /= count


		'''Validation'''
		for valid_batch_idx, valid_batch in enumerate(chunks(valid_set, args.batch_size)):

			valid_loss = torch.zeros(1, dtype=torch.float32).to(device)
			valid_count += 1

			"""process contextual embeddings in sentences batches of size args.batch_size"""
			for valid_sent_info in valid_batch:
				valid_idx_map_abs = valid_sent_info['idx_map_abs']

				valid_sent_bert = get_bert_embedding(valid_sent_info['tokenized_sentence'])

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
						valid_vec_c = torch.mean(torch.stack([valid_sent_bert[i][1] for i in valid_tok_idxs]), dim=0).to(device)
						valid_gloss_vec = torch.from_numpy(gloss_vecs[sense]).to(device)
						valid_cont_vec_c = torch.cat((valid_vec_c, valid_gloss_vec), 0)
						valid_cont_vec_c = valid_cont_vec_c.reshape(2048, 1)

						for k in valid_tok_idxs:
							valid_token_word = valid_sent_info['tokens'][k]
							
							if valid_token_word in glove_embeddings.keys():
								valid_multi_words.append(valid_token_word)

						if len(valid_multi_words) == 0:	
							continue
							
						valid_vec_g = torch.mean(torch.stack([glove_embeddings[w] for w in valid_multi_words]), dim=0).to(device)

						valid_loss += ((torch.mm(W[0], valid_cont_vec_c).squeeze(1)) - A[valid_index] * valid_vec_g).norm() ** 2
						##loss += (vec_c - A[index] * vec_g).norm() ** 2  ### no W
						
			valid_cum_loss += valid_loss.item()
		valid_cum_loss /= valid_count

		logging.info("epoch: %d, train_loss: %f, valid_loss: %f" %(epoch, cum_loss, valid_cum_loss))

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
			logging.info('best epoch: %d, minimun validation loss: %f' %(best_epoch, min_valid_loss))

			early_stop_count = 0

		else:
			early_stop_count += 1
			if early_stop_count > 10:
				break

	print('shape of each matrix A:', matrix_A[0].shape)
	print('shape of weight matrix w:', weight.shape)
	logging.info('Total number of senses: %d ' % len(sense_matrix))
	logging.info('Written %s' % save_sense_matrix_path)	
	logging.info('Written %s' % save_weight_path)

