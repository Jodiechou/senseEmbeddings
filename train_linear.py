import numpy as np 
import torch.optim as optim
import torch
from torch.autograd import Variable

import logging
import argparse

import lxml.etree
import xml.etree.ElementTree as ET

import os

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM



# torch.cuda.set_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def load_training_set(train_path, keys_path):	
	"""Parse XML of split set and return list of instances (dict)."""
	train_instances = []
	sense_mapping = get_sense_mapping(keys_path)
	tree = ET.parse(train_path)
	for text in tree.getroot():
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

			# handling multi-word expressions, mapping allows matching tokens with mw features
			idx_map_abs = []
			idx_map_rel = [(i, list(range(len(t.split()))))
							for i, t in enumerate(inst['tokens_mw'])]
			token_counter = 0
			for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
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


def get_args(
		num_epochs = 5,
		emb_dim = 300,
		diag = False
			 ):

	parser = argparse.ArgumentParser(description='BERT Word Sense Embeddings')
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	parser.add_argument('--sense_matrix_path', type=str, default='data/vectors/senseMatrix.semcor_{}_{}.txt'.format(emb_dim, emb_dim))
	parser.add_argument('--save_sense_emb_path', default='data/vectors/senseEmbed.semcor_{}.txt'.format(emb_dim))
	parser.add_argument('--save_sense_matrix_path', default='data/vectors/senseEmbed.semcor_{}.npz'.format(emb_dim))
	parser.add_argument('--save_weight_path', default='data/vectors/weight.semcor_1024_{}.npz'.format(emb_dim))
	parser.add_argument('--num_epochs', default=num_epochs, type=int)
	parser.add_argument('--loss', default='standard', type=str, choices=['standard'])
	parser.add_argument('--emb_dim', default=emb_dim, type=int)
	parser.add_argument('--diagonalize', default=diag, type=bool)
	parser.add_argument('--device', default='cpu', type=str)
	parser.add_argument('--wsd_fw_path', help='Path to Semcor', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('--dataset', default='semcor', help='Name of dataset', required=False,
						choices=['semcor', 'semcor_omsti'])
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
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
			embeddings[word] = vec
	return embeddings


def get_bert_embedding(sent):
	tokenized_text = tokenizer.tokenize("[CLS] {0} [SEP]".format(sent))
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [0 for i in range(len(indexed_tokens))]
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])
	tokens_tensor = tokens_tensor.to(cuda0)
	segments_tensors = segments_tensors.to(cuda0)
	model.to(cuda0)
	with torch.no_grad():
		outputs = model(tokens_tensor, token_type_ids=segments_tensors)
	res = list(zip(tokenized_text[1:-1], outputs[0].cpu().detach().numpy()[0][1:-1])) ## [1:-1] is used to get rid of CLS] and [SEP]
	
	## merge subtokens
	sent_tokens_vecs = []
	for token in sent.split():
		token_vecs = []
		sub = []
		for subtoken in tokenizer.tokenize(token):
			encoded_token, encoded_vec = res.pop(0)
			sub.append(encoded_token)
			token_vecs.append(encoded_vec)
			merged_vec = np.array(token_vecs).mean(axis=0) ###
			merged_vec = torch.from_numpy(merged_vec.reshape(1, 1024)).to(cuda1)

		sent_tokens_vecs.append((token, merged_vec))
		# del merged_vec
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
	cuda0 = torch.device("cuda:0")
	cuda1 = torch.device("cuda:1")

	sense2idx, sense2matrix, sense_matrix = {}, {}, {}
	idx, index, out_of_vocab_num = 0, 0, 0

	lr = 1e-3

	count_loss = 0

	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased')
	model.eval()
	
	logging.info("Loading Training Data........")
	train_instances = load_training_set(train_path, keys_path)
	logging.info("Done. Loaded %d instances from dataset" % len(train_instances))
	
	## Build sense2idx dictionary
	## Use dictionary to filter sense with the same id
	logging.info("Loading Glove Embeddings........")
	glove_embeddings_full = load_glove_embeddings(args.glove_embedding_path)

	glove_embeddings = {}
	for sent_instance in train_instances:
		for i in range(len(sent_instance['senses'])):
			if sent_instance['senses'][i] is None:
				continue

			# filter out of vocabulary words 
			for sense in sent_instance['senses'][i]:
				if sense in sense2idx:
					continue

				word = sense.split('%')[0]
				if word not in glove_embeddings_full.keys():
					out_of_vocab_num += 1
					continue

				glove_embeddings[word] = torch.from_numpy(glove_embeddings_full[word].reshape(1, 300)).to(cuda1)

				sense2idx[sense] = idx
				idx += 1

	logging.info("Done. Loaded %d words from GloVe embeddings" % len(glove_embeddings))
	
	num_senses = len(sense2idx)


	
	A = torch.randn(num_senses, args.emb_dim, args.emb_dim, requires_grad=True, dtype=torch.float32, device=cuda1)
	# A = torch.randn(num_senses, args.emb_dim, args.emb_dim, requires_grad=True, dtype=torch.float32)
	# A = torch.randn(num_senses, args.emb_dim, dtype=torch.float32, device=cuda1)
	# A = Variable(torch.diag_embed(A), requires_grad=True)
	# W = torch.randn(args.emb_dim, 1024, requires_grad=True, dtype=torch.float32)
	W = torch.randn(1024, args.emb_dim, requires_grad=True, dtype=torch.float32, device=cuda1)
	optimizer = optim.Adam((A, W), lr)
	

	logging.info("------------------Training-------------------")

	for epoch in range(args.num_epochs):
		for batch_idx, batch in enumerate(chunks(train_instances, args.batch_size)):
		
			loss = 0
			batch_bert = []

			# process contextual embeddings in sentences batches of size args.batch_size
			# batch_bert = bert_embed(batch_sents, merge_strategy=args.merge_strategy)
			print('Im here 1')

			for sent_info in batch:
				idx_map_abs = sent_info['idx_map_abs']

				sent_bert = get_bert_embedding(sent_info['tokenized_sentence'])

				print('Im here 2')

				for mw_idx, tok_idxs in idx_map_abs:
					if sent_info['senses'][mw_idx] is None:
						continue

					for sense in sent_info['senses'][mw_idx]:
						if sense not in sense2idx:
							continue

						index = sense2idx[sense]
						word = sense.split('%')[0]
						print('Im here 3')

						vec_g = glove_embeddings[word]
						print('Im here 4')

						# For the case of taking multiple words as a instance
						# for example, obtaining the embedding for 'too much' instead of two embeddings for 'too' and 'much'
						# we use mean to compute the averaged vec for a multiple words expression
						# vec_c = np.array([sent_bert[i][1] for i in tok_idxs], dtype=np.float32).mean(axis=0)

						vec_c = torch.mean(torch.stack([sent_bert[i][1] for i in tok_idxs]), dim=0)		
						print('Im here 5')				
					
						loss += (torch.mm(vec_c, W) - torch.mm(vec_g, A[index])).norm() ** 2
						print('loss', loss.item())
						print('Im here 6')

						del vec_c, vec_g
						print('Im here 7')
			

		
			optimizer.zero_grad()
			print('Im here 8') ### the code is able to print out numer 8 
			loss.backward() 
			print('Im here 9')
			optimizer.step()
			print('Im here 10')


		logging.info("epoch: %d, loss: %f " %(epoch, loss.item()))

	weight = W.cpu().detach().numpy()
	matrix_A = A.cpu().detach().numpy()



	logging.info('number of out of vocab word: %d' %(out_of_vocab_num))
	print('shape of each matrix A', matrix_A[0].shape)
	print('shape of weight matrix w', weight.shape)	

	# Build the structures of W and A matrices
	for n in range(len(sense2idx)):
		sense_matrix[list(sense2idx.keys())[n]] = matrix_A[n]

	logging.info("Total number of senses %d " %(len(sense_matrix)))

	write_to_file(args.sense_matrix_path, sense_matrix)

	np.savez(args.save_sense_matrix_path, sense_matrix)
	logging.info('Written %s' % args.save_sense_matrix_path)

	np.savez(args.save_weight_path, weight)
	logging.info('Written %s' % args.save_weight_path)

