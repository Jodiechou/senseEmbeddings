import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import matplotlib                                                                                                         
matplotlib.use('Agg')
import numpy as np
import math
# import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import gensim
import matplotlib as mpl
import logging
import lxml.etree
import tensorflow as tf
import sensebert
from sensebert import SenseBert
from scipy.spatial import distance
from transformers import BertTokenizer, BertModel, BertForMaskedLM


def lemmatize(w):
	lemma = w.split('%')[0]
	return lemma


def load_instances(train_path, keys_path):	
	"""Parse XML of split set and return list of instances (dict)."""
	instances = []
	sense_mapping = get_sense_mapping(keys_path)
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
		instances.append(inst)
	return instances


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


def get_bert_embedding(sent):
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
	res = list(zip(tokenized_text[1:-1], outputs[0][0][1:-1].cpu().detach().numpy())) ## [1:-1] is used to get rid of CLS] and [SEP]
	
	## merge subtokens
	sent_tokens_vecs = []
	for token in sent.split():
		token_vecs = []
		sub = []
		for subtoken in tokenizer.tokenize(token):
			encoded_token, encoded_vec = res.pop(0)
			sub.append(encoded_token)
			token_vecs.append(encoded_vec)
			merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0) 
			merged_vec = torch.from_numpy(merged_vec).to(device) 
		sent_tokens_vecs.append((token, merged_vec))

	return sent_tokens_vecs


"""Get embeddings from files"""
def load_glove_embeddings(fn):
	embeddings = {}
	with open(fn, 'r') as gfile:
		for line in gfile:
			splitLine = line.split(' ')
			word = splitLine[0]
			vec = np.array(splitLine[1:], dtype='float32')
			vec = torch.from_numpy(vec)  ### sense embeddings
			embeddings[word] = vec
	return embeddings


def gelu(x):
	""" Original Implementation of the gelu activation function in Google Bert repo when initialy created.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def load_npz(sv_path):
	logging.info("Loading Pre-trained Sense Matrices ...")
	A = np.load(sv_path, allow_pickle=True)	# A is loaded a 0d array
	A = np.atleast_1d(A.f.arr_0)			# convert it to a 1d array with 1 element
	A = A[0]								# a dictionary, key is sense id and value is sense matrix 
	logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(A))
	return A


def load_lmms(npz_vecs_path):
	lmms = {}
	loader = np.load(npz_vecs_path)
	labels = loader['labels'].tolist()
	vectors = loader['vectors']
	for label, vector in list(zip(labels, vectors)):
		lmms[label] = vector
	return lmms


def load_ares(ares_path):
		sense_vecs = {}
		with open(ares_path, 'r') as sfile:
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

def get_sensebert_embeddings(sent):
	sensebert_model = SenseBert(model, tokenizer, session=session)
	input_ids, input_mask = sensebert_model.tokenize(sent)
	# print('sensebert_model.tokenize(sent)', sensebert_model.tokenize(sent))
	model_outputs = sensebert_model.run(input_ids, input_mask)

	context_embeddings, mlm_logits, supersense_logits = model_outputs
	sensebert_embeddings = context_embeddings
	sensebert_embeddings = torch.from_numpy(sensebert_embeddings)

	return sensebert_embeddings


if __name__ == "__main__":


	sv_path = 'data/vectors/senseMatrix.semcor_diagonal_linear_large_bertlast4layers_multiword_300_50.npz'
	glove_embedding_path = 'external/glove/glove.840B.300d.txt'

	wsd_fw_set_path = 'example.xml'
	wsd_fw_gold_path = 'example.key.txt'
	logging.info('Formating data')
	instances = load_instances(wsd_fw_set_path, wsd_fw_gold_path)
	logging.info('Finish formating data')

	target_word = 'bank'
	target_sense = 'bank%1:17:01::'  #### bank%1:17:01::    bank%1:14:00::

	device = torch.device('cuda')

	idx2word = {}
	word2idx ={}
	similarities = []

	### BERT nn --------------
	# sent_bert_all = []
	# tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	# model = BertModel.from_pretrained('bert-large-cased')
	# model.eval()

	for sent_info in instances:
		print("sent_info['tokenized_sentence']", sent_info['tokenized_sentence'])

	# 	sent_bert_temp = get_bert_embedding(sent_info['tokenized_sentence'])
	# 	sent_bert_all.append(sent_bert_temp)
	# # print('sent_bert_all', sent_bert_all)

	# first_sent = sent_bert_all[0]
	# second_sent = sent_bert_all[1]
	# sent_bert = first_sent+second_sent
	# # print('second_sent', second_sent)

	# for idx, inst in enumerate(sent_bert):
	# 	idx2word[idx] = inst[0]
	# 	word2idx[inst[0]] = idx
	# # print("word2idx", word2idx)

	# for word, vec in first_sent:
	# 	if word == target_word:
	# 		target_vec = vec
	# target_idx = word2idx[target_word]
	# # print("target_idx", target_idx)

	# # print("sent_bert[0]", sent_bert[2])
	# # print('sent_bert[0][0]', sent_bert[2][0])
	# # print('sent_bert[0][1]', sent_bert[2][1])

	# for i in range(len(sent_bert)):
	# 	if i == target_idx:
	# 		continue

	# 	# print('sent_bert[i][0]', sent_bert[i][0])
	# 	embedding = sent_bert[i][1]
	# 	sim = torch.dot(target_vec, embedding) / (target_vec.norm() * embedding.norm())
	# 	token = idx2word[i]
	# 	similarities.append((token, sim))
	# # print("similarities", similarities)

	# sort_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
	# print('sort_sims', sort_sims)
	### ------------------


	### GloVe -------------------
	# logging.info("Loading Glove Embeddings........")
	# glove_embeddings = load_glove_embeddings(glove_embedding_path)
	# logging.info("Done. Loaded words from GloVe embeddings")

	# token_list_all = []
	# for sent_info in instances:
	# 	token_list_temp = sent_info['tokens']
	# 	token_list_all.append(token_list_temp)

	# first_sent = token_list_all[0]
	# second_sent = token_list_all[1]
	# token_list = first_sent+second_sent
	# # print('token_list', token_list)
	# # print('second_sent', second_sent)

	# for idx, word in enumerate(token_list):
	# 	word2idx[word] = idx

	# for word in first_sent:
	# 	if word == target_word:
	# 		target_vec = glove_embeddings[word]

	# target_idx = word2idx[target_word]
	# print("target_idx", target_idx)

	# for word in token_list:
	# 	if word not in glove_embeddings:
	# 		continue
	# 	embedding = glove_embeddings[word]
	# 	sim = 1- distance.cosine(target_vec, embedding)
	# 	similarities.append((word, sim))
	# # print("similarities", similarities)
	# sort_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
	# print('sort_sims', sort_sims)

	### --------------

	### sense embeddings----------
	sense2embeddings = []
	A = load_npz(sv_path)
	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")
	ares = load_ares('external/ares/ares_bert_large.txt')

	for sent_info in instances:
		idx_map_abs = sent_info['idx_map_abs']
		target_A_matrix = torch.from_numpy(A[target_sense]).to(device)
		target_g_vec = glove_embeddings[target_word].to(device)
		target_sense_vec = target_A_matrix * target_g_vec
		target_ares_vec = torch.from_numpy(ares[target_sense]).to(device)
		target_cdes_vec = torch.cat((target_ares_vec, target_sense_vec), 0)

		for mw_idx, tok_idxs in idx_map_abs:
			curr_sense = sent_info['senses'][mw_idx]

			if curr_sense is None:
				continue
			A_matrix = torch.from_numpy(A[curr_sense[0]]).to(device)
			multi_words = []

			for j in tok_idxs:
				token_word = sent_info['tokens'][j]			
				if token_word in glove_embeddings.keys():
					multi_words.append(token_word)

			if len(multi_words) == 0:
				currVec_g = torch.randn(300, dtype=torch.float32, device=device, requires_grad=False).to(device)
			else:
				currVec_g = torch.mean(torch.stack([glove_embeddings[w] for w in multi_words]), dim=0).to(device)

			sense_vec = A_matrix * currVec_g
			ares_vec = torch.from_numpy(ares[curr_sense[0]]).to(device)
			cdes_vec = torch.cat((ares_vec, sense_vec), 0)
			sense2embeddings.append((curr_sense, cdes_vec))

	for sense_temp, vec in sense2embeddings:
		sense = sense_temp[0]
		sim = torch.dot(target_cdes_vec, vec) / (target_cdes_vec.norm() * vec.norm())
		similarities.append((sense, sim))

	sort_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
	print('sort_sims', sort_sims)
	### -----------------------


	### LMMS and ARES-------------------
	# sense2embeddings = []
	# lmms = load_lmms('../bias-sense/data/lmms_2048.bert-large-cased.npz')
	# # ares = load_ares('external/ares/ares_bert_large.txt')

	# for sent_info in instances:
	# 	idx_map_abs = sent_info['idx_map_abs']
	# 	target_sense_vec = torch.from_numpy(lmms[target_sense]).to(device)

	# 	for mw_idx, tok_idxs in idx_map_abs:
	# 		curr_sense = sent_info['senses'][mw_idx]

	# 		if curr_sense is None:
	# 			continue

	# 		vector = torch.from_numpy(lmms[curr_sense[0]]).to(device)
	# 		sense2embeddings.append((curr_sense, vector))

	# for sense, vec in sense2embeddings:
	# 	sim = torch.dot(target_sense_vec, vec) / (target_sense_vec.norm() * vec.norm())
	# 	similarities.append((sense[0], sim))

	# sort_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
	# print('sort_sims', sort_sims)
	###-------------------------


	### sensebert-------------------
	# sent_sensebert_all = []
	# all_tokens = []

	# ## use sensebert ****
	# config = tf.ConfigProto() 
	# config.gpu_options.allow_growth = True 
	# session = tf.Session(config=config)
	# model = sensebert._load_model("sensebert-large-uncased", session=session)
	# tokenizer = sensebert.load_tokenizer("sensebert-large-uncased")

	# for sent_info in instances:
	# 	# idx_map_abs = sent_info['idx_map_abs']
	# 	sent_sensebert_temp = get_sensebert_embeddings(sent_info['tokenized_sentence'])
	# 	# sent_sensebert_temp = sent_sensebert_temp.tolist()
	# 	sent_sensebert_all.append(sent_sensebert_temp)
	# 	tokens = sent_info['tokens']
	# 	for tok in tokens:
	# 		all_tokens.append(tok) 
	
	# first_sent = sent_sensebert_all[0][0]
	# second_sent = sent_sensebert_all[1][0]
	
	# print(second_sent[0])

	# sent_sensebert = torch.cat((first_sent, second_sent), dim=0)
	# sensebert_final = list(zip(all_tokens, sent_sensebert))

	# for idx, inst in enumerate(sensebert_final):
	# 	idx2word[idx] = inst[0]
	# 	word2idx[inst[0]] = idx

	# first_sent_final = sensebert_final[0:58]
	# second_sent_final = sensebert_final[58:]
	# print('first_sent_final', first_sent_final)
	# print('second_sent_final', second_sent_final)
	# print('-----------len s1', len(first_sent_final))
	# print('-----------len s2', len(second_sent_final))

	# for word, vec in first_sent_final:
	# 	if word == target_word:
	# 		target_vec = vec
	# target_idx = word2idx[target_word]
	
	# for i in range(len(sensebert_final)):
	# 	if i == target_idx:
	# 		continue

	# 	embedding = sensebert_final[i][1]
	# 	sim = torch.dot(target_vec, embedding) / (target_vec.norm() * embedding.norm())
	# 	token = idx2word[i]
	# 	similarities.append((token, sim))
	
	# sort_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
	# print('sort_sims', sort_sims)
	###-------------------------
