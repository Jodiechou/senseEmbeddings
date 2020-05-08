import os
import logging
import argparse
from time import time
from datetime import datetime
from collections import defaultdict
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import torch

from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import wordnet as wn
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def get_args(
		emb_dim = 300,
		batch_size = 64,
		diag = False
			 ):

	parser = argparse.ArgumentParser(description='WSD Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	parser.add_argument('-sv_path', help='Path to sense matrices', required=True)
	parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_1024_{}.npz'.format(emb_dim))
	parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('-test_set', default='ALL', help='Name of test set', required=False,
						choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'])
	parser.add_argument('-batch_size', type=int, default=batch_size, help='Batch size', required=False)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
	# parser.add_argument('-ignore_lemma', dest='use_lemma', action='store_false', help='Ignore lemma features', required=False)
	parser.add_argument('-ignore_pos', dest='use_pos', action='store_false', help='Ignore POS features', required=False)
	parser.add_argument('-thresh', type=float, default=-1, help='Similarity threshold', required=False)
	parser.add_argument('-k', type=int, default=1, help='Number of Neighbors to accept', required=False)
	parser.add_argument('-quiet', dest='debug', action='store_false', help='Less verbose (debug=False)', required=False)
	parser.add_argument('--device', default='cuda', type=str)
	# parser.set_defaults(use_lemma=True)
	parser.set_defaults(use_pos=True)
	parser.set_defaults(debug=True)
	args = parser.parse_args()

	return args


def load_wsd_fw_set(wsd_fw_set_path):
	"""Parse XML of split set and return list of instances (dict)."""
	eval_instances = []
	tree = ET.parse(wsd_fw_set_path)
	for text in tree.getroot():
		for sent_idx, sentence in enumerate(text):
			inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
			for e in sentence:
				inst['tokens_mw'].append(e.text)
				inst['lemmas'].append(e.get('lemma'))
				inst['senses'].append(e.get('id'))
				inst['pos'].append(e.get('pos'))

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

			eval_instances.append(inst)

	return eval_instances


def get_id2sks(wsd_eval_keys):
	"""Maps ids of split set to sensekeys, just for in-code evaluation."""
	id2sks = {}
	with open(wsd_eval_keys) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			id2sks[id_] = keys
	return id2sks


def run_scorer(wsd_fw_path, test_set, results_path):
	"""Runs the official java-based scorer of the WSD Evaluation Framework."""
	cmd = 'cd %s && java Scorer %s %s' % (wsd_fw_path + 'Evaluation_Datasets/',
										  '%s/%s.gold.key.txt' % (test_set, test_set),
										  '../../../../' + results_path)
	print(cmd)
	os.system(cmd)


def chunks(l, n):
	"""Yield successive n-sized chunks from given list."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def str_scores(scores, n=3, r=5):  ###
	"""Convert scores list to a more readable string."""
	return str([(l, round(s, r)) for l, s in scores[:n]])


def load_senseMatrices_npz(path):
	logging.info("Loading Pre-trained Sense Matrices ...")
	A = np.load(path, allow_pickle=True)	# A is loaded a 0d array
	A = np.atleast_1d(A.f.arr_0)			# convert it to a 1d array with 1 element
	A = A[0]								# a dictionary, key is sense id and value is sense matrix 
	logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(A))
	return A


def load_senseMatrices_txt(path):
	logging.info("Loading Pre-trained Sense Matrices ...")
	A = {}
	with open(path, 'r') as sfile:
		for line in sfile:
			splitLine = line.split(' ')
			sense = splitLine[0]
			matrix = np.array(splitLine[1:])
			A[sense] = matrix
	logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(A))
	return A


def load_weight(path):
	logging.info("Loading Model Parameters W ...")
	weight = np.load(path)
	weight = weight.f.arr_0
	logging.info('Loaded Model Parameters W')
	return weight


def get_sk_pos(sk, tagtype='long'):
	# merges ADJ with ADJ_SAT

	if tagtype == 'long':
		type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
		return type2pos[get_sk_type(sk)]

	elif tagtype == 'short':
		type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
		return type2pos[get_sk_type(sk)]

def get_sk_type(sensekey):
	return int(sensekey.split('%')[1].split(':')[0])


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
			merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0) 
			merged_vec = torch.from_numpy(merged_vec.reshape(1024, 1)).to(device)
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
			embeddings[word] = vec
	return embeddings


if __name__ == '__main__':

	args = get_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because Jodie doesn't have a GPU !!")
		args.device = 'cpu'

	device = torch.device(args.device)

	"""
	Load pre-trianed sense embeddings for evaluation.
	Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
	Load fastText static embeddings if required.
	"""
	sensekeys = []
	lemmas = []
	vectors = []

	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased')
	model.eval()

	"""
	load pre-trained parameters
	A is a dictionary, key is sense id and value is sense matrix
	"""
	A = load_senseMatrices_npz(args.sv_path)
	W = load_weight(args.load_weight_path)
	W = torch.from_numpy(W).to(device)
	

	senseKeys = list(A.keys())
	matrices = list(A.values())
	lemmas = [elem.split('%')[0] for elem in senseKeys]
	
	"""
	Initialize various counters for calculating supplementary metrics.
	"""
	n_instances, n_correct, n_unk_lemmas, acc_sum = 0, 0, 0, 0
	failed_by_pos = defaultdict(list)

	pos_confusion = {}
	for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
		pos_confusion[pos] = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}

	"""
	Load evaluation instances and gold labels.
	Gold labels (sensekeys) only used for reporting accuracy during evaluation.
	"""
	wsd_fw_set_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (args.test_set, args.test_set)
	wsd_fw_gold_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (args.test_set, args.test_set)
	id2senses = get_id2sks(wsd_fw_gold_path)
	logging.info('Formating testing data')
	eval_instances = load_wsd_fw_set(wsd_fw_set_path)
	logging.info('Finish formating testing data')

	"""
	build sense2idx dictionary
	use dictionary to filter sense with the same id
	"""
	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(args.glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")
	
	
	"""
	Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
	File with predictions is processed by the official scorer after iterating over all instances.
	"""
	results_path = 'data/results/%d.%s.%s.key' % (int(time()), args.test_set, args.merge_strategy)
	with open(results_path, 'w') as results_f:
		for batch_idx, batch in enumerate(chunks(eval_instances, args.batch_size)):

			for sent_info in batch:
				idx_map_abs = sent_info['idx_map_abs']
				sent_bert = get_bert_embedding(sent_info['tokenized_sentence'])

				for mw_idx, tok_idxs in idx_map_abs:
					curr_id = sent_info['senses'][mw_idx]
					"""check if a word contains sense id"""
					if curr_id is None:
						continue
			
					curr_lemma = sent_info['lemmas'][mw_idx]
					if curr_lemma not in lemmas:
						continue  # skips hurt performance in official scorer

					curr_postag = sent_info['pos'][mw_idx]
					curr_tokens = [sent_info['tokens'][i] for i in tok_idxs]
					if curr_lemma not in glove_embeddings.keys():
						continue
					currVec_g = torch.from_numpy(glove_embeddings[curr_lemma].reshape(300, 1)).to(device)
					currVec_c = torch.mean(torch.stack([sent_bert[i][1] for i in tok_idxs]), dim=0)				

					disSims = []

					for sense_id in senseKeys:
						"""check if curr_lemma is in both pre-trained sense matrices and current dataset"""
						if curr_lemma == sense_id.split('%')[0]: 
							A_matrix = torch.from_numpy(A[sense_id]).to(device)
							z = (torch.mm(W, currVec_c) - torch.mm(A_matrix, currVec_g)).norm() ** 2
							disSims.append((z.item(), sense_id))
						
					"""take the neareast neighbour as the predicted sense"""
					disSims = sorted(disSims, key=lambda x: x[0])
					# print('disSims', disSims)
					minDissim = disSims[0]
					predict = minDissim[1]
					
					results_f.write('%s %s\n' % (curr_id, predict))

					"""check if our prediction(s) was correct, register POS of mistakes"""
					n_instances += 1
					wsd_correct = False
					gold_sensekeys = id2senses[curr_id]
					# print('gold_sensekeys', gold_sensekeys[0])
					# print('predict', predict)

					#if len(set(predict).intersection(set(gold_sensekeys[0]))) > 0:
					if gold_sensekeys[0] == predict:
						n_correct += 1
						wsd_correct = True
					else:
						if len(predict) > 0:
							failed_by_pos[curr_postag].append((predict, gold_sensekeys))
						else:
							failed_by_pos[curr_postag].append((None, gold_sensekeys))


					"""register if our prediction belonged to a different POS than gold"""
					if len(predict) > 0:
						pred_sk_pos = get_sk_pos(predict)
						gold_sk_pos = get_sk_pos(gold_sensekeys[0])
						pos_confusion[gold_sk_pos][pred_sk_pos] += 1

					acc = n_correct / n_instances
					logging.info('ACC: %.3f (%d %d/%d)' % (
						acc, n_instances, sent_info['idx'], len(eval_instances)))



	if args.debug:
		logging.info('POS Failures:')
		for pos, fails in failed_by_pos.items():
			logging.info('%s fails: %d' % (pos, len(fails)))

		logging.info('POS Confusion:')
		for pos in pos_confusion:
			logging.info('%s - %s' % (pos, str(pos_confusion[pos])))


	logging.info('Running official scorer ...')
	run_scorer(args.wsd_fw_path, args.test_set, results_path)			
