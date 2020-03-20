import os
import logging
import argparse
from time import time
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
#import cupy as cp
from nltk.corpus import wordnet as wn

from bert_as_service import bert_embed
# from vectorspace import SensesVSM
# from vectorspace import get_sk_pos

import re


logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


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
			# logging.info('number of sents: %d' %(sent_idx))

	return eval_instances


@lru_cache()
def wn_sensekey2synset(sensekey):
	"""Convert sensekey to synset."""
	lemma = sensekey.split('%')[0]
	for synset in wn.synsets(lemma):
		for lemma in synset.lemmas():
			if lemma.key() == sensekey:
				return synset
	return None


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


def str_scores(scores, n=3, r=5):
	"""Convert scores list to a more readable string."""
	return str([(l, round(s, r)) for l, s in scores[:n]])


def load_senseEmbeddings(path):
	logging.info("Loading Pre-trained Sense Embeddings ...")
	embed_sense = {}
	with open(path, 'r') as sfile:
		for line in sfile:
			splitLine = line.split(' ')
			sense = splitLine[0]
			vec_sense = np.array(splitLine[1:], dtype='float32')
			embed_sense[sense] = vec_sense
	logging.info("Done. Loaded %d words from Pre-trained Sense Embeddings" % len(embed_sense))
	return embed_sense


def load_weight(path):
	logging.info("Loading Model Parameters W ...")
	weight = np.load(path)
	weight = np.array(weight)
	logging.info('Loaded Model Parameters W')
	return weight


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Nearest Neighbors WSD Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-sv_path', help='Path to sense vectors', required=True)
	parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_1024_{}.npy'.format(300))
	# parser.add_argument('-ft_path', help='Path to fastText vectors', required=False,
	# 					default='external/fasttext/crawl-300d-2M-subword.bin')
	parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('-test_set', default='ALL', help='Name of test set', required=False,
						choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'])
	parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
	parser.add_argument('-ignore_lemma', dest='use_lemma', action='store_false', help='Ignore lemma features', required=False)
	parser.add_argument('-ignore_pos', dest='use_pos', action='store_false', help='Ignore POS features', required=False)
	parser.add_argument('-thresh', type=float, default=-1, help='Similarity threshold', required=False)
	parser.add_argument('-k', type=int, default=1, help='Number of Neighbors to accept', required=False)
	parser.add_argument('-quiet', dest='debug', action='store_false', help='Less verbose (debug=False)', required=False)
	parser.set_defaults(use_lemma=True)
	parser.set_defaults(use_pos=True)
	parser.set_defaults(debug=True)
	args = parser.parse_args()

	"""
	Load pre-trianed sense embeddings for evaluation.
	Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
	Load fastText static embeddings if required.
	"""
	sensekeys = []
	lemmas = []
	vectors = []
	sense2idx = []


	sensesEmbed= load_senseEmbeddings(args.sv_path)
	senseKeys = list(sensesEmbed.keys())
	vectors = list(sensesEmbed.values())
	#print('sensesEmbed', sensesEmbed)
	lemmas = [elem.split('%')[0] for elem in senseKeys]
	sense2idx = [(sen, i) for i, sen in enumerate(senseKeys)]
	#print('vectors shape:', vectors.shape) #  shape: (206, 300) -- (the number of senses, 300 dimemsions)

	w = load_weight(args.load_weight_path)
	
	"""
	Initialize various counters for calculating supplementary metrics.
	"""
	n_instances, n_correct, n_unk_lemmas = 0, 0, 0
	correct_idxs = []
	num_options = []
	predictions = []
	currSenses = []
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
	eval_instances = load_wsd_fw_set(wsd_fw_set_path)
	
	"""
	Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
	File with predictions is processed by the official scorer after iterating over all instances.
	"""

	results_path = 'data/results/%d.%s.%s.key' % (int(time()), args.test_set, args.merge_strategy)
	with open(results_path, 'w') as results_f:
		for batch_idx, batch in enumerate(chunks(eval_instances, args.batch_size)):
			batch_sents = [sent_info['tokenized_sentence'] for sent_info in batch]

			# process contextual embeddings in sentences batches of size args.batch_size
			batch_bert = bert_embed(batch_sents, merge_strategy=args.merge_strategy)

			for sent_info, sent_bert in zip(batch, batch_bert):
				idx_map_abs = sent_info['idx_map_abs']

				for mw_idx, tok_idxs in idx_map_abs:
					curr_id = sent_info['senses'][mw_idx]

					if curr_id is None:
						continue

					curr_lemma = sent_info['lemmas'][mw_idx]
					print('curr_lemma', curr_lemma)
					print('sense2idx', sense2idx[0][0].split('%')[0])

					if args.use_lemma and curr_lemma not in lemmas:
						continue  # skips hurt performance in official scorer

					curr_postag = sent_info['pos'][mw_idx]
					curr_tokens = [sent_info['tokens'][i] for i in tok_idxs]
					currVec_c = np.array([sent_bert[i][1] for i in tok_idxs]).mean(axis=0)
					# This is for converting different dimensions of vectors. 
					currVec_c = currVec_c / np.linalg.norm(currVec_c)
					#print('current lemmas:', curr_lemma)  # curr_lemma is the lemma in both pre-trained sense lemmas and in current dataset

					disSims = []
					

					# Check the disimilarity between senses and a word
					# Only consider the possible set of senses (with the same lemma) for each word 
					for sen2idx in sense2idx:
						if curr_lemma == sen2idx[0].split('%')[0]: 
							index = sen2idx[1]
							senseVec = vectors[index]
							z = np.matmul(currVec_c, w) - senseVec
							disSim = np.dot(z, z)
							disSims.append((disSim, sen2idx[0]))
						else:
							predictions.append(None)	
					disSims.sort()
					minDissim = disSims[0]
					predict = minDissim[1]
					

					results_f.write('%s %s\n' % (curr_id, predict))


					# check if our prediction(s) was correct
					n_instances += 1
					gold_sensekeys = id2senses[curr_id]
					print('gold_sensekeys', gold_sensekeys[0])
					print('predict', predict)

					if gold_sensekeys[0] == predict:
						n_correct += 1


					acc = n_correct / n_instances
					logging.debug('ACC: %.3f (%d /%d)' % (acc, n_instances, len(eval_instances)))





					




					
