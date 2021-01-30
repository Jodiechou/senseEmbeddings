import os
import argparse
import logging
from functools import lru_cache
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

wn_lemmatizer = WordNetLemmatizer()

import sys  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def get_args(
		emb_dim = 300,
		diag = False
			 ):

	parser = argparse.ArgumentParser(description='Evaluation of WiC solution using LMMS for sense comparison.')
	parser.add_argument('--emb_dim', default=emb_dim, type=int)
	parser.add_argument('-eval_set', default='train', help='Evaluation set', required=False, choices=['train', 'dev', 'test'])
	parser.add_argument('-sv_path', help='Path to sense vectors', required=False, default='data/vectors/senseMatrix.semcor_diagonal_gelu_large_{}_50.npz'.format(emb_dim))
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	parser.add_argument('-out_path', help='Path to .pkl classifier generated', default='data/models/wic_ac_gelu50.pkl', required=False)
	parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_diagonal_gelu_1024_{}_50.npz'.format(emb_dim))
	parser.add_argument('-device', default='cuda', type=str)
	args = parser.parse_args()

	return args


@lru_cache()
def wn_sensekey2synset(sensekey):
	"""Convert sensekey to synset."""
	lemma = sensekey.split('%')[0]
	for synset in wn.synsets(lemma):
		for lemma in synset.lemmas():
			if lemma.key() == sensekey:
				return synset
	return None


@lru_cache()
def wn_lemmatize(w, postag=None):
	w = w.lower()
	if postag is not None:
		return wn_lemmatizer.lemmatize(w, pos=postag[0].lower())
	else:
		return wn_lemmatizer.lemmatize(w)


def load_wic(setname='train', wic_path='external/wic'):
	data_entries = []
	pos_map = {'N': 'NOUN', 'V': 'VERB'}
	data_path = '%s/%s/%s.data.txt' % (wic_path, setname, setname)
	for line in open(data_path):
		word, pos, idxs, ex1, ex2 = line.strip().split('\t')
		idx1, idx2 = list(map(int, idxs.split('-')))
		data_entries.append([word, pos_map[pos], idx1, idx2, ex1, ex2])

	if setname == 'test':  # no gold
		return [e + [None] for e in data_entries]

	gold_entries = []
	gold_path = '%s/%s/%s.gold.txt' % (wic_path, setname, setname)
	for line in open(gold_path):
		gold = line.strip()
		if gold == 'T':
			gold_entries.append(True)
		elif gold == 'F':
			gold_entries.append(False)

	assert len(data_entries) == len(gold_entries)
	return [e + [gold_entries[i]] for i, e in enumerate(data_entries)]


def load_weight(path):
	logging.info("Loading Model Parameters W ...")
	weight = np.load(path)
	weight = weight.f.arr_0
	logging.info('Loaded Model Parameters W')
	return weight


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
			merged_vec = torch.from_numpy(merged_vec.reshape(1024, 1)).to(device)
			##merged_vec = torch.from_numpy(merged_vec.reshape(768, 1)).to(device)
			##merged_vec = torch.from_numpy(merged_vec).to(device)
		sent_tokens_vecs.append((token, merged_vec))

	return sent_tokens_vecs


def get_sk_pos(sk, tagtype='long'):
	# merges ADJ with ADJ_SAT

	if tagtype == 'long':
		type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
		return type2pos[get_sk_type(sk)]

	elif tagtype == 'short':
		type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
		return type2pos[get_sk_type(sk)]


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


def get_sk_type(sensekey):
	return int(sensekey.split('%')[1].split(':')[0])


class SensesVSM(object):

	def __init__(self, vecs_path, normalize=False):
		self.vecs_path = vecs_path
		self.labels = []
		self.matrix = []
		self.indices = {}
		self.ndims = 0

		if self.vecs_path.endswith('.txt'):
			self.load_txt(self.vecs_path)

		elif self.vecs_path.endswith('.npz'):
			self.load_npz(self.vecs_path)
		self.load_aux_senses()
		'''
		if normalize:
			self.normalize()
		'''

	def load_txt(self, txt_vecs_path):
		self.vectors = []
		with open(txt_vecs_path, encoding='utf-8') as vecs_f:
			for line_idx, line in enumerate(vecs_f):
				elems = line.split()
				self.labels.append(elems[0])
				self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))
		self.vectors = np.vstack(self.vectors)

		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}


	def load_npz(self, path):
		self.matrices = []
		logging.info("Loading Pre-trained Sense Matrices ...")
		loader = np.load(path, allow_pickle=True)    # A is loaded a 0d array
		loader = np.atleast_1d(loader.f.arr_0)       # convert it to a 1d array with 1 element
		self.A = loader[0]								 # a dictionary, key is sense id and value is sense matrix 
		self.labels = list(self.A.keys())
		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}
		logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(self.A))


	def load_aux_senses(self):

		self.sk_lemmas = {sk: get_sk_lemma(sk) for sk in self.labels}
		self.sk_postags = {sk: get_sk_pos(sk) for sk in self.labels}

		self.lemma_sks = defaultdict(list)
		for sk, lemma in self.sk_lemmas.items():
			self.lemma_sks[lemma].append(sk)
		self.known_lemmas = set(self.lemma_sks.keys())

		self.sks_by_pos = defaultdict(list)
		for s in self.labels:
			self.sks_by_pos[self.sk_postags[s]].append(s)
		self.known_postags = set(self.sks_by_pos.keys())

	'''
	def normalize(self, norm='l2'):
		norms = np.linalg.norm(self.vectors, axis=1)
		self.vectors = (self.vectors.T / norms).T
	'''

	def get_A(self, label):
		# return self.A[self.indices[label]]	## For txt file
		return self.A[label]	## For npz file


	# def lemma2sense_id(self):
	# 	self.inst = defaultdict(list)
	# 	for sk in self.labels:
	# 		lemself.sk_lemmas[sk]
	# 		self.inst[sk].append()



	def match_senses(self, vec, W, vec_g, lemma=None, postag=None, topn=100):
	##def match_senses(self, vec, vec_g, lemma=None, postag=None, topn=100):
		matches = []
		relevant_sks = []
		distance = []
		sense_scores = []
		
		if (lemma is None) or (lemma in self.known_lemmas):
			# if (postag is None) or (postag in self.sks_by_pos[self.sk_postags[s]] for s in self.lemma_sks[lemma]):
				for sk in self.lemma_sks[lemma]:
					relevant_sks.append(sk)
					A_matrix = torch.from_numpy(self.A[sk]).to(device)
					senseVec = A_matrix * vec_g
					##senseVec = torch.mm(A_matrix, vec_g).squeeze(1)
					##context_vec = vec
					context_vec = torch.mm(W, vec).squeeze(1)
					sim = torch.dot(context_vec, senseVec) / (context_vec.norm() * senseVec.norm())
					sense_scores.append(sim)
			# else:
			# 	print('postag is not matched, lemma:{}, postag:{}'.fotmat(lemma, postag))
			# 	for unk_sk in self.labels:
			# 		relevant_sks.append(unk_sk)
			# 		unk_A_matrix = torch.from_numpy(self.A[unk_sk]).to(device)
			# 		unk_senseVec = unk_A_matrix * vec_g
			# 		unk_context_vec = torch.mm(W, vec).squeeze(1)
			# 		unk_sim = torch.dot(unk_context_vec, unk_senseVec) / (unk_context_vec.norm() * unk_senseVec.norm())
			# 		sense_scores.append(unk_sim)

		else:		## For the lemma that is not related to any sense_id
			print('lemma not related to any sense_id:', lemma)
			## return [None, 0.0]
			for unk_sk in self.labels:
				relevant_sks.append(unk_sk)
				unk_A_matrix = torch.from_numpy(self.A[unk_sk]).to(device)
				unk_senseVec = unk_A_matrix * vec_g
				unk_context_vec = torch.mm(W, vec).squeeze(1)
				unk_sim = torch.dot(unk_context_vec, unk_senseVec) / (unk_context_vec.norm() * unk_senseVec.norm())
				sense_scores.append(unk_sim)

		matches = list(zip(relevant_sks, sense_scores))
		matches = sorted(matches, key=lambda x: x[1], reverse=True)
		# print('matches', matches)
		return matches[:topn]


if __name__ == '__main__':

	args = get_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because Jodie doesn't have a GPU !!")
		args.device = 'cpu'

	device = torch.device(args.device)
	
	# results_path = 'data/results/wic.compare.%s.txt' % args.eval_set


	word2id = dict()
	word2sense = dict()

	
	lemmas = []
	vectors = []
	sense2idx = []

	senses_vsm = SensesVSM(args.sv_path, normalize=True)
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased')
	model.eval()

	
	W = load_weight(args.load_weight_path)
	W = torch.from_numpy(W).to(device)

	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(args.glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")


	logging.info('Processing sentences ...')
	instances, labels = [], []
	for wic_idx, wic_entry in enumerate(load_wic(args.eval_set, wic_path='external/wic')):
		word, postag, idx1, idx2, ex1, ex2, gold = wic_entry
			

		bert_ex1 = get_bert_embedding(ex1)
		bert_ex2 = get_bert_embedding(ex2)

		# example1
		ex1_curr_word, ex1_curr_vector = bert_ex1[idx1]
		ex1_curr_lemma = wn_lemmatize(word, postag)
		if ex1_curr_lemma not in glove_embeddings:
			vec_g1 = torch.randn(args.emb_dim, dtype=torch.float32, device=device, requires_grad=False)
			# continue
		##vec_g1 = torch.from_numpy(glove_embeddings[ex1_curr_lemma].reshape(300, 1)).to(device)
		else:
			vec_g1 = torch.from_numpy(glove_embeddings[ex1_curr_lemma]).to(device)
	
		ex1_matches = senses_vsm.match_senses(ex1_curr_vector, W, vec_g1, lemma=ex1_curr_lemma, postag=postag, topn=None)
		##ex1_matches = senses_vsm.match_senses(ex1_curr_vector, vec_g1, lemma=ex1_curr_lemma, postag=postag, topn=None)
		# if len(ex1_matches) == 0:
		#    continue

		# ex1_match_senses = [(wn_sensekey2synset(sk), score) for sk, score in ex1_matches]

		ex1_A_matric = torch.from_numpy(senses_vsm.get_A(ex1_matches[0][0])).to(device)
		ex1_sense_vec = ex1_A_matric * vec_g1
		ex1_sense_vec = ex1_sense_vec.cpu().detach().numpy()
		ex1_context_vec = torch.mm(W, ex1_curr_vector).squeeze(1).cpu().detach().numpy()

		# example2
		ex2_curr_word, ex2_curr_vector = bert_ex2[idx2]
		ex2_curr_lemma = wn_lemmatize(word, postag)
		if ex2_curr_lemma not in glove_embeddings:
			vec_g2 = torch.randn(args.emb_dim, dtype=torch.float32, device=device, requires_grad=False)
			# continue
		##vec_g2 = torch.from_numpy(glove_embeddings[ex2_curr_lemma].reshape(300, 1)).to(device)
		else:
			vec_g2 = torch.from_numpy(glove_embeddings[ex2_curr_lemma]).to(device)
			
		ex2_matches = senses_vsm.match_senses(ex2_curr_vector, W, vec_g2, lemma=ex2_curr_lemma, postag=postag, topn=None)
		##ex2_matches = senses_vsm.match_senses(ex2_curr_vector, vec_g2, lemma=ex2_curr_lemma, postag=postag, topn=None)
		# if len(ex2_matches) == 0:
		#    continue

		# ex2_match_senses = [(wn_sensekey2synset(sk), score) for sk, score in ex2_matches]

		ex2_A_matric = torch.from_numpy(senses_vsm.get_A(ex2_matches[0][0])).to(device)
		ex2_sense_vec = ex2_A_matric * vec_g2
		ex2_sense_vec = ex2_sense_vec.cpu().detach().numpy()
		ex2_context_vec = torch.mm(W, ex2_curr_vector).squeeze(1).cpu().detach().numpy()


		# ex1_best = ex1_match_senses[0][0]
		# ex2_best = ex2_match_senses[0][0]

		# '''For baseline ***'''
		# ex1_best = vec_g1
		# ex2_best = vec_g2

		# s1_sim = np.dot(ex1_best.cpu().detach(), ex2_best.cpu().detach())	
		# instances.append([s1_sim])
		# '''***'''

		s1_sim = np.dot(ex1_context_vec, ex2_context_vec)
		s2_sim = np.dot(ex1_sense_vec, ex2_sense_vec)
		s3_sim = np.dot(ex1_context_vec, ex1_sense_vec)
		s4_sim = np.dot(ex2_context_vec, ex2_sense_vec)

		instances.append([s1_sim, s2_sim, s3_sim, s4_sim])
		labels.append(gold)

	logging.info('Training Logistic Regression ...')
	clf = LogisticRegression(random_state=42)
	clf.fit(instances, labels)

	logging.info('Saving model to %s' % args.out_path)
	joblib.dump(clf, args.out_path)
