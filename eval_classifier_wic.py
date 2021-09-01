import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import logging
from functools import lru_cache
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import math
from numpy.linalg import norm
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.linear_model import LogisticRegression
import joblib
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')
wn_lemmatizer = WordNetLemmatizer()
def get_args(
		emb_dim = 300,
		diag = False
			 ):
	parser = argparse.ArgumentParser(description='Evaluation of WiC solution using LMMS for sense comparison.')
	parser.add_argument('--emb_dim', default=emb_dim, type=int)
	parser.add_argument('--gloss_embedding_path', default='data/vectors/gloss_embeddings.npz')
	parser.add_argument('--lmms_embedding_path', default='../bias-sense/data/lmms_2048.bert-large-cased.npz')
	parser.add_argument('--ares_embedding_path', default='external/ares/ares_bert_large.txt')
	parser.add_argument('-eval_set', default='dev', help='Evaluation set', required=False, choices=['train', 'dev', 'test'])
	parser.add_argument('-sv_path', help='Path to sense vectors', required=False, default='data/vectors/senseMatrix.semcor_diagonal_linear_large_bertlast4layers_multiword_{}_50.npz'.format(emb_dim))
	parser.add_argument('-load_weight_path', default='data/vectors/weight.semcor_diagonal_linear_bertlast4layers_multiword_1024_{}_50.npz'.format(emb_dim))
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	parser.add_argument('-clf_path', help='Path to .pkl LR classifier', required=False, default='data/models/wic_linear_bertlast4layers_50.pkl')
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
def gelu(x):
	""" Original Implementation of the gelu activation function in Google Bert repo when initialy created.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
@lru_cache()
def wn_first_sense(lemma, postag=None):
	pos_map = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}
	first_synset = wn.synsets(lemma, pos=pos_map[postag])[0]
	found = False
	for lem in first_synset.lemmas():
		key = lem.key()
		if key.startswith('{}%'.format(lemma)):
			found = True
			break
	assert found
	return key
def load_wic(setname='dev', wic_path='external/wic'):
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
def load_lmms(npz_vecs_path):
	lmms = {}
	loader = np.load(npz_vecs_path)
	labels = loader['labels'].tolist()
	vectors = loader['vectors']
	for label, vector in list(zip(labels, vectors)):
		lmms[label] = vector
	return lmms
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
	# res = list(zip(tokenized_text[1:-1], outputs[0].cpu().detach().numpy()[0][1:-1])) ## [1:-1] is used to get rid of CLS] and [SEP]
	layers_vecs = np.sum([outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]], axis=0) ### use the last 4 layers
	res = list(zip(tokenized_text[1:-1], layers_vecs.cpu().detach().numpy()[0][1:-1]))
	
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
def load_gloss_embeddings(path):
	gloss_embeddings = {}
	loader = np.load(path, allow_pickle=True)    # gloss_embeddings is loaded a 0d array
	loader = np.atleast_1d(loader.f.arr_0)       # convert it to a 1d array with 1 element
	embeddings = loader[0]				 # a dictionary, key is sense id and value is embeddings
	for key, emb in embeddings.items():
		gloss_embeddings[key] = torch.from_numpy(emb)
	logging.info("Loaded %d gloss embeddings" % len(gloss_embeddings))
	return gloss_embeddings
def load_ares_txt(path):
		sense_vecs = {}
		with open(path, 'r') as sfile:
			for idx, line in enumerate(sfile):
				if idx == 0:
					continue
				splitLine = line.split(' ')
				label = splitLine[0]
				vec = np.array(splitLine[1:], dtype='float32')
				dim = vec.shape[0]
				sense_vecs[label] = vec
		return sense_vecs
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
		self.labels = list(ares_embeddings.keys())
		# self.labels = list(lmms.keys())
		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}
		logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(self.A))
	def load_B(self, path):
		self.matrices = []
		logging.info("Loading Pre-trained Sense Matrices ...")
		loader = np.load(path, allow_pickle=True)    # A is loaded a 0d array
		loader = np.atleast_1d(loader.f.arr_0)       # convert it to a 1d array with 1 element
		self.B = loader[0]								 # a dictionary, key is sense id and value is sense matrix 
		self.B_labels = list(self.B.keys())
		self.B_labels_set = set(self.B_labels)
		self.B_indices = {l: i for i, l in enumerate(self.B_labels)}
		logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(self.B))
	def load_C(self, path):
		self.matrices = []
		logging.info("Loading Pre-trained Sense Matrices ...")
		loader = np.load(path, allow_pickle=True)    # A is loaded a 0d array
		loader = np.atleast_1d(loader.f.arr_0)       # convert it to a 1d array with 1 element
		self.C = loader[0]								 # a dictionary, key is sense id and value is sense matrix 
		self.C_labels = list(self.C.keys())
		self.C_labels_set = set(self.C_labels)
		self.C_indices = {l: i for i, l in enumerate(self.C_labels)}
		logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(self.C))
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
	def get_A(self, label):
		if label not in self.A.keys():
			sense_matrix = None
		else: 
			sense_matrix = self.A[label]
		# return self.A[self.indices[label]]	## For txt file
		return sense_matrix	## For npz file
	def match_senses(self, currVec_c, currVec_g, lemma=None, postag=None, topn=100):
		matches = []
		relevant_sks = []
		distance = []
		sense_scores = []
		for sk in self.labels:
			if (lemma is None) or (self.sk_lemmas[sk] == lemma):
				if (postag is None) or (self.sk_postags[sk] == postag):
					relevant_sks.append(sk)
					if sk in self.A.keys():
						A_matrix = torch.from_numpy(self.A[sk]).to(device)
						static_sense_vec = A_matrix * currVec_g
					else:
						static_sense_vec = currVec_g
					cont_vec = torch.cat((currVec_c, currVec_c), 0)
					context_vec = torch.cat((currVec_g, cont_vec), 0)				
					ares_vec = torch.from_numpy(ares_embeddings[sk]).to(device)
					sense_vec = torch.cat((static_sense_vec, ares_vec), 0)
					sim = torch.dot(context_vec, sense_vec) / (context_vec.norm() * sense_vec.norm())
					sense_scores.append(sim)
		matches = list(zip(relevant_sks, sense_scores))
		matches = sorted(matches, key=lambda x: x[1], reverse=True)
		return matches[:topn]
if __name__ == '__main__':

	args = get_args()
	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because Jodie doesn't have a GPU !!")
		print("Switching to CPU because no GPU !!")
		args.device = 'cpu'
	device = torch.device(args.device)

	word2id = dict()
	word2sense = dict()
	lemmas = []
	vectors = []
	sense2idx = []
	relu = nn.ReLU(inplace=True)
	ares_embeddings = load_ares_txt(args.ares_embedding_path)
	# lmms = load_lmms(args.lmms_embedding_path)
	senses_vsm = SensesVSM(args.sv_path, normalize=True)
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
	model.eval()
	logging.info('Loading LR Classifer ...')
	clf = joblib.load(args.clf_path)
	results_path = 'data/results/wic.classify.%s.linear50.txt' % args.eval_set
	
	W = load_weight(args.load_weight_path)
	W = torch.from_numpy(W).to(device)
	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(args.glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")
	gloss_vecs = load_gloss_embeddings(args.gloss_embedding_path)
	
	logging.info('Processing sentences ...')
	n_instances, n_correct = 0, 0
	with open(results_path, 'w') as results_f:  # store results in WiC's format
		for wic_idx, wic_entry in enumerate(load_wic(args.eval_set, wic_path='external/wic')):
			word, postag, idx1, idx2, ex1, ex2, gold = wic_entry
			bert_ex1 = get_bert_embedding(ex1)
			bert_ex2 = get_bert_embedding(ex2)
			# example1
			ex1_curr_word, ex1_curr_vector = bert_ex1[idx1]
			ex1_curr_lemma = wn_lemmatize(word, postag)
			if ex1_curr_lemma not in glove_embeddings:
				vec_g1 = torch.randn(args.emb_dim, dtype=torch.float32, device=device, requires_grad=False)
			else:
				vec_g1 = torch.from_numpy(glove_embeddings[ex1_curr_lemma]).to(device)
				ex1_matches = senses_vsm.match_senses(ex1_curr_vector, vec_g1, lemma=ex1_curr_lemma, postag=postag, topn=None)
				
			ex1_A_matrix = senses_vsm.get_A(ex1_matches[0][0])
			if ex1_A_matrix is None:
				continue
			else:
				ex1_sense_matric = torch.from_numpy(ex1_A_matrix).to(device)
				ex1_static_sense_vec = ex1_sense_matric * vec_g1
			ex1_ares_vec = torch.from_numpy(ares_embeddings[ex1_matches[0][0]]).to(device)
			ex1_sense_vec = torch.cat((ex1_ares_vec, ex1_static_sense_vec), 0)
			ex1_sense_vec = ex1_sense_vec.cpu().detach().numpy()
			ex1_context_vec = torch.cat((ex1_curr_vector, ex1_curr_vector), 0)
			ex1_context_vec = torch.cat((ex1_context_vec, vec_g1), 0)
			ex1_context_vec = ex1_context_vec.cpu().detach().numpy()
			# example2
			ex2_curr_word, ex2_curr_vector = bert_ex2[idx2]
			ex2_curr_lemma = wn_lemmatize(word, postag)
			if ex2_curr_lemma not in glove_embeddings:
				vec_g2 = torch.randn(args.emb_dim, dtype=torch.float32, device=device, requires_grad=False)
			else:
				vec_g2 = torch.from_numpy(glove_embeddings[ex2_curr_lemma]).to(device)
				ex2_matches = senses_vsm.match_senses(ex2_curr_vector, vec_g2, lemma=ex2_curr_lemma, postag=postag, topn=None)
			
			ex2_A_matrix = senses_vsm.get_A(ex2_matches[0][0])
			if ex2_A_matrix is None:
				continue
			else:
				ex2_sense_matric = torch.from_numpy(ex2_A_matrix).to(device)
				ex2_static_sense_vec = ex2_sense_matric * vec_g2
			ex2_ares_vec = torch.from_numpy(ares_embeddings[ex2_matches[0][0]]).to(device)
			ex2_sense_vec = torch.cat((ex2_ares_vec, ex2_static_sense_vec), 0)
			ex2_sense_vec = ex2_sense_vec.cpu().detach().numpy()
			ex2_context_vec = torch.cat((ex2_curr_vector, ex2_curr_vector), 0)
			ex2_context_vec = torch.cat((ex2_context_vec, vec_g2), 0)
			ex2_context_vec = ex2_context_vec.cpu().detach().numpy()
			n_instances += 1
			s1_sim = np.dot(ex1_context_vec, ex2_context_vec)
			s2_sim = np.dot(ex1_sense_vec, ex2_sense_vec)
			s3_sim = np.dot(ex1_context_vec, ex1_sense_vec)
			s4_sim = np.dot(ex2_context_vec, ex2_sense_vec)
			s5_sim = np.dot(ex1_context_vec, ex2_sense_vec)
			s6_sim = np.dot(ex2_context_vec, ex1_sense_vec)
			pred = clf.predict([[s1_sim, s2_sim, s3_sim, s4_sim, s5_sim, s6_sim]])[0]
			
			if pred == True:
				results_f.write('T\n')
			else:
				results_f.write('F\n')
			if pred == gold:
				n_correct += 1
			acc = n_correct/n_instances
			logging.info('ACC: %f (%d/%d)' % (acc, n_correct, n_instances))
logging.info('Saved predictions to %s' % results_path)

