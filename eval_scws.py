import os
import sys
import nltk
import numpy as np
import torch
from functools import lru_cache
# from numpy import dot
# from numpy.linalg import norm
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
import argparse
import scipy
wn_lemmatizer = WordNetLemmatizer()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def get_args(
		emb_dim = 300,
		diag = False
			 ):

	parser = argparse.ArgumentParser(description='Evaluation on SCWS dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
	parser.add_argument('-sv_path', help='Path to sense vectors', required=True)
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_1024_{}.npz'.format(emb_dim))
	parser.add_argument('--device', default='cuda', type=str)
	args = parser.parse_args()

	return args


@lru_cache()
def wn_lemmatize(w):
	w = w.lower()
	return wn_lemmatizer.lemmatize(w)


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


def extractLocation(context, word):
	line = context.strip().split(' ')
	# print('line', line)
	location = -1
	for j in range(len(line)):
		if line[j] == '<b>':
			location = j
			line = line[:j]+[line[j+1]]+line[j+3:]
			break
	if line[location]!=word:
		print(line[location], word)
	line = ' '.join(line)
	return location, line

 
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


def load_eval_scws_set():
	instance = []
	with open('external/SCWS/ratings.txt') as infp:
		for line in infp:
			#print(line)
			line = line.strip().lower()
			line = line.split('\t')
			id = line[0]

			word1 = line[1]
			POS1 = line[2]
			word2 = line[3]
			POS2 = line[4]

			if word1 not in lemmas or word2 not in lemmas:
				continue
			# if word2sense[word1]!='3' or word2sense[word2]!='3':
			# 	print(word2sense[word1], word2sense[word2])
					
			context1 = sent_tokenize(line[5])
			context2 = sent_tokenize(line[6])
			# context1 = line[5]
			# context2 = line[6]
			# print('context1', context1)
			# print('context2', context2)

			found = False
			for sentence in context1:
				# print('sentence', sentence)
				if '<b>' in sentence and '</b>' in sentence:
					context1 = sentence
					#print('context1 = sentence', context1)
					found = True
					break
			if not found:
				print(context1)
			found = False
			for sentence in context2:
				if '<b>' in sentence and '</b>' in sentence:
					context2 = sentence
					found = True
					break
			if not found:
				print(context2)
					
			aveR = (line[7])
			Rs = map(float,line[8:])

			# print('context1 without smooth', context1)
					
			# context1 = smooth(context1)
			# print('context1 after smooth', context1)
			# context2 = smooth(context2)
			location1, context1 = extractLocation(context1, word1)
			location2, context2 = extractLocation(context2, word2)

			instance.append((context1, context2, location1, location2, aveR))
			# instance = {'context1':[], 'context2': [], 'location1': [], 'location2': [], 'aveR': []}

	return instance



if __name__ == '__main__':

	args = get_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because Jodie doesn't have a GPU !!")
		args.device = 'cpu'

	device = torch.device(args.device)

	word2id = dict()
	word2sense = dict()

	sensekeys = []
	lemmas = []
	vectors = []
	sense2idx = []

	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased')
	model.eval()

	logging.info("Loading Glove Embeddings........")
	glove_embeddings = load_glove_embeddings(args.glove_embedding_path)
	logging.info("Done. Loaded words from GloVe embeddings")
	

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

	logging.info('Formating testing data')
	eval_instances = load_eval_scws_set()
	logging.info('Finish formating testing data')

	human_ratings = []
	similarities = []
	# print('eval_instances', eval_instances)

	for inst in eval_instances:
		# print('inst', inst)
		sent_bert1 = get_bert_embedding(inst[0])
		sent_bert2 = get_bert_embedding(inst[1])
		# print("sent_bert1", sent_bert1)
		# print('sent_bert2', sent_bert1)

		"""
		obtain contextualised word embeddings for w1 and w2
		inst[2] and inst[3] are the locations for w1 and w2, respectively
		"""
		contEmbed1 = sent_bert1[inst[2]]
		contEmbed2 = sent_bert2[inst[3]]

		# print('word 1: ', contEmbed1[0])
		# print('word 2: ', contEmbed2[0])

		dist2vec1 = []
		dist2vec2 = []

		for sense_id in senseKeys:
			# print('word 1: ', contEmbed1[0])
			# print('*****',sense_id.split('%')[0])
			curr_lemma1 = wn_lemmatize(contEmbed1[0])
			if curr_lemma1 == sense_id.split('%')[0]:
				# index1 = s[1]
				# senseVec1 = vectors[index1]
				# z1 = np.matmul(contEmbed1[0][1], w) - senseVec1
				currVec_g1 = torch.from_numpy(glove_embeddings[contEmbed1[0]].reshape(300, 1)).to(device)
				A_matrix1 = torch.from_numpy(A[sense_id]).to(device)
				senseVec1 = torch.mm(A_matrix1, currVec_g1)
				z1 = (torch.mm(W, contEmbed1[1]) - senseVec1).norm() ** 2
				dist2vec1.append((z1, senseVec1))

			curr_lemma2 = wn_lemmatize(contEmbed2[0])
			if curr_lemma2 == sense_id.split('%')[0]:
				# index2 = s[1]
				# senseVec2 = vectors[index2]
				currVec_g2 = torch.from_numpy(glove_embeddings[contEmbed2[0]].reshape(300, 1)).to(device)
				A_matrix2 = torch.from_numpy(A[sense_id]).to(device)
				senseVec2 = torch.mm(A_matrix2, currVec_g2)
				z2 = (torch.mm(W, contEmbed2[1]) - senseVec2).norm() ** 2
				dist2vec2.append((z2, senseVec2))

		if len(dist2vec1)>0 and len(dist2vec2)>0:
			sort_dist1 = sorted(dist2vec1, key=lambda x: x[0])
			sort_dist2 = sorted(dist2vec2, key=lambda x: x[0])


			# print('sort_dist1', sort_dist1)
			# print('sort_dist2', sort_dist2)
		
			minDist1 = sort_dist1[0]
			minDist2 = sort_dist2[0]

			embed1 = minDist1[1].squeeze(1)
			embed2 = minDist2[1].squeeze(1)

			cos_sim = torch.dot(embed1, embed2)/(embed1.norm()*embed2.norm())

			# print('cos_sim', cos_sim)
			similarities.append(cos_sim.cpu().detach().numpy())

			curr_rating = float(inst[4])
			human_ratings.append(curr_rating)

	# Compute the Spearman Rank and Pearson Correlation Coefficient against human ratings
	similarities = np.array(similarities)
	human_ratings = np.array(human_ratings)

	# print('human_ratings', human_ratings)
	# print('similarities', similarities)

	spr = scipy.stats.spearmanr(human_ratings, similarities)
	pearson = scipy.stats.pearsonr(human_ratings, similarities)


	print('Spearman Rank:', spr)
	print('Pearson Correlation:', pearson)
