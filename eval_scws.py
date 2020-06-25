import os
import sys
import nltk
import numpy as np
import torch
from functools import lru_cache
import torch.nn as nn
import torch.nn.functional as F
# from numpy import dot
# from numpy.linalg import norm
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
import argparse
import scipy
from scipy.spatial.distance import cosine
wn_lemmatizer = WordNetLemmatizer()


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def get_args(
		emb_dim = 1024,
		diag = False
			 ):
	
	parser = argparse.ArgumentParser(description='Evaluation on SCWS dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--emb_dim', default=emb_dim, type=int)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
	parser.add_argument('-sv_path', help='Path to sense vectors', required=False, default='data/old/senseMatrix.semcor_diagonalI_{}_50.npz'.format(emb_dim))
	##parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	##parser.add_argument('--glove_embedding_path', default='external/glove/glove.768.txt')
	##parser.add_argument('--glove_embedding_path', default='external/glove/glove.500k.768.txt')
	parser.add_argument('--glove_embedding_path', default='external/glove/glove.1024.txt')
	##parser.add_argument('--load_weight_path', default='data/old/weight.semcor_diagonal_1024_{}_50.npz'.format(emb_dim))
	##parser.add_argument('--load_weight_path', default='data/old/weight.semcor_diagonalI_base_{}_100.npz'.format(emb_dim))
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
			##merged_vec = torch.from_numpy(merged_vec.reshape(1024, 1)).to(device)
			##merged_vec = torch.from_numpy(merged_vec.reshape(768, 1)).to(device)
			merged_vec = torch.from_numpy(merged_vec).to(device)
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


def load_eval_scws_set(lemmas):
	instance = []
	with open('external/SCWS/ratings.txt') as infp:
		for line in infp:
			line = line.strip().lower()
			line = line.split('\t')
			id = line[0]

			word1 = line[1]
			POS1 = line[2]
			word2 = line[3]
			POS2 = line[4]

			if word1 not in lemmas or word2 not in lemmas:
				continue
					
			context1 = sent_tokenize(line[5])
			context2 = sent_tokenize(line[6])
		
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

			location1, context1 = extractLocation(context1, word1)
			location2, context2 = extractLocation(context2, word2)

			instance.append((context1, context2, location1, location2, aveR))
			
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
	##W = load_weight(args.load_weight_path)
	##W = torch.from_numpy(W).to(device)
	#print('A matrixes:', A)

	vector_ones = torch.ones(args.emb_dim, dtype=torch.float32, device=device, requires_grad=False)
	
	senseKeys = list(A.keys())
	matrices = list(A.values())
	lemmas = [elem.split('%')[0] for elem in senseKeys]

	logging.info('Formating testing data')
	eval_instances = load_eval_scws_set(lemmas)
	logging.info('Finish formating testing data')

	human_ratings = []
	similarities = []

	for inst in eval_instances:
		sent_bert1 = get_bert_embedding(inst[0])
		sent_bert2 = get_bert_embedding(inst[1])
		

		'''
		obtain contextualised word embeddings for w1 and w2
		inst[2] and inst[3] are the locations for w1 and w2, respectively
		'''
		contEmbed1 = sent_bert1[inst[2]]
		contEmbed2 = sent_bert2[inst[3]]

		# contEmbed1_norm = contEmbed1[1] / contEmbed1[1].norm()
		# contEmbed2_norm = contEmbed2[1] / contEmbed2[1].norm()

		sense_scores1 = []
		sense_scores2 = []
		sense_vecors1 = []
		sense_vecors2 = []
		

		curr_lemma1 = wn_lemmatize(contEmbed1[0])
		curr_lemma2 = wn_lemmatize(contEmbed2[0])

		for sense_id in senseKeys:
			if curr_lemma1 == sense_id.split('%')[0]:
				
				##currVec_g1 = torch.from_numpy(glove_embeddings[contEmbed1[0]].reshape(300, 1)).to(device)
				currVec_g1 = torch.from_numpy(glove_embeddings[contEmbed1[0]]).to(device)
				A_matrix1 = torch.from_numpy(A[sense_id]).to(device)
				senseVec1 = A_matrix1 * currVec_g1
				# senseVec1 = senseVec1 / senseVec1.norm()
				# y1 = torch.mm(W, contEmbed1[1]).squeeze(1)
				# y1 = y1/y1.norm()

				sen_score1 = torch.dot(contEmbed1[1], senseVec1) / (contEmbed1[1].norm() * senseVec1.norm())
				
				sense_scores1.append(sen_score1)
				sense_vecors1.append(senseVec1)
 
			if curr_lemma2 == sense_id.split('%')[0]:
				##currVec_g2 = torch.from_numpy(glove_embeddings[contEmbed2[0]].reshape(300, 1)).to(device)
				currVec_g2 = torch.from_numpy(glove_embeddings[contEmbed2[0]]).to(device)
				A_matrix2 = torch.from_numpy(A[sense_id]).to(device)
				senseVec2 = A_matrix2 * currVec_g2
				# senseVec2 = senseVec2 / senseVec2.norm()
				# y2 = torch.mm(W, contEmbed2[1]).squeeze(1)
				# y2 = y2/y2.norm()

				sen_score2 = torch.dot(contEmbed2[1], senseVec2) / (contEmbed2[1].norm() * senseVec2.norm())

				sense_scores2.append(sen_score2)
				sense_vecors2.append(senseVec2)

		sense_scores1 = torch.tensor(sense_scores1)
		sense_scores2 = torch.tensor(sense_scores2)
		# print('curr_lemma1: %s, curr_lemma2: %s' %(curr_lemma1, curr_lemma2))
		# print('sense_scores1', sense_scores1)
		# print('sense_scores2', sense_scores2)

		m = nn.Softmax(dim=0)
		probability1 = m(sense_scores1)
		probability2 = m(sense_scores2)

		# print('probability1', probability1)
		# print('probability2', probability2)

	
	# 	'''Compute AvgSimC $$$'''
	# 	if len(sense_vecors1)>0 and len(sense_vecors2)>0:
	# 		AvgSimC = 0

	# 		# sort_dist1 = sorted(dist2vec1, key=lambda x: x[0])
	# 		# sort_dist2 = sorted(dist2vec2, key=lambda x: x[0])
		
	# 		# minDist1 = sort_dist1[0][0]
	# 		# minDist2 = sort_dist2[0][0]

	# 		# prob_max1 = torch.sigmoid(-minDist1)
	# 		# prob_max2 = torch.sigmoid(-minDist2)

	# 		for i in range(len(sense_vecors1)):
	# 			for j in range(len(sense_vecors2)):
	# 				# prob1 = probability1[i] / prob_max1
	# 				# prob2 = probability2[j] / prob_max2
	# 				prob1 = probability1[i]
	# 				prob2 = probability2[j]
	# 				# prob1 = prob1.cpu().detach().numpy()
	# 				# prob2 = prob2.cpu().detach().numpy()
	# 				# sense_vec1 = sense_vecors1[i]
	# 				# sense_vec2 = sense_vecors2[j]
	# 				# print('distance1: %f, distance2: %f' %(dist2vec1[i][0], dist2vec2[j][0]))
	# 				# print('probability1: %f, probability2: %f' %(prob1, prob2))
	# 				cos_sim = torch.dot(sense_vecors1[i], sense_vecors2[j]) / (sense_vecors1[i].norm() * sense_vecors2[j].norm())
	# 				# cos_sim = 1-cosine(sense_vec1,sense_vec2)
	# 				# cos_sim = cos_sim.cpu().detach().numpy()
	# 				AvgSimC += prob1 * prob2 * cos_sim
	# 		AvgSimC = AvgSimC.cpu().detach().numpy()
	# 		similarities.append(AvgSimC)	
	# 		curr_rating = float(inst[4])
	# 		human_ratings.append(curr_rating)
	# 		# print('**************')
	# print('Finished computing AvgSimC')
	# '''$$$'''

		'''Compute MaxSimC ***'''
		if len(probability1)>0 and len(probability2)>0:
		
			i = torch.argmax(probability1)
			j = torch.argmax(probability2)

			MaxSimC = torch.dot(sense_vecors1[i], sense_vecors2[j]) / (sense_vecors1[i].norm() * sense_vecors2[j].norm())
			similarities.append(MaxSimC.cpu().detach().numpy())
		
			curr_rating = float(inst[4])
			human_ratings.append(curr_rating)
	print('Finished computing MaxSimC')
	'''***'''
		

	# Compute the Spearman Rank and Pearson Correlation Coefficient against human ratings
	similarities = np.array(similarities)
	human_ratings = np.array(human_ratings)

	print('human_ratings', human_ratings)
	print('similarities', similarities)

	spr = scipy.stats.spearmanr(human_ratings, similarities)
	pearson = scipy.stats.pearsonr(human_ratings, similarities)


	print('Spearman Rank:', spr)
	print('Pearson Correlation:', pearson)
