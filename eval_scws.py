import sys
#import time
import nltk

import numpy as np
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import sent_tokenize
from bert_as_service import bert_embed
import logging
import argparse
import scipy



logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')




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


def extractLocation(context, word):
	line = context.strip().split(' ')
	location = -1
	for j in range(len(line)):
		if line[j] == '<b>':
			location = j
			line = line[:j]+[line[j+1]]+line[j+3:]
			break
	if line[location]!=word:
		print(line[location], word)
	return location, line

 
def smooth(context):
	line = context.split(' ')
	i = 0
	while i!=len(line):
	#	if line[i] == '(' or line[i] == ')' or line[i] == '"' or line[i] == '[' or line[i] == ']':
	#		line = line[:i]+line[i+1:]
	#		continue
		if line[i] not in lemmas:
			if '-' in line[i] and line[i] != '-':
				line[i] = line[i].replace('-',' - ')
				line = line[:i]+line[i].split(' ')+line[i+1:] 
				continue
			elif '/' in line[i] and line[i]!='</b>' and line[i] != '/' :
				line[i] = line[i].replace('/',' / ')
				line = line[:i]+line[i].split(' ')+line[i+1:] 
				continue
			elif line[i] != '<b>' and line[i] != '</b>':
				line = line[:i]+line[i+1:]
				continue
		i+=1
	return ' '.join(line)


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
					
			# context1 = smooth(context1)
			# context2 = smooth(context2)
			# print('comtext1 after smoothing', context1)
			location1, context1 = extractLocation(context1, word1)
			location2, context2 = extractLocation(context2, word2)

			instance.append((context1, context2, location1, location2, aveR))

	return instance




if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation on SCWS dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
	parser.add_argument('-sv_path', help='Path to sense vectors', required=True)
	# parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_1024_{}.npy'.format(300))
	parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_1024_300.npy')
	args = parser.parse_args()


	word2id = dict()
	word2sense = dict()

	sensekeys = []
	lemmas = []
	vectors = []
	sense2idx = []
	


	# matrix_a = np.load('data/vectors/senseEmbed.semcor_300.npz')


	sensesEmbed= load_senseEmbeddings(args.sv_path)
	senseKeys = list(sensesEmbed.keys())
	vectors = list(sensesEmbed.values())
	#print('sensesEmbed', sensesEmbed)
	lemmas = [elem.split('%')[0] for elem in senseKeys]
	sense2idx = [(sen, i) for i, sen in enumerate(senseKeys)]
	#print('vectors shape:', vectors.shape) #  shape: (206, 300) -- (the number of senses, 300 dimemsions)

	w = load_weight(args.load_weight_path)


	# if len(sys.argv) != 3:
	# 	print('usage: python eval_scws.py test.in test.out')
	# testInFile = sys.argv[1]
	# testOutFile = sys.argv[2]

	
	#start_time = time.time()
	# wordCount = 0
	# UNKCount = 0

	logging.info('Formating testing data')
	eval_instances = load_eval_scws_set()
	logging.info('Finish formating testing data')
	# print(eval_instances[0])

	human_ratings = []
	similarities = []

	for sent in eval_instances:
		sent_bert1 = bert_embed(sent[0], merge_strategy=args.merge_strategy)
		sent_bert2 = bert_embed(sent[1], merge_strategy=args.merge_strategy)


		contEmbed1 = sent_bert1[sent[2]]
		contEmbed2 = sent_bert2[sent[3]]

		print('word 1: ', contEmbed1[0][0])
		print('word 2: ', contEmbed2[0][0])
		

		dist2vec1 = []
		dist2vec2 = []

		

		for inst in sense2idx:
			if contEmbed1[0][0] == inst[0].split('%')[0]:
				index1 = inst[1]
				senseVec1 = vectors[index1]
				z1 = np.matmul(contEmbed1[0][1], w) - senseVec1
				curr_dist1 = norm(z1)
				dist2vec1.append((curr_dist1, senseVec1))

			if contEmbed2[0][0] == inst[0].split('%')[0]:
				index2 = inst[1]
				senseVec2 = vectors[index2]
				z2 = np.matmul(contEmbed2[0][1], w) - senseVec2
				curr_dist2 = norm(z2)
				dist2vec2.append((curr_dist2, senseVec2))

		if len(dist2vec1)>0 and len(dist2vec2)>0:
			sort_dist1 = sorted(dist2vec1, key=lambda x: x[0])
			sort_dist2 = sorted(dist2vec2, key=lambda x: x[0])


			# print('sort_dist1', sort_dist1)
			# print('sort_dist2', sort_dist2)
		
			minDist1 = sort_dist1[0]
			minDist2 = sort_dist2[0]

			embed1 = minDist1[1]
			embed2 = minDist2[1]

			cos_sim = dot(embed1, embed2)/(norm(embed1)*norm(embed2))

			print('cos_sim', cos_sim)
			similarities.append(cos_sim)

			curr_rating = float(sent[4])
			human_ratings.append(curr_rating)

	# Compute the Spearman Rank and Pearson Correlation Coefficient against human ratings
	similarities = np.array(similarities)
	human_ratings = np.array(human_ratings)

	print('human_ratings', human_ratings)
	print('similarities', similarities)

	spr = scipy.stats.spearmanr(human_ratings, similarities)
	pearson = scipy.stats.pearsonr(human_ratings, similarities)


	if args.debug:
		logging.info('Spearman Rank: %.3f' % (spr))
		logging.info('Pearson Correlation Coefficient: %.3f' % (pearson))

