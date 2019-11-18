import logging
from time import time
import argparse

import numpy as np
import lxml.etree


# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import Dataset, TensorDataset, DataLoader
# from torch.utils.data.dataset import random_split
# from torch.autograd import Variable

from bert_as_service import tokenizer as bert_tokenizer
from bert_as_service import bert_embed


logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

#device = 'cuda' if torch.cuda.is_availabe() else 'cpu'


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


def load_glove(glove_path):
	logging.info("Loading Glove Model........")
	f = open(glove_path,'r')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		vec = np.array(splitLine[1:])
		model[word] = vec
	logging.info("Done. Loaded %d words" % len(model))
	return model


		
def train(train_path, eval_path, glove_path, merge_strategy='mean', max_seq_len=512, max_instances=float('inf')):
	glove_embed = load_glove(glove_path)
	

	#words_found = 0
	#sense_mapping = get_sense_mapping(key_path)
	batch, batch_idx, batch_t0 = [], 0, time()
	for sent_idx, sent_et in enumerate(read_xml_sents(train_path)):
		entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'pos', 'id']}
		for ch in sent_et.getchildren():
			for k, v in ch.items():
				entry[k].append(v)
			entry['token_mw'].append(ch.text)

			# if 'id' in ch.attrib.keys():
			#   entry['senses'].append(sense_mapping[ch.attrib['id']])
			# else:
			#   entry['senses'].append(None)

		entry['token'] = sum([t.split() for t in entry['token_mw']], [])
		entry['sentence'] = ' '.join([t for t in entry['token_mw']])
		#print("token", entry['token'])

		


		bert_tokens = bert_tokenizer.tokenize(entry['sentence'])
		#print('bert_tokens', bert_tokens)


		if len(bert_tokens) < max_seq_len:
			batch.append(entry)

		if len(batch) == args.batch_size:

			embed_c = {}
			embed_g = {}
			emb_c = []
			emb_g = []


			batch_sents = [e['sentence'] for e in batch]
			#print("batch_sents", batch_sents)

			#Embed batch sentences and obtain contextualised word embeddings (1024 dimensions)
			batch_bert = bert_embed(batch_sents, merge_strategy=merge_strategy)
			#print('batch_bert', batch_bert)

			#Use a dictionary to store contextulised word embeddings
			for i in range(len(batch_bert)):
				#C=np.zeros((len()))
				#token_w = bert_tokenizer.tokenize(senteces.lower())
				for idx, embed in enumerate(batch_bert[i]):
					word = embed[0]
					vec_c = embed[1]
					embed_c[word] = vec_c
					emb_c.append(vec_c)
					
					if word in glove_embed:
						embed_g[word] = glove_embed[word]
						#words_found += 1
					else:
						embed_g[word] = np.random.normal(scale=0.6, size=(300, ))
					emb_g.append(embed_g[word])
                                                          
			emb_g = np.array(emb_g)
			emb_c = np.array(emb_c)
			

			
			#Start training
			#logging.info('Training.....')
			#logging.info('#Words in GloVe: %d' % words_found)
	  




			batch_tspan = time() - batch_t0
			logging.info('%.3f sents/sec - %d sents, %d words' % (args.batch_size/batch_tspan, sent_idx, len(embed_c)))

			batch, batch_t0 = [], time()
			batch_idx += 1
	  
	logging.info('#Number of contextualised embeddings: %d' % len(embed_c))

	logging.info('Writing Pre-trained GloVe Word Vectors ...')
	print("#Number of word embeddings = %d" % len(embed_g))


	outg_path = 'data/vectors/glove_embeddings.128.1.txt'
	outc_path = 'data/vectors/bert_embeddings.128.1.txt'
	with open(outg_path, 'w') as fvecs_out1:
		for vecs_g in emb_g:
			embed_str1 = ' '.join([str(v) for v in vecs_g])
			fvecs_out1.write('%s\n' % (embed_str1))
	logging.info('Written %s' % outg_path)
 
	with open(outc_path, 'w') as fvecs_out2:
		for vecs_c in emb_c:
			embed_str2 = ' '.join([str(v) for v in vecs_c])
			fvecs_out2.write('%s\n' % (embed_str2))
	logging.info('Written %s' % outc_path)


	# outg_path = 'data/vectors/glove_embeddings.128.full.txt'
	# outc_path = 'data/vectors/bert_embeddings.128.full.txt'
	# with open(outg_path, 'w') as fvecs_out1:
	# 	for w_g, vecs_g in embed_g.items():
	# 		embed_str1 = ' '.join([str(v) for v in vecs_g])
	# 		fvecs_out1.write('%s %s\n' % (w_g, embed_str1))
	# logging.info('Written %s' % outg_path)
 
	# with open(outc_path, 'w') as fvecs_out2:
	# 	for w_c, vecs_c in embed_c.items():
	# 		embed_str2 = ' '.join([str(v) for v in vecs_c])
	# 		fvecs_out2.write('%s %s\n' % (w_c ,embed_str2))
	# logging.info('Written %s' % outc_path)
 
 

 



   
   
   
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create Initial Contextualised Embeddings.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-wsd_fw_path', help='Path to Semcor', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('-dataset', default='semcor', help='Name of dataset', required=False,
						choices=['semcor', 'semcor_omsti'])
	parser.add_argument('-glove_path', help='Path to GloVe', required=False,
						default='external/glove/glove.840B.300d.txt')
	parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
	parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
						choices=['mean', 'first', 'sum'])
	parser.add_argument('-max_instances', type=float, default=float('inf'), help='Maximum number of examples for each sense', required=False)
	parser.add_argument('-out_path', help='Path to resulting vector set', required=False)
	args = parser.parse_args()


	if args.dataset == 'semcor':
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/small.data.128.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	elif args.dataset == 'semcor_omsti':
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'

	train(train_path, keys_path, args.glove_path, args.merge_strategy, args.max_seq_len, args.max_instances)