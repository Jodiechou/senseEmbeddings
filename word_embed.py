import logging
from time import time
import argparse

import numpy as np
import lxml.etree

from bert_as_service import tokenizer as bert_tokenizer
from bert_as_service import bert_embed


logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def get_sense_mapping(eval_path):
	sensekey_mapping = {}
	with open(eval_path) as keys_f:
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


def load_glove(glove_path):
	logging.info("Loading Glove Model........")
	f = open(glove_path,'r', encoding='utf8')
	model = {}
	for line in f:
		splitLine = line.split(' ')
		word = splitLine[0]
		vec = np.array(splitLine[1:], dtype='float32')
		model[word] = vec
	logging.info("Done. Loaded %d words" % len(model))
	return model


		
def train(train_path, eval_path, glove_path, merge_strategy='mean', max_seq_len=512, max_instances=float('inf')):
	glove_embed = load_glove(glove_path)
	embed_c = {}
	embed_g = {}
	vec = []
	

	sense_dict = {}
	senses_total = []
	sense_mapping = get_sense_mapping(eval_path)

	
	batch, batch_idx, batch_t0 = [], 0, time()
	for sent_idx, sent_et in enumerate(read_xml_sents(train_path)):
		entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'senses', 'pos', 'id']}
		for ch in sent_et.getchildren():
			for k, v in ch.items():
				entry[k].append(v)
			entry['token_mw'].append(ch.text)

			if 'id' in ch.attrib.keys():
				entry['senses'].append(sense_mapping[ch.attrib['id']])
			else:
				entry['senses'].append(None)


		entry['token'] = sum([t.split() for t in entry['token_mw']], [])
		entry['sentence'] = ' '.join([t for t in entry['token_mw']])


		bert_tokens = bert_tokenizer.tokenize(entry['sentence'])


		if len(bert_tokens) < max_seq_len:
			batch.append(entry)

		if len(batch) == args.batch_size:
			
			batch_sents = [e['sentence'] for e in batch]

			#Embed batch sentences and obtain contextualised word embeddings (1024 dimensions)
			batch_bert = bert_embed(batch_sents, merge_strategy=merge_strategy)


			for sent_info, sent_bert in zip(batch, batch_bert):
				idx_map_abs = []
				idx_map_rel = [(i, list(range(len(t.split()))))
								for i, t in enumerate(sent_info['token_mw'])]
				
				token_counter = 0
				for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
					idx_tokens = [i+token_counter for i in idx_tokens]
					token_counter += len(idx_tokens)
					idx_map_abs.append([idx_group, idx_tokens])


				
				for mw_idx, tok_idxs in idx_map_abs:
					if sent_info['senses'][mw_idx] is None:
						continue

					#senses_total.append(sent_info['senses'][mw_idx][0])

					# For the case of taking multiple words as a instance, for example, obtaining the embedding for 'too much' instead of two embeddings for 'too' and 'much'
					# Therefore, we use mean to compute the averaged vec of multiple words and consider it as the embedding for a instance which contains multiple words 
					vec_c = np.array([sent_bert[i][1] for i in tok_idxs], dtype=np.float32).mean(axis=0) 

					mw = sent_info['token_mw'][mw_idx]
					embed_c[mw] = vec_c

					if mw in glove_embed:
						embed_g[mw] = glove_embed[mw]
					else:
						embed_g[mw] = np.random.normal(scale=0.6, size=(300, ))

					sense = sent_info['senses'][mw_idx][0]
					sense_dict[mw] = sense



			batch_tspan = time() - batch_t0
			logging.info('%.3f sents/sec - %d sents, %d words' % (args.batch_size/batch_tspan, sent_idx, len(embed_c)))

			batch, batch_t0 = [], time()
			batch_idx += 1

	  
	logging.info('#Number of contextualised embeddings: %d' % len(embed_c))
	logging.info("#Number of word embeddings = %d" % len(embed_g))
	logging.info('#Number of senses: %d' % len(sense_dict))
	

	logging.info('Writing Pre-trained GloVe Word Vectors ...')
	


	#Save word embeddings and senses
	outg_path = 'data/vectors/glove_embeddings.semcor.full.txt'
	outc_path = 'data/vectors/bert_embeddings.semcor.full.txt'
	outs_path = 'data/vectors/senses.semcor.full.txt'
	with open(outg_path, 'w') as fvecs_out1:
		for w_g, vecs_g in embed_g.items():
			embed_str1 = ' '.join([str(v) for v in vecs_g])
			fvecs_out1.write('%s %s\n' % (w_g, embed_str1))
	logging.info('Written %s' % outg_path)
 
	with open(outc_path, 'w') as fvecs_out2:
		for w_c, vecs_c in embed_c.items():
			embed_str2 = ' '.join([str(v) for v in vecs_c])
			fvecs_out2.write('%s %s\n' % (w_c, embed_str2))
	logging.info('Written %s' % outc_path)

	with open(outs_path, 'w') as fvecs_out3:
		for mw, sense in sense_dict.items():
			fvecs_out3.write('%s %s\n' % (mw, sense))
	logging.info('Written %s' % outs_path)



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
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
	elif args.dataset == 'semcor_omsti':
		train_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
		keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'

	train(train_path, keys_path, args.glove_path, args.merge_strategy, args.max_seq_len, args.max_instances)