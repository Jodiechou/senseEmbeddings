import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np 
import torch

import logging
import argparse
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')


def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def get_sense_data():
	data = []

	for synset in wn.all_synsets():
		# all_lemmas = [fix_lemma(lemma.name()) for lemma in synset.lemmas()]
		gloss = ' '.join(word_tokenize(synset.definition()))
		for lemma in synset.lemmas():
			# lemma_name = fix_lemma(lemma.name())
			# d_str = lemma_name + ' - ' + ' , '.join(all_lemmas) + ' - ' + gloss
			definition_str = gloss
			data.append((synset, lemma.key(), definition_str))

	data = sorted(data, key=lambda x: x[0])
	return data

if __name__ == '__main__':

	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer('stsb-bert-large')

	parser = argparse.ArgumentParser(description='Creates sense embeddings based on glosses and lemmas.')
	parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
	parser.add_argument('-out_path', help='Path to resulting vector set', required=False, default='data/vectors/gloss_embeddings.npz')
	args = parser.parse_args()

	logging.info('Preparing Gloss Data ...')
	glosses = get_sense_data()
	glosses_vecs = {}
	# print('glosses length', len(glosses))   ### glosses length 206978

	logging.info('Embedding Senses ...')
	t0 = time.time()
	for batch_idx, glosses_batch in enumerate(chunks(glosses, args.batch_size)):
		definitions = [e[-1] for e in glosses_batch]
		definitions_bert = model.encode(definitions)
		
		for (synset, sensekey, definition), definition_bert in zip(glosses_batch, definitions_bert):
			# print('*******************', sensekey, definition, definition_bert)
			glosses_vecs[sensekey] = definition_bert

		t_span = time.time() - t0
		n = (batch_idx + 1) * args.batch_size
		logging.info('%d/%d at %.3f per sec' % (n, len(glosses), n/t_span))

	print('len of glosses_vecs', len(glosses_vecs))


	logging.info('Writing Vectors %s ...' % args.out_path)
	np.savez(args.out_path, glosses_vecs)

	# with open(args.out_path, 'w') as vecs_senses_f:
	# 	for sensekey, sensekey_vecs in glosses_vecs.items():
	# 		vec = np.array(sensekey_vecs)
	# 		vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
	# 		vecs_senses_f.write('%s %s\n' % (sensekey, vec_str))
	logging.info('Done')

