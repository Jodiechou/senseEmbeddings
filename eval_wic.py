import os
import argparse
import logging
from functools import lru_cache
from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wn_lemmatizer = WordNetLemmatizer()

import sys  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from bert_as_service import bert_embed


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


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

"""
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
"""

def load_weight(path):
	logging.info("Loading Model Parameters W ...")
	weight = np.load(path)
	weight = np.array(weight)
	logging.info('Loaded Model Parameters W')
	return weight

"""
def match_senses(vec, lemma=None, postag=None, topn=100):

        relevant_sks = []
        for sk in self.labels:
            if (lemma is None) or (self.sk_lemmas[sk] == lemma):
                if (postag is None) or (self.sk_postags[sk] == postag):
                    relevant_sks.append(sk)
        relevant_sks_idxs = [self.indices[sk] for sk in relevant_sks]

        sims = np.dot(self.vectors[relevant_sks_idxs], np.array(vec))
        matches = list(zip(relevant_sks, sims))

        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        return matches[:topn]
"""


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
    #print('sensekey', sensekey)
    #print('sensekey %', sensekey.split('%')[1])
    #print('sensekey.split[1].split', sensekey.split('%')[1].split(':')[0])
    return int(sensekey.split('%')[1].split(':')[0])


class SensesVSM(object):

    def __init__(self, vecs_path, normalize=False):
        self.vecs_path = vecs_path
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
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
        self.ndims = self.vectors.shape[1]

    def load_npz(self, npz_vecs_path):
        loader = np.load(npz_vecs_path)
        self.labels = loader['labels'].tolist()
        self.vectors = loader['vectors']

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

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
    def get_vec(self, label):
        return self.vectors[self.indices[label]]


    def match_senses(self, vec, w, lemma=None, postag=None, topn=100):
        matches = []
        relevant_sks = []
        distance = []
        for sk in self.labels:
            if (lemma is None) or (self.sk_lemmas[sk] == lemma):
                if (postag is None) or (self.sk_postags[sk] == postag):
                    relevant_sks.append(sk)
        relevant_sks_idxs = [self.indices[sk] for sk in relevant_sks]
        # sims = np.dot(self.vectors[relevant_sks_idxs], np.array(vec))
        for idx in relevant_sks_idxs:
           z = np.matmul(vec, w) - self.vectors[idx]
           curr_dist = norm(z)
           distance.append(curr_dist)

        matches = list(zip(relevant_sks, distance))
        matches = sorted(matches, key=lambda x: x[1])
        print('matches', matches)
        return matches[:topn]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of WiC solution using LMMS for sense comparison.')
    parser.add_argument('-sv_path', help='Path to LMMS vectors', required=True)
    parser.add_argument('-eval_set', default='dev', help='Evaluation set', required=False, choices=['train', 'dev', 'test'])
    parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_1024_300.npy')
    args = parser.parse_args()

    results_path = 'data/results/wic.compare.%s.txt' % args.eval_set


    word2id = dict()
    word2sense = dict()

    sensekeys = []
    lemmas = []
    vectors = []
    sense2idx = []



    # matrix_a = np.load('data/vectors/senseEmbed.semcor_300.npz')

    senses_vsm = SensesVSM(args.sv_path, normalize=True)
    """
    sensesEmbed= load_senseEmbeddings(args.sv_path)
    senseKeys = list(sensesEmbed.keys())
    vectors = list(sensesEmbed.values())
    #print('sensesEmbed', sensesEmbed)
    lemmas = [elem.split('%')[0] for elem in senseKeys]
    sense2idx = [(sen, i) for i, sen in enumerate(senseKeys)]
    #print('vectors shape:', vectors.shape) #  shape: (206, 300) -- (the number of senses, 300 dimemsions)
    """
    weight = load_weight(args.load_weight_path)

    logging.info('Processing sentences ...')
    n_instances, n_correct = 0, 0
    with open(results_path, 'w') as results_f:  # store results in WiC's format
        for wic_idx, wic_entry in enumerate(load_wic(args.eval_set, wic_path='external/wic')):
            word, postag, idx1, idx2, ex1, ex2, gold = wic_entry

            bert_ex1, bert_ex2 = bert_embed([ex1, ex2], merge_strategy='mean')

            # example1
            ex1_curr_word, ex1_curr_vector = bert_ex1[idx1]
            ex1_curr_lemma = wn_lemmatize(word, postag)
            # ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)

            # if senses_vsm.ndims == 1024:
            #    ex1_curr_vector = ex1_curr_vector

            # elif senses_vsm.ndims == 1024+1024:
            #    ex1_curr_vector = np.hstack((ex1_curr_vector, ex1_curr_vector))

            # ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)
            ex1_matches = senses_vsm.match_senses(ex1_curr_vector, weight, lemma=ex1_curr_lemma, postag=postag, topn=None)
            if len(ex1_matches) == 0:
               continue

            ex1_synsets = [(wn_sensekey2synset(sk), score) for sk, score in ex1_matches]
            ex1_wsd_vector = senses_vsm.get_vec(ex1_matches[0][0])

            # example2
            ex2_curr_word, ex2_curr_vector = bert_ex2[idx2]
            ex2_curr_lemma = wn_lemmatize(word, postag)
            # ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)
            """
            if senses_vsm.ndims == 1024:
                ex2_curr_vector = ex2_curr_vector

            elif senses_vsm.ndims == 1024+1024:
                ex2_curr_vector = np.hstack((ex2_curr_vector, ex2_curr_vector))
            """
            # ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)
            ex2_matches = senses_vsm.match_senses(ex2_curr_vector, weight, lemma=ex2_curr_lemma, postag=postag, topn=None)
            if len(ex2_matches) == 0:
               continue

            ex2_synsets = [(wn_sensekey2synset(sk), score) for sk, score in ex2_matches]
            ex2_wsd_vector = senses_vsm.get_vec(ex2_matches[0][0])

            ex1_best = ex1_synsets[0][0]
            ex2_best = ex2_synsets[0][0]
            print('ex1_best', ex1_best)
            print('ex2_best', ex2_best)
            n_instances += 1

            pred = False
            if len(ex1_synsets) == 1:
                pred = True

            elif ex1_best == ex2_best:
                pred = True

            elif ex1_best != ex2_best:
                pred = False

            if pred:
                results_f.write('T\n')
            else:
                results_f.write('F\n')

            if pred == gold:
                n_correct += 1
            # else:
            #     print('WRONG')

            # print(wic_idx, pred, gold)
            # print(word, postag, idx1, idx2, ex1, ex2, gold)
            # print(ex1_synsets)
            # print(ex2_synsets)

            acc = n_correct/n_instances
            logging.info('ACC: %f (%d/%d)' % (acc, n_correct, n_instances))

logging.info('Saved predictions to %s' % results_path)
