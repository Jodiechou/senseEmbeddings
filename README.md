# Sense embeddings

## Table of Contents

   * [Dependencise](#dependencise)
   * [Installation](#installation)
   * [Obtain contextualised embeddings using BERT](#obtain-contextualised-embeddings-using-BERT)

## Dependencise
- Pytorch
- Numpy
- [bert-as-service](https://github.com/hanxiao/bert-as-service)

## Installation
### Packages
To install additional packages used by this project run:
```
$ pip install bert-serving-server server 
$ pip3  install  bert-serving-client  client  #independent of 'bert-serving-server'
```
Note that the server MUST be running on Python>=3.5 with Tensorflow>=1.10.
### External Data
Download pre-trained BERT (large-cased)
```
$ cd external/bert  # from repo home
$ wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
$ unzip cased_L-24_H-1024_A-16.zip
```
Download pre-trained GloVe (Common Crawl, 840B tokens)
```
$ cd external/bert  # from repo home
$ wget http://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip cased_L-24_H-1024_A-16.zip
```

Download the WSD Evaluation Framework
```
$ cd external/wsd_eval  # from repo home
$ wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
$ unzip WSD_Evaluation_Framework.zip
```
## Obtain contextualised embeddings using BERT
### Loading BERT

