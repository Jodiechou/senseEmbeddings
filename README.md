# Sense embeddings

## Table of Contents

   * [Installation](#installation)

## Installation
### Packages
To install additional packages used by this project run:
```
$ pip install bert-serving-server server 
$ pip3  install  bert-serving-client  client  #independent of 'bert-serving-server'
```
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
## Experiment
