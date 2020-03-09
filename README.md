# Sense embeddings

## Table of Contents

   * [Dependencise](#dependencise)
   * [Installation](#installation)
   * [Obtain contextualised and word embeddings](#obtain-contextualised-and-word-embeddings)

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
$ cd external/glove  # from repo home
$ wget http://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip glove.840B.300d.zip
```

Download the WSD Evaluation Framework
```
$ cd external/wsd_eval  # from repo home
$ wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
$ unzip WSD_Evaluation_Framework.zip
```
## Obtain contextualised and word embeddings
### Loading BERT
If ```~/.local/bin``` is not on your path, then do this in your .profile or something that gets loaded by your shell:
```
$ export PATH=~/.local/bin:$PATH
```
Start the bert service using this:
```
$ bert-serving-start -pooling_strategy NONE -model_dir external/bert/cased_L-24_H-1024_A-16 -pooling_layer -1 -2 -3 -4 -max_seq_len 512 -max_batch_size 32 -num_worker=1 -device_map 0 -cased_tokenization
```
After the server finishes preparing BERT for inference, you should see a message like this:
```
I:VENTILATOR:[__i:_ru:163]:all set, ready to serve request!
```

Now you need to leave this process running in this session and open a new session (i.e. new terminal or tab), return to the repository, and continue.

### Get BERT and GloVe embeddings
Usage descriptions:
```
Create Initial Sense Embeddings.

optional arguments:
  -h, --help            show this help message and exit
  -wsd_fw_path WSD_FW_PATH
                        Path to WSD Evaluation Framework (default: external/wsd_eval/WSD_Evaluation_Framework/)
  -dataset {semcor,semcor_omsti}
                        Name of dataset (default: semcor)
  -batch_size BATCH_SIZE
                        Batch size (BERT) (default: 32)
  -max_seq_len MAX_SEQ_LEN
                        Maximum sequence length (BERT) (default: 512)
  -merge_strategy {mean,first,sum}
                        WordPiece Reconstruction Strategy (default: mean)
  -max_instances MAX_INSTANCES
                        Maximum number of examples for each sense (default: inf)
```
Get contextualised and word embeddings by runing this:
```
$ python word_embed.py
```
## Train the model
After obtaining word embeddings and contextualised embeddings, we can train the model using this:
```
$ python train_linear.py
```

## Evaluation
```
$ python eval.py
```
