# Sense embeddings

## Table of Contents

   * [Dependencise](#dependencise)
   * [Installation](#installation)
   * [Obtain contextualised and word embeddings](#obtain-contextualised-and-word-embeddings)
   * [Train the model](#train-the-model)
   * [Evaluation](#evaluation)

## Dependencise
- Pytorch
- Numpy
- Scipy
- transformers

## Installation
Note that the server MUST be running on Python>=3.5 with Pytorch>=1.10.

### Additional Packages
To install additional packages used by this project run:
```
pip install -r requirements.txt
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

## Train the model
After obtaining word embeddings and contextualised embeddings, we can train the model using this:
```
usage: train_linear_diagonal.py [-h]
                                [--glove_embedding_path GLOVE_EMBEDDING_PATH]
                                [--num_epochs NUM_EPOCHS] [--loss {standard}]
                                [--emb_dim EMB_DIM]
                                [--diagonalize DIAGONALIZE] [--device DEVICE]
                                [--bert BERT] [--wsd_fw_path WSD_FW_PATH]
                                [--dataset {semcor,semcor_omsti}]
                                [--batch_size BATCH_SIZE] [--lr LR]
                                [--merge_strategy {mean,first,sum}]

Word Sense Mapping

optional arguments:
  -h, --help            show this help message and exit
  --glove_embedding_path GLOVE_EMBEDDING_PATH
  --num_epochs NUM_EPOCHS
  --loss {standard}
  --emb_dim EMB_DIM
  --diagonalize DIAGONALIZE
  --device DEVICE
  --bert BERT
  --wsd_fw_path WSD_FW_PATH
                        Path to Semcor
  --dataset {semcor,semcor_omsti}
                        Name of dataset
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --merge_strategy {mean,first,sum}
                        WordPiece Reconstruction Strategy

```

## WSD Evaluation
Usage description.
```
$ python eval.py -h
usage: eval.py [-h] [-glove_embedding_path GLOVE_EMBEDDING_PATH]
               [-sv_path SV_PATH] [-load_weight_path LOAD_WEIGHT_PATH]
               [-wsd_fw_path WSD_FW_PATH]
               [-test_set {senseval2,senseval3,semeval2007,semeval2013,semeval2015,ALL}]
               [-batch_size BATCH_SIZE] [-merge_strategy MERGE_STRATEGY]
               [-ignore_pos] [-thresh THRESH] [-k K] [-quiet] [-device DEVICE]

WSD Evaluation.

optional arguments:
  -h, --help            show this help message and exit
  -glove_embedding_path GLOVE_EMBEDDING_PATH
  -sv_path SV_PATH      Path to sense vectors (default: data/vectors/senseMatr
                        ix.semcor_diagonal_linear_large_300_200.npz)
  -load_weight_path LOAD_WEIGHT_PATH
  -wsd_fw_path WSD_FW_PATH
                        Path to WSD Evaluation Framework (default:
                        external/wsd_eval/WSD_Evaluation_Framework/)
  -test_set {senseval2,senseval3,semeval2007,semeval2013,semeval2015,ALL}
                        Name of test set (default: ALL)
  -batch_size BATCH_SIZE
                        Batch size (default: 64)
  -merge_strategy MERGE_STRATEGY
                        WordPiece Reconstruction Strategy (default: mean)
  -ignore_pos           Ignore POS features (default: True)
  -thresh THRESH        Similarity threshold (default: -1)
  -k K                  Number of Neighbors to accept (default: 2)
  -quiet                Less verbose (debug=False) (default: True)
  -device DEVICE

```
To replicate, use as follows:
```
$  python eval.py -sv_path data/vectors/senseMatrix.semcor_diagonal_linear_large_300_200.npz -test_set ALL
```


## WiC evaluation
You'll need to download the WiC dataset and place it in 'external/wic/':
```
$ cd external/wic
$ wget https://pilehvar.github.io/wic/package/WiC_dataset.zip
$ unzip WiC_dataset.zip
```


### Sense Comparison
Usage description.
```
usage: train_classifier.py [-h] [--emb_dim EMB_DIM]
                           [-glove_embedding_path GLOVE_EMBEDDING_PATH]
                           [-eval_set {train,dev,test}] [-sv_path SV_PATH]
                           [-load_weight_path LOAD_WEIGHT_PATH]
                           [-out_path OUT_PATH] [-device DEVICE]

Evaluation of WiC solution using LMMS for sense comparison.

optional arguments:
  -h, --help            show this help message and exit
  --emb_dim EMB_DIM
  -glove_embedding_path GLOVE_EMBEDDING_PATH
  -eval_set {train,dev,test}
                        Evaluation set
  -sv_path SV_PATH      Path to sense vectors
  -load_weight_path LOAD_WEIGHT_PATH
  -out_path OUT_PATH    Path to .pkl classifier generated
  -device DEVICE
```

To train binary classifier, use:
```
$ python train_classifier.py
```
Evaluation using Classifier
```
$ python eval_classifier_wic.py
```



