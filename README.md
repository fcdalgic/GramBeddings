![GRAM](https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/assets/images/logo.png "GRAM")
# Overview

The name *Grambeddings*   comes from the N-gram Embeddings. Grambeddings a novel and scalable deep learning model that is specialized to to recognize malicious websites from a given URL by employing both character and multiple n-gram embeddings in parallel sub-networks.  The suggested network model contains CNN, LSTM, and Attention (CLA) architecture to extract discriminative syntactical and temporal features to obtain high performance in terms of accuracy.

This repository contains the original implementation of the paper named "Grambeddings: An End-To-End Neural Model for Phishing URL-Classification Through N-Gram Embeddings"

You can check the paper from [here](https://www.sciencedirect.com/science/article/pii/S016740482200356X "here").
You can also download the published Grambeddings dataset from [here](https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/ "here").

![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg)

# Project Structure
|   Folder/File Name | Purpose  |
| ----------- | ------------ |
|   /data	|  Where the dataset files are stored under their unique dataset names.  For detailed explanation please refer to this [file](https://github.com/fcdalgic/GramBeddings/tree/main/data/Dataset.md). 
| /Utils | Where the common helper classes and function are stored. For detailed explanation please refer to this [file](https://github.com/fcdalgic/GramBeddings/tree/main/Utils/Utils.md).  |
| /outputs | Where the output/deployed files are stored.<br/>/features folder stores the feature_extraction.py results (deep features) <br/>/tensorboard folder stores tensorboard results<br/>/training folder stores the result.csv file which contains the training metrics. |
| train.py | Trains the Grambeddings model according to given arguements. |
| test.py  | Tests the pre-trained Grambeddings model according to given arguements. |
| single_channel_train.py | The proposed GramBeddings architecture is the composition of multiple CLA (CNN+LSTM+Attention) structures. This file provides training the single CLA architecture while preserving original implementation of proposed architecture. Therefore you can also conduct comparison between the scores of each level of n-grams.|
| NGramSequenceTransformer.py | The Grambeddings involves two phases, in the first part, we extract whole n-grams from given textual corpus, select top-k n-grams and prepares appropriate input for the Embedding layer by converting a given textual input into one dimensional n-gram array. This file conducts whole of these operation by performing fit and transform processes.|
| Model.py | Contains the class to produce Grambeddings' architecture |
| feature_extraction.py | This file is responsible for extracting deep features from pre-trained model file by hooking the input of Fully Connected Layer. The output features could be used in visualizing features with some libraries as we've done by employing UMap in our implementation/paper. |
| adverserial_generator.py | The additional file to generate adversarial examples by using specified samples |


# Benchmarking with Your Work
|   Benchmarking Type | How To  |
| ----------- | ------------ |
| Dataset-wise | In order to conduct benchmark between your dataset and Grambeddings Dataset, please download our 800k samples from [here](https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/ "here") |
| Algorithm-wise |In order to conduct benchmark test between our implementation and yours. Please follow the instructions given below: <br/> - Copy your dataset into the /data folder and convert your files into expected file formats. <br/> - Inside the DataUtils.py file locate the DatasetOptions classs and add your dataset name with it's relative folder name there.   <br/> -Train the model with your dataset and do not forget to cite us :) |

# Training
In order to train our model use the train.py file. The descriotion of each input arguement that affects the training phase are given below.

|   Arguement | Description  |
| ----------- | ------------ |
|  dataset | An enumeration value, specifies which dataset will be used to train model |
| output   | output directory, by defaults it is /output folder |
| model_name | the output model file name, if it is None then automatically named from given arguments |
| ngram_1 | Ngram value of first       ngram embedding layer |
| ngram_2 | Ngram value of second  ngram embedding layer |
| ngram_3 | Ngram value of third      ngram embedding layer |
| max_features | Maximum number of features will be used to initialize Embedding Matrix |
| maxf | Refers to Maximum Term Frequency, which will be used to filter out n-gram whose Term Frequency value is higher than given threshold. In our implementation we ignored it (set to 1) but for future works we put this additional functionality. |
| max_df | Refers to Maximum Document Frequency, which will be used to filter out n-gram whose DF value is higher than given threshold. In our implementation we ignored it (set to 1) but for future works we put this additional functionality. |
| min-df | Refers to Minimum Document Frequency, which will be used to filter out n-gram whose DF value is lower than given threshold. In our implementation we ignored it (set to 1) but for future works we put this additional functionality. |
| max_seq_len | Used to define the length of one-dimensional Embedding Matrix input array. We either trimmed or padded transformed n-gram sequences according to this arguements. |
| case_insensitive | Another additiona functionality, that we did not use in our paper but added for future works. Disables or enables the case sensivity while selecting n-grams and affects the whole model input data representation. |
| warm_start | Experimental use only, not fully implemented yet. We trying to figure out that if we pre-defined the Embedding Matrix's initial weigts, could we get better results.|
| warm_mode | Experimental use only, not fully implemented yet. The selected Embedding Layer weight initializing method. Only matters when warm_start is set True. |
| embed_dim | Embedding dimension for Embedding Layer |
| attn_width | The attention layer width|
| rnn_cell_size | The recurrent size|
| batch_size | Batch size |
| epochs | The number of epoch to train our model |

    #  Basic/General Usage:
    python .\train.py dataset=grambeddings --ngram_1=3 --ngram_2=4 --ngram_3=5 --max_seq_len=128 --attn_width=10 --embed_dim=15 --max_features=160000 --max_df=0.7 --min_df=1e-06 --rnn_cell_size=256
    python .\train.py dataset=ebubekir --ngram_1=4 --ngram_2=5 --ngram_3=6 --max_seq_len=16 --attn_width=5 --embed_dim=20 --max_features=1200 --max_df=0.9 --min_df=1e-06 --rnn_cell_size=256
    # To enable warm_start just specify the related argument:
    python .\train.py warm_start -> warm_start is enabled
    python .\train.py            -> warm_start is disabled
    
    # To enable case_insensitive just specify the related argument:
    python .\train.py case_insensitive -> case_insensitive is enabled
    python .\train.py                  -> case_insensitive is disabled



