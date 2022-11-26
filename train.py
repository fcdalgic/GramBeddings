# Copyright 2015 The Grambeddings: An End-To-End Neural Model for Phishing URLClassification Through N-gram
# Embeddings Authors {Fırat Coşkun Dalgıç}.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from random import random

import numpy
import numpy as np
import csv
import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from Utils.DataUtils import GetDataAndLabelsFromFiles, CreateModelFileNameFromArgs, SaveResults, DatasetOptions, \
    add_bool_arg
from Model import NBeddingModel
from NGramSequenceTransformer import NBeddingTransformer, CharacterLevelTransformer, WeightInitializer

##################### TO MAKE MORE DETERMINISTIC EXPERIMENTS #########################
SEED = 43


# Function to initialize seeds for all libraries which might have stochastic behavior

import random
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

# Activate Tensorflow deterministic behavior
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    # Below code is not supported in Windows
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# Call the above function with seed value
set_global_determinism(seed=SEED)

######################################################################################


train_file = ''
val_file = ''
dataset_name = 'grambeddings'
out_dir = 'outputs'
CHAR_EMBEDDING_DIM = 95
loss = "binary_crossentropy"
optimizer = "Adam"
LABEL_PHISH = 1
LABEL_LEGIT = 0

"""
    # Example usages: 
    python .\train.py dataset=grambeddings --ngram_1=3 --ngram_2=4 --ngram_3=5 --max_seq_len=128 --attn_width=10 --embed_dim=15 --max_features=160000 --max_df=0.7 --min_df=1e-06 --rnn_cell_size=256
    python .\train.py dataset=ebubekir --ngram_1=4 --ngram_2=5 --ngram_3=6 --max_seq_len=16 --attn_width=5 --embed_dim=20 --max_features=1200 --max_df=0.9 --min_df=1e-06 --rnn_cell_size=256
    # To enable warm_start just specify the related argument:
    python .\train.py warm_start -> warm_start is enabled
    python .\train.py            -> warm_start is disabled
    
    # To enable case_insensitive just specify the related argument:
    python .\train.py case_insensitive -> case_insensitive is enabled
    python .\train.py                  -> case_insensitive is disabled
    
"""


def get_args():
    parser = argparse.ArgumentParser(
        """Extracting Top-K Selected NGrams according tp selected scoring method.""")

    parser.add_argument("-d", "--dataset", type=DatasetOptions, default=DatasetOptions.grambeddings,
                        choices=list(DatasetOptions), help="dataset name")
    parser.add_argument("-o", "--output", type=str, default=out_dir,
                        help="The output directory where scores will be stored")

    parser.add_argument("-mn", "--model_name", type=str, default='asdas'
                        , help="Model filename, if it is None then automatically named from given arguments.")

    # Input ngram selections
    parser.add_argument("-n1", "--ngram_1", type=int, default=4, help="Ngram value of first   ngram embedding layer")
    parser.add_argument("-n2", "--ngram_2", type=int, default=5, help="Ngram value of second  ngram embedding layer")
    parser.add_argument("-n3", "--ngram_3", type=int, default=6, help="Ngram value of third   ngram embedding layer")
    # Feature Selection Parameters
    parser.add_argument("-maxf", "--max_features", type=int, default=160000, help="Maximum number of features")
    parser.add_argument("-madf", "--max_df", type=float, default=0.7, help="Embedding dimension for Embedding Layer")
    parser.add_argument("-midf", "--min_df", type=float, default=1e-06, help="Embedding dimension for Embedding Layer")
    parser.add_argument("-msl", "--max_seq_len", type=int, default=128,
                        help="The maximum sequence length to trim our transformed sequences")
    add_bool_arg(parser, 'case_insensitive', False)
    add_bool_arg(parser, 'warm_start', False)
    parser.add_argument("-wm", "--warm_mode", type=WeightInitializer, default=WeightInitializer.randomly_initialize,
                        choices=list(WeightInitializer), help="The selected Embedding Layer weight initializing "
                                                              "method. Only matters when warm_start is set True")

    parser.add_argument("-ed", "--embed_dim", type=int, default=15, help="Embedding dimension for Embedding Layer")
    parser.add_argument("-aw", "--attn_width", type=int, default=10, help="The attention layer width")
    parser.add_argument("-rnn", "--rnn_cell_size", type=int, default=128, help="The recurrent size")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epoch to train our model")

    parser.add_argument("-dp", "--save_deep_features", type=int, default=0,
                        help="Whether save or not logits. 0 False, True Otherwise")

    args = parser.parse_args()
    return args


def Process(args):
    print(args)
    ####################################### Loading Dataset  #######################################
    print('####################################### Loading Dataset  #######################################')
    train_file = 'data/' + args.dataset.value + '/train.csv'
    val_file = 'data/' + args.dataset.value + '/test.csv'
    train_samples, train_labels = GetDataAndLabelsFromFiles(train_file)
    val_samples, val_labels = GetDataAndLabelsFromFiles(val_file)
    print('Completed')

    ################################ Character Level Transformation ################################
    print('################################ Character Level Transformation ################################')
    transformer_char = CharacterLevelTransformer(args.max_seq_len, embedding_dim=CHAR_EMBEDDING_DIM,
                                                 case_insensitive=args.case_insensitive)
    char_vocab_size, char_embedding_matrix = transformer_char.Fit()
    train_sequences_char = transformer_char.Transform(train_samples)
    val_sequences_char = transformer_char.Transform(val_samples)
    print('Completed')

    ############################### First NGram Input Transformation ###############################
    print('############################### First NGram Input Transformation ###############################')
    transformer_1 = NBeddingTransformer(
        ngram_value=args.ngram_1,
        max_num_features=args.max_features,
        max_document_length=args.max_seq_len,
        min_df=args.min_df,
        max_df=args.max_df,
        embedding_dim=args.embed_dim,
        case_insensitive=args.case_insensitive,
        weight_mode = args.warm_mode.value,
    )
    print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_1)
    selected_ngrams_1, selected_ngram_scores_1, weight_matrix_1, vocab_size_1, idf_dict_1 = transformer_1.Fit(
        train_samples, train_labels)
    print("Starting convert train texts to train sequences for n = ", args.ngram_1)
    train_sequences_1 = transformer_1.Transform(train_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram_1)
    val_sequences_1 = transformer_1.Transform(val_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_1)
    train_sequences_1 = np.array(train_sequences_1, dtype='float32')
    val_sequences_1 = np.array(val_sequences_1, dtype='float32')
    print('Completed')

    ################################ 2nd NGram Input Transformation ################################
    print('################################ 2nd NGram Input Transformation ################################')
    transformer_2 = NBeddingTransformer(
        ngram_value=args.ngram_2,
        max_num_features=args.max_features,
        max_document_length=args.max_seq_len,
        min_df=args.min_df,
        max_df=args.max_df,
        embedding_dim=args.embed_dim,
        case_insensitive=args.case_insensitive,
        weight_mode=args.warm_mode.value,
    )
    print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_2)
    selected_ngrams_2, selected_ngram_scores_2, weight_matrix_2, vocab_size_2, idf_dict_2 = transformer_2.Fit(
        train_samples, train_labels)
    print("Starting convert train texts to train sequences for n = ", args.ngram_2)
    train_sequences_2 = transformer_2.Transform(train_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram_2)
    val_sequences_2 = transformer_2.Transform(val_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_2)
    train_sequences_2 = np.array(train_sequences_2, dtype='float32')
    val_sequences_2 = np.array(val_sequences_2, dtype='float32')
    print('Completed')

    ################################ 3rd NGram Input Transformation ################################
    print('################################ 3rd NGram Input Transformation ################################')
    transformer_3 = NBeddingTransformer(
        ngram_value=args.ngram_3,
        max_num_features=args.max_features,
        max_document_length=args.max_seq_len,
        min_df=args.min_df,
        max_df=args.max_df,
        embedding_dim=args.embed_dim,
        case_insensitive=args.case_insensitive,
        weight_mode=args.warm_mode.value,
    )
    print("Fitting input data in transformer to select best ngrams for n = ", args.ngram_3)
    selected_ngrams_3, selected_ngram_scores_3, weight_matrix_3, vocab_size_3, idf_dict_3 = transformer_3.Fit(
        train_samples, train_labels)
    print("Starting convert train texts to train sequences for n = ", args.ngram_3)
    train_sequences_3 = transformer_3.Transform(train_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram_3)
    val_sequences_3 = transformer_3.Transform(val_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_3)
    train_sequences_3 = np.array(train_sequences_3, dtype='float32')
    val_sequences_3 = np.array(val_sequences_3, dtype='float32')
    print('Completed')

    ############################## Initializing the Parallel Networks ##############################
    print('############################## Initializing the Parallel Networks ##############################')
    char_model = NBeddingModel(
        vocab_size=char_vocab_size,
        embedding_dim=CHAR_EMBEDDING_DIM,
        max_seq_length=args.max_seq_len,
        embedding_matrix=char_embedding_matrix,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    signal_model_1 = NBeddingModel(
        vocab_size=vocab_size_1,
        embedding_dim=args.embed_dim,
        max_seq_length=args.max_seq_len,
        embedding_matrix=weight_matrix_1,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    signal_model_2 = NBeddingModel(
        vocab_size=vocab_size_2,
        embedding_dim=args.embed_dim,
        max_seq_length=args.max_seq_len,
        embedding_matrix=weight_matrix_2,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    signal_model_3 = NBeddingModel(
        vocab_size=vocab_size_3,
        embedding_dim=args.embed_dim,
        max_seq_length=args.max_seq_len,
        embedding_matrix=weight_matrix_3,
        rnn_cell_size=args.rnn_cell_size,
        attention_width=args.attn_width,
        warm_start=args.warm_start
    )

    print('Completed')

    ################################ Merging the Parallel Networks #################################
    print('################################ Merging the Parallel Networks #################################')
    last_layer_char, input_layer_char = char_model.CreateModel(embedding_layer_name="embed_char")
    last_layer_1, input_layer_1 = signal_model_1.CreateModel(embedding_layer_name="embed_ngram_1")
    last_layer_2, input_layer_2 = signal_model_2.CreateModel(embedding_layer_name="embed_ngram_2")
    last_layer_3, input_layer_3 = signal_model_3.CreateModel(embedding_layer_name="embed_ngram_3")

    # Merging whole ngram layers
    embedded_concats = tf.keras.layers.Concatenate()(
        [last_layer_char, last_layer_1, last_layer_2, last_layer_3])
    dense1 = Dense(2 * args.rnn_cell_size, activation="relu", name='deep_features')(embedded_concats)
    dropout = Dropout(0.2)(dense1)
    predictions = Dense(1, activation="sigmoid")(dropout)

    # Build and compile model
    model_name = ''
    if args.model_name is None:
        model_name = CreateModelFileNameFromArgs(opt=args)
    else:
        model_name = args.model_name

    log_dir = "outputs/tensorboard/" + model_name
    model = Model(inputs=[input_layer_char, input_layer_1, input_layer_2, input_layer_3], outputs=predictions)
    model.compile(optimizer=Adam(), loss=loss, metrics=[
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ])
    model.summary()

    train_labels = train_labels.reshape(-1, 1)
    val_labels = val_labels.reshape(-1, 1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print("Tensorbaord is activated url : ", log_dir)
    checkpoint_path = "outputs/training/models/" + model_name + ".h5"

    import time
    start = time.time()
    history = model.fit(
        x=[train_sequences_char, train_sequences_1, train_sequences_2, train_sequences_3],
        y=train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        # shuffle=True,
        validation_data=([val_sequences_char, val_sequences_1, val_sequences_2, val_sequences_3], val_labels),
        callbacks=[
            # EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=5, verbose=1, min_delta=1e-4, mode='min'),
            tensorboard_callback
        ]
    )

    # tf.keras.models.save_model(model, "outputs/training/models/best_model")
    print('Training is Completed')
    end = time.time()
    ############################ Extracting Best Epoch and it's scores #############################
    print("############################ Extracting Best Epoch and it's scores #############################")
    best_epoch_index = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
    best_train_accu = history.history['accuracy'][best_epoch_index]
    best_train_loss = history.history['loss'][best_epoch_index]
    best_valid_accu = history.history['val_accuracy'][best_epoch_index]
    best_valid_loss = history.history['val_loss'][best_epoch_index]

    best_tp = history.history['val_tp'][best_epoch_index]
    best_fp = history.history['val_fp'][best_epoch_index]
    best_tn = history.history['val_tn'][best_epoch_index]
    best_fn = history.history['val_fn'][best_epoch_index]
    # Calculating the TPR  ==> tpr = tp / (tp+fn)
    best_tpr = best_tp / (best_tp + best_fn)
    # Calculating the FPR  ==> fpr = fp / (tn+fp)
    best_fpr = best_fp / (best_tn + best_fp)

    best_precision = history.history['val_precision'][best_epoch_index]
    best_recall = history.history['val_recall'][best_epoch_index]
    best_auc = history.history['val_auc'][best_epoch_index]

    elapsed_time = end - start

    ######################################## Saving Results ########################################
    print("######################################## Saving Results ########################################")
    SaveResults(best_epoch_index + 1, best_train_accu, best_train_loss, best_valid_accu, best_valid_loss,
                best_tp, best_fp, best_tn, best_fn, best_tpr, best_fpr, best_precision, best_recall, best_auc,
                opt, elapsed_time)

    # embed_weights_1 = GetSpecifiedLayerWeightsByName('embed_ngram_1', model)
    # CreateAndSaveClassDistributionFromEmbeddingMatrix(opt.ngram_1, embed_weights_1, selected_ngrams_1)
    #
    # embed_weights_2 = GetSpecifiedLayerWeightsByName('embed_ngram_2', model)
    # CreateAndSaveClassDistributionFromEmbeddingMatrix(opt.ngram_2, embed_weights_2, selected_ngrams_2)
    #
    # embed_weights_3 = GetSpecifiedLayerWeightsByName('embed_ngram_3', model)
    # CreateAndSaveClassDistributionFromEmbeddingMatrix(opt.ngram_3, embed_weights_3, selected_ngrams_3)

    # deep_model_output =  GetSpecifiedLayerOutputByName('deep_features', model)
    # deep_model_inputs  = model.input
    # deep_model = Model(deep_model_inputs, deep_model_output)
    # deep_features_train = deep_model.predict(x=[train_sequences_char, train_sequences_1, train_sequences_2, train_sequences_3], verbose=1)
    # deep_features_valid = deep_model.predict(x=[val_sequences_char, val_sequences_1, val_sequences_2 , val_sequences_3])


if __name__ == "__main__":
    opt = get_args()
    Process(opt)
