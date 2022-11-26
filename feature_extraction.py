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

# This file is responsible for extracting deep features from the FCL which is the last part before proceeding
# classification task.

import numpy as np
import csv
import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras import Model

from Utils.DataUtils import GetDataAndLabelsFromFiles, CreateModelFileNameFromArgs, DatasetOptions, add_bool_arg, \
    check_dir
from Model import ZhangAttention
from NGramSequenceTransformer import NBeddingTransformer, CharacterLevelTransformer, WeightInitializer

train_file = 'data/train.csv'
val_file = 'data/test.csv'
out_dir = 'outputs/features/'
CHAR_EMBEDDING_DIM = 69
loss = "binary_crossentropy"
optimizer = "Adam"
LABEL_PHISH = 1
LABEL_LEGIT = 0

PREDICT_BATCH_SIZE = 80000
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_args():
    parser = argparse.ArgumentParser(
        """Extracting Top-K Selected NGrams according tp selected scoring method.""")

    parser.add_argument("-d", "--dataset", type=DatasetOptions, default=DatasetOptions.grambeddings_augmMode_not_trained,
                        choices=list(DatasetOptions), help="dataset name")

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

    parser.add_argument("-mn", "--model_name", type=str, default='best_model'
                        , help="Model filename, if it is None then automatically named from given arguments.")

    args = parser.parse_args()
    return args


def formatdata(data):
    for row in data:
        yield ["%0.4f" % v for v in row if isinstance(v, float)]


def Process(args):
    print(args)

    model_name = ''
    if args.model_name is None:
        model_name = CreateModelFileNameFromArgs(opt=args)
    else:
        model_name = args.model_name

    file_name = 'outputs/training/models/' + model_name + '.h5'
    model = tf.keras.models.load_model(file_name, custom_objects={'ZhangAttention': ZhangAttention})
    print(model.count_params())

    deep_model_output = GetSpecifiedLayerOutputByName('deep_features', model)
    deep_model_inputs = model.input
    deep_model = Model(deep_model_inputs, deep_model_output)
    deep_model.trainable = False

    # tf.keras.utils.plot_model(
    #     model, to_file='model222.png', show_shapes=True, show_dtype=False,
    #     show_layer_names=False, rankdir='LR', expand_nested=False, dpi=300
    # )

    deep_model.summary()

    ################################ Character Level Transformation #########################################
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
        weight_mode=args.warm_mode.value,
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

    print("Checking if output directory exists")
    check_dir(out_dir)
    check_dir(out_dir +  args.model_name)
    print("Done")

    print("Starting to extract features from training file")
    training_feature_file = out_dir +  args.model_name + "/train.csv"
    # with open(training_feature_file, 'w', encoding='utf-8', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     index = 1
    #     total_data = len(train_labels)
    #     for batch_char, batch_n1, batch_n2, batch_n3, batch_label in zip(get_every_n(train_sequences_char),
    #                                                                      get_every_n(train_sequences_1),
    #                                                                      get_every_n(train_sequences_2),
    #                                                                      get_every_n(train_sequences_3),
    #                                                                      get_every_n(train_labels),
    #                                                                      ):
    #         batch_features = deep_model.predict(x=[batch_char, batch_n1, batch_n2, batch_n3], verbose=1)
    #         batch_features = batch_features.astype(np.float16)
    #         batch_features = np.column_stack((batch_features, batch_label.astype(np.int32)))
    #         print("Appending current iterations deep features into training csv file. Completed :",
    #               (index * PREDICT_BATCH_SIZE), "\ttotal : ", total_data)
    #         writer.writerows(formatdata(batch_features))
    #         index = index + 1

    print("DOne")
    print("Starting to extract features from validation file")
    validation_feature_file = out_dir + args.model_name + "/valid.csv"
    with open(validation_feature_file, 'w', encoding='utf-8', newline='') as csv_file:
        total_data = len(val_labels)
        writer = csv.writer(csv_file)
        index = 1
        # for batch_char, batch_n1, batch_n2, batch_n3, batch_label in zip(get_every_n(val_sequences_char),
        #                                                                  get_every_n(val_sequences_1),
        #                                                                  get_every_n(val_sequences_2),
        #                                                                  get_every_n(val_sequences_3),
        #                                                                  get_every_n(val_labels),
        #                                                                  ):
        batch_features = deep_model.predict(x=[val_sequences_char, val_sequences_1, val_sequences_2, val_sequences_3], verbose=1)
        batch_features = batch_features.astype(np.float16)
        batch_features = np.column_stack((batch_features, val_labels))
        print("Appending current iterations deep features into validation csv file. Completed :",
              (index * PREDICT_BATCH_SIZE), "\ttotal : ", total_data)
        # writer.writerows(batch_features)
        writer.writerows(batch_features)
            # index = index + 1

    print("asd")


def get_every_n(a, batch_size=PREDICT_BATCH_SIZE):
    for i in range(a.shape[0] // batch_size):
        yield a[batch_size * i:batch_size * (i + 1)]

    remaining = a.shape[0] % batch_size
    if remaining != 0:
        yield a[-remaining:]


def GetSpecifiedLayerOutputByName(layer_name, model: tensorflow.keras.models.Model):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer.output


if __name__ == "__main__":
    opt = get_args()
    Process(opt)
