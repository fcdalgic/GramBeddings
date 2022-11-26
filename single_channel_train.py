# Copyright 2015 The Grambeddings: An End-To-End Neural Model for Phishing URLClassification Through N-gram
# Embeddings Authors {Fırat Coşkun Dalgıç, Ahmet Selman Bozkır}.
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

import numpy
import numpy as np
import csv
import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from Utils.DataUtils import GetDataAndLabelsFromFiles
from Model import NBeddingModel
from NGramSequenceTransformer import NBeddingTransformer, CharacterLevelTransformer

train_file = 'data/grambeddings/train.csv'
val_file = 'data/grambeddings/test.csv'
out_dir = 'outputs'
CHAR_EMBEDDING_DIM = 15
loss = "binary_crossentropy"
optimizer = "Adam"
LABEL_PHISH = 1
LABEL_LEGIT = 0





def CreateModelFileNameFromArgs(opt):
    header = CreateFileNameHeaderFromArgs(opt)
    model_name = header + ".h5"
    return model_name


def CreateFileNameHeaderFromArgs(opt):
    header = "model"
    header = header + "_n_" + str(opt.ngram)
    header = header + "_mf_" + str(opt.max_features)
    header = header + "_maxdf_" + str(opt.max_df)
    header = header + "_mindf_" + str(opt.min_df)
    header = header + "_msl_" + str(opt.max_seq_len)
    header = header + "_dim_" + str(opt.embed_dim)
    header = header + "_attn_" + str(opt.attn_width)
    header = header + "_rnn_" + str(opt.rnn_cell_size)
    header = header + "_batch_" + str(opt.batch_size)
    header = header + "_epoch_" + str(opt.epochs)
    return header


def ConvertInputsToSequences(args, train_samples, train_labels, val_samples, val_labels):
    sequence_transformer = None
    selected_ngrams_1 = None
    selected_ngram_scores_1 = None

    if args.ngram == 1: # Character Model
        sequence_transformer = CharacterLevelTransformer(args.max_seq_len, embedding_dim=15)
        vocab_size_1, weight_matrix_1 = sequence_transformer.Fit()
    elif args.ngram > 2:
        sequence_transformer = NBeddingTransformer(
            ngram_value=args.ngram,
            max_num_features = args.max_features,
            max_document_length = args.max_seq_len,
            min_df = args.min_df,
            max_df=args.max_df,
            EMBEDDING_DIM = args.embed_dim,
        )
        print("Fitting input data in transformer to select best ngrams for n = ", args.ngram)
        selected_ngrams_1, selected_ngram_scores_1, weight_matrix_1, vocab_size_1, idf_dict_1 = sequence_transformer.Fit(
            train_samples, train_labels)


    print("Starting convert train texts to train sequences for n = ", args.ngram)
    train_sequences_1 =  sequence_transformer.Transform(train_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram)
    val_sequences_1   = sequence_transformer.Transform(val_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ", args.ngram)
    train_sequences_1 = np.array(train_sequences_1, dtype='float32')
    val_sequences_1  = np.array(val_sequences_1, dtype='float32')

    return train_sequences_1, val_sequences_1, selected_ngrams_1, weight_matrix_1, vocab_size_1


def Process(args):
    print(args)

    train_samples, train_labels = GetDataAndLabelsFromFiles(args.train)
    val_samples  , val_labels   = GetDataAndLabelsFromFiles(args.val)

    train_sequences, val_sequences, selected_ngrams, weight_matrix, vocab_size = ConvertInputsToSequences(args, train_samples, train_labels, val_samples, val_labels)

    pre_model = None
    if args.ngram == 1:
        pre_model = NBeddingModel(
            vocab_size=vocab_size,
            embedding_dim=CHAR_EMBEDDING_DIM,
            max_seq_length=args.max_seq_len,
            embedding_matrix=weight_matrix,
            rnn_cell_size=args.rnn_cell_size,
            attention_width=args.attn_width
        )
    else:
        pre_model = NBeddingModel(
            vocab_size=vocab_size,
            embedding_dim=args.embed_dim,
            max_seq_length=args.max_seq_len,
            embedding_matrix=weight_matrix,
            rnn_cell_size=args.rnn_cell_size,
            attention_width=args.attn_width
        )

    # last_layer_char, input_layer_char = char_model.CreateModel(embedding_layer_name="embed_char")
    last_layer_1, input_layer_1  = pre_model.CreateModel(embedding_layer_name="embed_ngram")

    # Merging whole ngram layers
    dense1 = Dense(2 * args.rnn_cell_size, activation="relu")(last_layer_1)
    dropout = Dropout(0.2)(dense1)
    predictions = Dense(1, activation="sigmoid")(dropout)

    # Build and compile model
    model = Model(inputs=[input_layer_1], outputs=predictions)
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
    val_labels   = val_labels.reshape(-1, 1)

    model_name = CreateModelFileNameFromArgs(opt=args)
    log_dir = "logs/fit/" + model_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print("Tensorbaord is activated url : ", log_dir)
    history = model.fit(
                        x=[train_sequences],
                        y=train_labels,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose = 1,
                        shuffle=True,
                        validation_data=([val_sequences], val_labels),
                        callbacks =[
                         #EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', restore_best_weights=True),
                         ModelCheckpoint(filepath=model_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=7,verbose=1, min_delta=1e-4, mode='min'),
                         tensorboard_callback
                        ]
    )

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

    SaveResults(best_epoch_index, best_train_accu, best_train_loss, best_valid_accu, best_valid_loss,
                best_tp, best_fp, best_tn, best_fn, best_tpr, best_fpr, best_precision, best_recall, best_auc,
                opt)


def SaveResults(best_epoch_index, best_train_accu, best_train_loss, best_valid_accu, best_valid_loss,
                best_tp, best_fp, best_tn, best_fn, best_tpr, best_fpr, best_precision, best_recall, best_auc,
                opt):
    # Saving results on CSV File
    csv_filename = "results.csv"
    # Check if file exists
    # ngram_1   max_features	max_df	min_df	max_seq_len	embed_dim	attn_width	rnn_cell_size	batch_size	epochs	best_epoch_index	best_train_acc	best_train_loss	best_val_acc	best_val_loss   best_tp   best_fp  best_tn best_fn    best_tpr  best_fpr    best_precision    best_recall   best_auc
    csv_content = list()
    csv_content.append(opt.ngram)
    csv_content.append(opt.max_features)
    csv_content.append(opt.max_df)
    csv_content.append(opt.min_df)
    csv_content.append(opt.max_seq_len)
    csv_content.append(opt.embed_dim)
    csv_content.append(opt.attn_width)
    csv_content.append(opt.rnn_cell_size)
    csv_content.append(opt.batch_size)
    csv_content.append(opt.epochs)

    csv_content.append(best_epoch_index)
    csv_content.append(best_valid_accu)
    csv_content.append(best_precision)
    csv_content.append(best_recall)
    csv_content.append(best_auc)

    # csv_content.append(best_train_accu)
    # csv_content.append(best_train_loss)
    # csv_content.append(best_valid_loss)
    #
    # csv_content.append(best_tp)
    # csv_content.append(best_fp)
    # csv_content.append(best_tn)
    # csv_content.append(best_fn)
    # csv_content.append(best_tpr)
    # csv_content.append(best_fpr)


    with open(csv_filename, 'a', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_content)

def GetSpecifiedLayerWeightsByName(layer_name, model : tensorflow.keras.models.Model):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer.get_weights()

def GetSpecifiedLayerOutputByName(layer_name, model : tensorflow.keras.models.Model):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer.output


def CreateAndSaveClassDistributionFromEmbeddingMatrix(ngram_value, embedding_weights: numpy.ndarray, selected_features):

    print('Going to calculate and save class-favor distribution from embedding matrix for n = .', ngram_value)

    filename = 'embedding_matrix_weigths_with_feature_names_and_class_info_for_ngram_' + str(ngram_value) + '.csv'

    cols = embedding_weights[0].shape[1]
    rows = embedding_weights[0].shape[0]
    # Converting numpy array to a list. Because it is easier to manipulate.
    weights_list = embedding_weights[0].tolist()

    # Finding the corresponding class similarities for each selected features
    train_samples, train_labels = GetDataAndLabelsFromFiles(train_file, convert_to_array=False)
    val_samples, val_labels = GetDataAndLabelsFromFiles(val_file, convert_to_array=False)
    total_samples = train_samples + val_samples
    total_labels = train_labels + val_labels

    indexes_phish = [idx for idx, element in enumerate(total_labels) if element == LABEL_PHISH]
    samples_phish = [total_samples[i] for i in indexes_phish]

    indexes_legit = [idx for idx, element in enumerate(total_labels) if element == LABEL_LEGIT]
    samples_legit = [total_samples[i] for i in indexes_legit]

    # Calculating class occurrences for each selected ngram
    print('Calculating class occurrences for each selected ngram')
    occurence_dict_phish = dict()
    occurence_dict_legit = dict()
    for ngram in selected_features:
        occurence_phish = sum(1 if ngram in sample else 0 for sample in samples_phish)
        occurence_legit = sum(1 if ngram in sample else 0 for sample in samples_legit)

        occurence_dict_phish[ngram] = occurence_phish
        occurence_dict_legit[ngram] = occurence_legit

        progress = len(occurence_dict_legit)
        if (progress % 100) == 0:
            progress_percent = (progress / len(selected_features)) * 100
            print("Progress : ", progress_percent)

    print('Class occurrences for each selected ngram is calculated')

    footer_class_info = list()
    print('Assigning favors of each selected ngram value')
    for ngram in selected_features:
        occurence_phish = occurence_dict_phish[ngram]
        occurence_legit = occurence_dict_legit[ngram]

        favor = 'not significant'
        percentage = 0
        if occurence_phish > occurence_legit:
            percentage = (100 * occurence_phish) / (occurence_legit + occurence_phish)
            favor = 'phish'
        elif occurence_legit > occurence_phish:
            percentage = (100 * occurence_legit) / (occurence_legit + occurence_phish)
            favor = 'legit'

        if percentage > 50:
            favor = 'highly ' + favor
        elif percentage > 25:
            favor = 'medium ' + favor
        else:
            favor = 'not significant'

        footer_class_info.append(favor)
        progress = len(footer_class_info)
        if (progress % 100) == 0:
            progress_percent = (progress / len(footer_class_info)) * 100
            print("Progress : ", progress_percent)

    final_matrix = list()
    for i, feature in enumerate(selected_features):
        weigths_for_feature =  weights_list[i]
        # Inserting the selected feature at the beginning of weights list
        weigths_for_feature.insert(0, feature)
        # Appending the favor at the end of the weight list
        weigths_for_feature.append(footer_class_info[i])
        final_matrix.append(weigths_for_feature)

    print('Ngram favors are assigned')

    weights_list.append(footer_class_info)

    print('Going to save Class-Favor Distribution of Embedding Matrix')
    with open(filename, 'a', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for list_item in final_matrix:
            writer.writerow(list_item)

if __name__ == "__main__":
    opt = get_args()
    Process(opt)
