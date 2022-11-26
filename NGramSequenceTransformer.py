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

# This class is responsible for converting textual input into n-gram sequences, selecting best top-k n-gram from
# obtained n-gram vocabulary.

import warnings

import numpy as np
import csv

from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from enum import Enum


clear = lambda: os.system('cls')  # on Windows System


class NBeddingTransformer:
    def __init__(self, ngram_value, max_num_features, max_document_length, min_df, max_df, embedding_dim,
                 case_insensitive=False, k_all_enabled=True, weight_mode='randomly_initialize',
                 scoring_algorithm=chi2):
        self.ngram_value = ngram_value
        self.ngram_range = (ngram_value, ngram_value)
        self.max_num_features = max_num_features
        self.max_document_length = max_document_length
        self.min_df = min_df
        self.max_df = max_df
        self.embedding_dim = embedding_dim
        self.weight_mode = weight_mode
        self.k_all_enabled = k_all_enabled
        self.case_insensitive = case_insensitive
        self.scoring_algorithm = scoring_algorithm
        self.flag_fit = False
        self.embed_dict = {}
        self.model_ngram_relations = None
        if self.case_insensitive:
            warnings.warn("Case insensitivity is activated. Now every single text will be converted to lower-case "
                          "before finding the ngrams ")
        self.tk = Tokenizer(num_words=None, char_level=True, oov_token='cs.hacettepe.edu.tr',
                            lower=self.case_insensitive)

    def __ngrams_selection(self, train_data, train_labels,
                           max_df, min_df,
                           ngram_range_=(4, 4), max_num_features=6000,
                           analyzer_type='char'):
        """ Selects best ngrams from given training data by using TfidfVectorizer and Chi2 scoring algorithms

        Args:
            train_data: list of train text samples
            train_labels: list of train labels
            ngram_range_: range of n-grams
            max_num_features: maximum number of features to select
            analyzer_type: analyzer type for TfidfVectorizer 'word' or 'char'
        Returns:
            nothing
            @param train_data:
            @param train_labels:
            @param max_df:
            @param min_df:
            @param ngram_range_:
            @param max_num_features:
            @param analyzer_type:
            @return:
        """
        vectorizer = TfidfVectorizer(ngram_range=ngram_range_,
                                     sublinear_tf=True,
                                     analyzer=analyzer_type,
                                     lowercase=self.case_insensitive,
                                     # min_df=min_df, max_df=max_df,
                                     )

        X_train = vectorizer.fit_transform(train_data)
        vector_count = X_train.shape[1]

        print("Extracted features count after TF-IDF and before Chi2 :", vector_count)
        if vector_count < max_num_features and self.k_all_enabled:
            print("Not enough ngram found after vectorizer fit_transform. Expected = ", max_num_features, " got = ",
                  vector_count, " training continues by using k = all")
            selector = SelectKBest(chi2, k='all')
        else:
            selector = SelectKBest(chi2, k=max_num_features)
        X_new = selector.fit_transform(X_train, train_labels)

        mask = selector.get_support()  # list of booleans
        idf_dict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        selected_features = []
        selected_feature_scores = []
        for bool, feature, score in zip(mask, vectorizer.get_feature_names(), selector.scores_):
            if bool:
                selected_features.append(feature)
                selected_feature_scores.append(score)

        selected_feature_scores, selected_features = zip(*sorted(zip(selected_feature_scores, selected_features)))
        return selected_features, selected_feature_scores, idf_dict

    def __create_weight_matrix(self, selected_ngram_scores):
        """
        Creates a weight matrix to be able to initialize Embedding Layer weights later.
        @param selected_ngram_scores: Depends on the weight_mode parameter, it might be used to initialize embedding layer weights
        @return: embedding_matrix which is the warm_start weights of embedding layer.
        """
        embedding_matrix = []
        if self.weight_mode == 'randomly_initialize':
            initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
            embedding_matrix = initializer(shape=(len(self.tk.word_index), self.embedding_dim))
        elif self.weight_mode == 'init_by_indicies':
            # # create a weight matrix for words in training docs
            # initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
            # embedding_matrix = initializer(shape=(len(self.tk.word_index) + 1, self.embedding_dim))
            # embedding_matrix = np.array(embedding_matrix)
            # for word, i in self.tk.word_index.items():
            #     embedding_vector = self.tk.word_index.get(word)
            #     if embedding_vector is not None:
            #         embedding_matrix[i] = embedding_vector

            embedding_matrix = np.zeros((len(self.tk.word_index), self.embedding_dim))
            for word, i in self.tk.word_index.items():
                embedding_vector = self.tk.word_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

        elif self.weight_mode == 'init_by_ngram_scores':
            embedding_matrix = np.zeros((len(self.tk.word_index), self.embedding_dim))
            for word, i in self.tk.word_index.items():
                # embedding_vector = embed_dict.get(word)
                chi2_score = None
                if i < len(selected_ngram_scores):
                    chi2_score = selected_ngram_scores[i]

                if chi2_score is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = chi2_score
        elif self.weight_mode == 'init_by_ngram_scores_punished':
            embedding_matrix = np.zeros((len(self.tk.word_index), self.embedding_dim))
            mean_val = np.mean(np.asarray(list(self.ngram_dict.values()), dtype="float32"))
            missing_val = 0
            if mean_val > 0:
                missing_val = 0 - mean_val
            else:
                missing_val = mean_val

            for word, i in self.tk.word_index.items():
                # embedding_vector = embed_dict.get(word)
                chi2_score = None
                if word in self.ngram_dict:
                    chi2_score = self.ngram_dict[word]
                else:
                    chi2_score = missing_val

                if chi2_score is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = chi2_score
        elif self.weight_mode == 'init_zeros_and_single_one':
            # embedding_matrix.append(np.zeros(self.embedding_dim))  # (0, 69)
            for char, i in self.tk.word_index.items():  # from index 1 to 69
                onehot = np.zeros(self.embedding_dim)
                index = (i - 1) % self.embedding_dim
                onehot[index] = 1
                embedding_matrix.append(onehot)

            embedding_matrix = np.array(embedding_matrix)
        elif self.weight_mode == 'init_zeros_and_single_one_punished':
            for char, i in self.tk.word_index.items():  # from index 1 to 69
                onehot = np.zeros(self.embedding_dim)
                index = (i - 1) % self.embedding_dim
                if char == self.tk.oov_token:
                    onehot[index] = -1
                else:
                    onehot[index] = 1
                embedding_matrix.append(onehot)

            embedding_matrix = np.array(embedding_matrix)

        return embedding_matrix

    def Fit(self, train_data, train_labels):
        """
        Fits the transformer by using given training data and labels. Consists of selecting features, initializing weights.
        Once you fit your data into this transformer, you will be able to transform any input data
        @param train_data:
        @param train_labels:
        @return:
        """
        selected_ngrams, selected_ngram_scores, idf_dict = self.__ngrams_selection(train_data, train_labels,
                                                                                   self.max_df,
                                                                                   self.min_df, self.ngram_range,
                                                                                   self.max_num_features)
        self.flag_fit = True

        self.ngram_dict = {}
        for i, char in enumerate(selected_ngrams):
            self.ngram_dict[char] = i + 1

        self.tk.fit_on_texts(train_data)

        # -----------------------Skip part start--------------------------
        # construct a new vocabulary
        self.embed_dict = {}
        for i, char in enumerate(selected_ngrams):
            self.embed_dict[char] = i + 1

        # Use char_dict to replace the tk.word_index
        self.tk.word_index = self.embed_dict.copy()
        # Add 'UNK' to the vocabulary
        self.tk.word_index[self.tk.oov_token] = max(self.embed_dict.values()) + 1

        weight_matrix = self.__create_weight_matrix(selected_ngram_scores)

        vocab_size = len(self.tk.word_index)

        return selected_ngrams, selected_ngram_scores, weight_matrix, vocab_size, idf_dict

    # Convert string to index
    def __texts_to_ngram_sequences(self, texts, ngram_width=4):
        """
        Converts given list of text to a list of ngram sequences in which every item/ngram in that sequence actually
        represented by its vocabulary index @param texts: @param ngram_dict: @param ngram_width: @return:
        @param texts:  input documents
        @param ngram_width: ngram value
        @return:
        """
        from nltk import ngrams
        result = []
        missing_ngram_index = self.tk.word_index[self.tk.oov_token]
        print("For n  = ", ngram_width, " the index of oov token is  = ", missing_ngram_index, " vocab size = ",
              len(self.tk.word_index), " max #features = ", self.max_num_features)
        for text in texts:
            sequence = []
            ngram_list = ngrams(text, ngram_width)
            ngram_list = [''.join(ngram_tuple) for ngram_tuple in ngram_list]
            sequence = [self.embed_dict.get(k, missing_ngram_index) for k in
                        ngram_list]

            if sequence is None or len(sequence) == 0:
                print("Something went wrong...")

            if (max(sequence) > missing_ngram_index):
                print("Something went wrong... n = ", self.ngram_range)

            result.append(sequence)

        return result

    # This function is not used. But for later researches you might consider to use it. Becuase it replace the unknown characters with the most similar ones.
    def Try2FindSimilarNGramIndex(self, lookup_ngram):
        result_index = self.tk.word_index[self.tk.oov_token]
        if self.model_ngram_relations is None:  # If user is not intended to initialize and use Context Model (Word2Vec or FastText then directly return index of oov_token
            result_index = self.tk.word_index[self.tk.oov_token]
        else:  # If Context Model will be used
            # First check we have given lookup_ngram whether presented or not in Context Model. If it is not then directly return index of oov token
            if lookup_ngram not in self.model_ngram_relations.wv:
                # print("The unknown ngram ", lookup_ngram, " is not represented in Context Model. The index is set to oov_token index")
                return result_index
            try:
                # If it is represented in Context Model then find the similar ngrams from Context Model
                similar_ngrams = self.model_ngram_relations.wv.most_similar(lookup_ngram, topn=1)
                # Now we will try to find it in our embedding dictionary to get index, if we are still not able to find
                # it then again it will return index of oov token
                for similar_ngram, similar_ngram_prob in similar_ngrams:
                    if similar_ngram not in self.embed_dict:
                        continue
                    else:
                        # print("The unknown ngram ", lookup_ngram,
                        #       " will be represented by ", similar_ngram , " with probability " , similar_ngram_prob)
                        result_index = self.embed_dict[similar_ngram]
                        break
            except:
                result_index = self.tk.word_index[self.tk.oov_token]

        return result_index

    def Transform(self, data):
        """
        Transform given data into ngram sequences where the sequence will be truncated or padded by using max_seq_length
        @param data:
        @return:
        """
        if not self.flag_fit:
            raise AttributeError('You need to Fit your data first before calling Transform method')

        signalized_sequences = self.__texts_to_ngram_sequences(data, self.ngram_value)
        padded_signalized_sequences = pad_sequences(signalized_sequences, maxlen=self.max_document_length,
                                                    padding='post')
        return padded_signalized_sequences


class CharacterLevelTransformer:
    def __init__(self, max_document_length, embedding_dim=69, case_insensitive=False):
        self.max_document_length = max_document_length
        self.embedding_dim = embedding_dim
        self.case_insensitive = case_insensitive
        self.char_dict = {}
        if self.case_insensitive:
            warnings.warn("Case insensitivity is activated. Now every single text will be converted to lower-case "
                          "before finding the ngrams ")

        if case_insensitive:
            self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        else:
            self.alphabet = "abcdefghijklmnopqrstuvwxyz" \
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ " \
                            "0123456789" \
                            ",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

        self.vocab_size = len(self.alphabet) + 1
        self.__construct_vocabulary__()
        # Tokenizer
        self.tk = Tokenizer(num_words=None, char_level=True, oov_token='cs.hacettepe.edu.tr',
                            lower=self.case_insensitive)
        # Embedding Weights
        self.embedding_matrix = []

    def __construct_vocabulary__(self):
        for i, char in enumerate(self.alphabet):
            self.char_dict[char] = i + 1

    def __init_embedding_weights(self):
        self.embedding_matrix.append(np.zeros(self.embedding_dim))
        for char, i in self.tk.word_index.items():  # from index 1 to 69
            onehot = np.zeros(self.embedding_dim)
            index = (i - 1) % self.embedding_dim
            onehot[index] = 1
            self.embedding_matrix.append(onehot)

        self.embedding_matrix = np.array(self.embedding_matrix)

    def Fit(self):
        # Use char_dict to replace the tk.word_index
        self.tk.word_index = self.char_dict.copy()
        # Add 'UNK' to the vocabulary
        self.tk.word_index[self.tk.oov_token] = max(self.char_dict.values()) + 1

        self.__init_embedding_weights()
        vocab_size = self.vocab_size + 1  # Plus OOV Character
        return vocab_size, self.embedding_matrix

    def Transform(self, data):
        # Convert string to index
        char_sequences = self.tk.texts_to_sequences(data)

        # Padding
        char_data = pad_sequences(char_sequences, maxlen=self.max_document_length, padding='post')

        # Convert to numpy array
        char_data = np.array(char_data, dtype='float32')

        return char_data


class WeightInitializer(Enum):
    randomly_initialize = 'randomly_initialize'
    init_by_indicies = 'init_by_indicies'
    init_by_ngram_scores = 'init_by_ngram_scores'
    init_by_ngram_scores_punished = 'init_by_ngram_scores_punished'
    init_zeros_and_single_one = 'init_zeros_and_single_one'
    init_zeros_and_single_one_punished = 'init_zeros_and_single_one_punished'
