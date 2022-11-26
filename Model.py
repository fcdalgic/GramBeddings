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


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.activations import *


class ZhangAttention(Layer):
    def __init__(self, units):
        super(ZhangAttention, self).__init__()
        self.units = units
        self.W1 = Dense(self.units)
        self.W2 = Dense(self.units)
        self.V = Dense(1)
        self._config = {
            'units': self.units,
            # 'W1':self.W1,
            # 'W2': self.W2,
            # 'V': self.V,
        }

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        return self._config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class NBeddingModel:
    def __init__(self, vocab_size, embedding_dim,  max_seq_length, rnn_cell_size = 256, attention_width = 10, warm_start = False, embedding_matrix =None):
        """
        The constructor of EmbeddingModel of both character level and ngram level representations

        @param vocab_size: Number of letters or ngrams will be used to initialize Embedding Layer which includes the OOV Token
        @param embedding_dim: The Embedding Dimension will be used to initialize Embedding Layer's output_dim
        @param max_seq_length: The maximum sequence length will be used to initialize Embedding Layer's input_length
        @param rnn_cell_size: The number of units will be used in LSTM layer.
        @param attention_width: Attention width of the Attention Layer
        @param warm_start: True if you desire to set initial weights for embedding layer.
        @param embedding_matrix: Initial weights for embedding layer
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.embedding_matrix = embedding_matrix
        self.warm_start = warm_start
        self.rnn_cell_size = rnn_cell_size
        self.attention_width = attention_width

        if self.warm_start and embedding_matrix is None:
            raise AttributeError('You need to set your embedding matrix weights in order to use Warm Start')

        self._config = {
            'vocab_size': self.vocab_size,
            'embedding_dim':self.embedding_dim,
            'max_seq_length':self.max_seq_length,
            'rnn_cell_size':self.rnn_cell_size,
            'attention_width':self.attention_width,
            'warm_start':self.warm_start,
            'embedding_matrix':self.embedding_matrix
        }

    def CreateModel(self, embedding_layer_name=None):
        """
        Creates a network model by using constructor parameters
        @param embedding_layer_name: None if you want to use default naming, sets the embedding layer name otherwise
        @return: First argument is the last layer of network model and the letter one is the input layer
        """
        embedding_layer = Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=self.max_seq_length,
            weights= [self.embedding_matrix] if self.warm_start else None,
            mask_zero=True,
            name=embedding_layer_name,
        )

        # Arranging Input of Model
        input_layer = Input(shape=(self.max_seq_length,), dtype='float32')
        # Embedding Layer
        x = embedding_layer(input_layer)

        x = Conv1D(filters=64, kernel_size=9, strides=1, activation='relu')(x)
        x = GlobalMaxPooling1D(data_format = "channels_first")(x)
        # x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=9, strides=1, activation='relu')(x)
        x = BatchNormalization()(x)
        last_layer = SpatialDropout1D(rate=0.2)(x)
        (lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(
            LSTM(self.rnn_cell_size, return_sequences=True, return_state=True))(last_layer)

        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        context_vector, attention_weights = ZhangAttention(self.attention_width)(lstm, state_h)

        return context_vector, input_layer

    def CreateLecunModel(self, embedding_layer_name=None):
        embedding_layer = Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=self.max_seq_length,
            weights=[self.embedding_matrix],
            mask_zero=True,
            name=embedding_layer_name
        )

        # Arranging Input of Model
        input_layer = Input(shape=(self.max_seq_length,), dtype='float32')
        # Embedding Layer
        x = embedding_layer(input_layer)

        x = Conv1D(filters=256, kernel_size=7, activation='relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(filters=256, kernel_size=7, activation='relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
        x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
        x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
        x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(3)(x)
        x = Flatten()(x)

        return x, input_layer

    def get_config(self):
        return self._config
