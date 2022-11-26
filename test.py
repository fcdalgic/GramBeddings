import numpy as np
import csv
import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from Utils.DataUtils import GetDataAndLabelsFromFiles, CreateModelFileNameFromArgs, add_bool_arg, DatasetOptions
from Model import ZhangAttention
from NGramSequenceTransformer import NBeddingTransformer, CharacterLevelTransformer, WeightInitializer
from tensorflow.keras.optimizers import Adam

train_file = 'data/train.csv'
val_file = 'data/test.csv'
out_dir = 'outputs'
CHAR_EMBEDDING_DIM = 69
loss = "binary_crossentropy"
optimizer = "Adam"
LABEL_PHISH = 1
LABEL_LEGIT = 0

PREDICT_BATCH_SIZE = 40000
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def get_args():
    parser = argparse.ArgumentParser(
        """Extracting Top-K Selected NGrams according tp selected scoring method.""")

    parser.add_argument("-d", "--dataset", type=DatasetOptions, default=DatasetOptions.grambeddings,
                        choices=list(DatasetOptions), help="dataset name")
    parser.add_argument("-ad", "--dataset_adv", type=DatasetOptions, default=DatasetOptions.grambeddings_adv,
                        choices=list(DatasetOptions), help="adversarial dataset name")
    parser.add_argument("-o", "--output", type=str, default=out_dir,
                        help="The output directory where scores will be stored")

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
    # Build and compile model
    model_name = CreateModelFileNameFromArgs(opt=args)
    file_name = 'outputs/training/models/' + model_name + '.h5'
    # file_name = 'outputs/training/models/best_model'
    model = tf.keras.models.load_model(file_name, custom_objects={'ZhangAttention': ZhangAttention})
    # model.trainable = False

    ####################################### Loading Dataset  #######################################
    print('####################################### Loading Dataset  #######################################')
    feature_selection_file = 'data/' + args.dataset.value + '/train.csv'
    test_data_file = 'data/' + args.dataset_adv.value + '/test_aug_mode2.csv'
    fs_samples, fs_labels   = GetDataAndLabelsFromFiles(feature_selection_file)
    adv_samples, adv_labels = GetDataAndLabelsFromFiles(test_data_file)

    import random
    c = list(zip(adv_samples, adv_labels))
    random.shuffle(c)
    adv_samples, adv_labels = zip(*c)
    adv_samples = list(adv_samples)
    adv_labels = np.array(adv_labels)
    print('Completed')

    ################################ Character Level Transformation ################################
    print('################################ Character Level Transformation ################################')
    transformer_char = CharacterLevelTransformer(args.max_seq_len, embedding_dim=CHAR_EMBEDDING_DIM,
                                                 case_insensitive=args.case_insensitive)
    char_vocab_size, char_embedding_matrix = transformer_char.Fit()
    test_sequences_char = transformer_char.Transform(adv_samples)
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
        fs_samples, fs_labels)
    print("Starting convert train texts to train sequences for n = ", args.ngram_1)
    test_sequences_1 = transformer_1.Transform(adv_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_1)
    test_sequences_1 = np.array(test_sequences_1, dtype='float32')
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
        fs_samples, fs_labels)
    print("Starting convert train texts to train sequences for n = ", args.ngram_2)
    test_sequences_2 = transformer_2.Transform(adv_samples)
    print("Starting convert validation texts to validation sequences for n = ", args.ngram_2)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_2)
    test_sequences_2 = np.array(test_sequences_2, dtype='float32')
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
        fs_samples, fs_labels)
    print("Starting convert train texts to train sequences for n = ", args.ngram_3)
    test_sequences_3 = transformer_3.Transform(adv_samples)
    print("Reshaping transformed inputs to arrange sizes before using them in Deep Learning Model  for n = ",
          args.ngram_3)
    test_sequences_3 = np.array(test_sequences_3, dtype='float32')
    print('Completed')


    # model.trainable=False
    print("asd")



    # predict_labels = model.predict([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3], verbose = 1)
    # metric_acc = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    # metric_pre = tf.keras.metrics.Precision(name='precision')
    # metric_rcl = tf.keras.metrics.Recall(name='recall')
    # metric_auc = tf.keras.metrics.AUC(name='auc')
    #
    # metric_acc.update_state(test_labels, predict_labels)
    # metric_pre.update_state(test_labels, predict_labels)
    # metric_rcl.update_state(test_labels, predict_labels)
    # metric_auc.update_state(test_labels, predict_labels)

    model.trainable = False
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

    adv_labels = adv_labels.reshape(-1, 1)
    results = model.evaluate([test_sequences_char, test_sequences_1, test_sequences_2, test_sequences_3], adv_labels, batch_size=128)
    print(results)


def get_every_n(a, n=PREDICT_BATCH_SIZE):
    for i in range(a.shape[0] // n):
        yield a[n*i:n*(i+1)]


def GetSpecifiedLayerOutputByName(layer_name, model : tensorflow.keras.models.Model):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer.output


if __name__ == "__main__":
    opt = get_args()
    Process(opt)

