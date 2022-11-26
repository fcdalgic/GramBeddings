import argparse
from DataUtils import add_bool_arg, DatasetOptions
from NGramSequenceTransformer import WeightInitializer
out_dir = 'outputs'


def get_multi_train_args():
    parser = argparse.ArgumentParser(
        """Extracting Top-K Selected NGrams according tp selected scoring method.""")

    parser.add_argument("-d", "--dataset", type=DatasetOptions, default=DatasetOptions.grambeddings,
                        choices=list(DatasetOptions), help="dataset name")
    parser.add_argument("-o", "--output", type=str, default=out_dir,
                        help="The output directory where scores will be stored")

    # Input ngram selections
    parser.add_argument("-n1", "--ngram_1", type=int, default=4, help="Ngram value of first   ngram embedding layer")
    parser.add_argument("-n2", "--ngram_2", type=int, default=5, help="Ngram value of second  ngram embedding layer")
    parser.add_argument("-n3", "--ngram_3", type=int, default=6, help="Ngram value of third   ngram embedding layer")
    # Feature Selection Parameters
    parser.add_argument("-maxf", "--max_features", type=int, default=160000, help="Maximum number of features")
    parser.add_argument("-madf", "--max_df", type=float, default=0.9, help="Embedding dimension for Embedding Layer")
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
    parser.add_argument("-e", "--epochs", type=int, default=3, help="number of epoch to train our model")

    parser.add_argument("-dp", "--save_deep_features", type=int, default=0,
                        help="Whether save or not logits. 0 False, True Otherwise")

    args = parser.parse_args()
    return args




# def get_single_train_args():
#     parser = argparse.ArgumentParser(
#         """Extracting Top-K Selected NGrams according tp selected scoring method.""")
#
#     parser.add_argument("-t", "--train", type=str, default=train_file, help="training file path")
#     parser.add_argument("-v", "--val", type=str, default=val_file, help="validation file path")
#     parser.add_argument("-o", "--output", type=str, default=out_dir, help="The output directory where scores will be stored")
#
#     # Input ngram selections
#     parser.add_argument("-n", "--ngram", type=int, default=1, help="Ngram value of first   ngram embedding layer")
#     # Feature Selection Parameters
#     parser.add_argument("-maxf", "--max_features", type=int, default=160000, help="Maximum number of features")
#     parser.add_argument("-madf", "--max_df", type=float, default=0.7, help="Embedding dimension for Embedding Layer")
#     parser.add_argument("-midf", "--min_df", type=float, default=1e-06, help="Embedding dimension for Embedding Layer")
#     parser.add_argument("-msl", "--max_seq_len", type=int, default=128, help="The maximum sequence length to trim our transformed sequences")
#
#     parser.add_argument("-ed", "--embed_dim", type=int, default=15, help="Embedding dimension for Embedding Layer")
#     parser.add_argument("-aw", "--attn_width", type=int, default=10, help="The attention layer width")
#     parser.add_argument("-rnn", "--rnn_cell_size", type=int, default=128, help="The recurrent size")
#     parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
#     parser.add_argument("-e", "--epochs", type=int, default=9, help="number of epoch to train our model")
#
#     parser.add_argument("-dp", "--save_deep_features", type=int, default=0, help="Whether save or not logits. 0 False, True Otherwise")
#
#     args = parser.parse_args()
#     return args
