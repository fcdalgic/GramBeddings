import argparse
from enum import Enum

from Utils.DataUtils import GetDataAndLabelsFromFiles
from Utils.DimensionReduction import DimensionReducer
import csv
import numpy as np
from pandas import read_csv

input_directory = '../outputs/features/'

output_file_name = "best_model_advTrain_advVal_augMode2_valid"

adversarial_training_file = '../data/grambeddings_Adversarial/train_aug_mode.csv'

adversarial_only_train_file = '../data/grambeddings_Adversarial/TRL_converted.csv'
adversarial_only_valid_file = '../data/grambeddings_Adversarial/TEL_converted.csv'

output_dir = "../outputs/features/"
augMode = 2


class enAdvMode(Enum):
    mode_1 = '../data/grambeddings_Adversarial/test_aug_mode1.csv'
    mode_2 = '../data/grambeddings_Adversarial/test_aug_mode2.csv'

class enTrainDataOption(Enum):
    including_adversarial     = '../data/grambeddings_Adversarial/train_aug_mode.csv',
    not_including_adversarial = '../data/grambeddings/train.csv'

class enDataDirectoryOptions(Enum):
    best_model = 'best_model'
    best_model_augmented_mode_1 = 'best_model_advTrain_advVal_augMode1'
    best_model_augmented_mode_2 = 'best_model_advTrain_advVal_augMode2'

def get_args():
    parser = argparse.ArgumentParser(
        """Extracting Top-K Selected NGrams according tp selected scoring method.""")

    parser.add_argument("-a", "--adv_mode", type=enAdvMode, default=enAdvMode.mode_2,
                        choices=list(enAdvMode), help="adversarial mode")

    parser.add_argument("-d", "--directory", type=enDataDirectoryOptions, default=enDataDirectoryOptions.best_model,
                        choices=list(enDataDirectoryOptions), help="Directory of where the feature files are stored")

    parser.add_argument("-t", "--training_data", type=enTrainDataOption, default=enTrainDataOption.including_adversarial,
                        choices=list(enTrainDataOption), help="The file path of training data")

    parser.add_argument("-e", "--epoch", type=int, default=200, help="The number of epoch in UMap")

    parser.add_argument("-n", "--neighbors", type=int, default=5, help="The n_neighbors in UMap")

    args = parser.parse_args()
    return args


def StartMapping(feature_file_path: str
                 , full_data_path: str
                 , adver_data_path: str
                 , pl_file_path: str
                 , pla_file_path: str
                 , epoch
                 , neighbors
                 ):
    print("Loading dataset....")
    df_train = read_csv(feature_file_path,  header=None)
    print("Done")

    print("Splitting panda frame into yTrain and xTrain")
    yTrain = df_train.values[:, df_train.values.shape[1] - 1]
    xTrain = df_train.values[:, 0:df_train.values.shape[1] - 1]
    xTrain = np.array(xTrain, dtype='float32')
    yTrain = np.array(yTrain, dtype=np.int)

    from copy import deepcopy
    yTrain_original = deepcopy(yTrain)
    yTrain_adversarial = deepcopy(yTrain)
    print("Done")

    print("###### Initializing the UMap Dimension Reducer for P-L Distribution ######")
    reducer = DimensionReducer(xTrain, yTrain, epoch=epoch, neighbors=neighbors)
    print("Done")

    LABEL_LEGIT = 0
    LABEL_PHISH = 1
    LABEL_ADVER = 2

    indexes_legit = [idx for idx, element in enumerate(yTrain_original) if element == LABEL_LEGIT]
    indexes_phish = [idx for idx, element in enumerate(yTrain_original) if element == LABEL_PHISH]
    indexes_adver = [idx for idx, element in enumerate(yTrain_original) if element == LABEL_ADVER]
    print("###### Class occurrences before indexing adversarial samples in yData ######")
    print('P: ', len(indexes_phish), '\t L : ', len(indexes_legit), '\t A : ', len(indexes_adver))

    print("###### Starting to Change label of Adversarial Sample Before Drawing the Distribution ######")
    print("###### Reading reference adversarial file and corresponding dataset file in order to index adversarial labels ######")
    full_samples, full_labels = GetDataAndLabelsFromFiles(full_data_path, convert_to_array=False,
                                                          label_legit=LABEL_LEGIT,
                                                          label_phish=LABEL_PHISH)
    adve_samples, adve_labels = GetDataAndLabelsFromFiles(adver_data_path, convert_to_array=False,
                                                          label_legit=LABEL_LEGIT,
                                                          label_phish=LABEL_PHISH, split_char=',')

    indexes_legit = [idx for idx, element in enumerate(full_labels) if element == LABEL_LEGIT]
    indexes_phish = [idx for idx, element in enumerate(full_labels) if element == LABEL_PHISH]
    indexes_adver = [idx for idx, element in enumerate(full_labels) if element == LABEL_ADVER]
    print("###### Class occurrences in full samples ######")
    print('Total : ', len(full_labels),  '\t P: ', len(indexes_phish), '\t L : ', len(indexes_legit), '\t A : ', len(indexes_adver))

    print("Done")
    print("### Starting to index labels with Adversarial index = 2 ######")
    data_len = len(yTrain_adversarial)
    for index, sample in enumerate(full_samples):
        if sample in adve_samples:
            if index < data_len:
                yTrain_adversarial[index] = LABEL_ADVER

        if index % 10000 == 0:
            print("Progress :", index, " - ", len(full_labels))

    print("Done")
    indexes_legit = [idx for idx, element in enumerate(yTrain_adversarial) if element == LABEL_LEGIT]
    indexes_phish = [idx for idx, element in enumerate(yTrain_adversarial) if element == LABEL_PHISH]
    indexes_adver = [idx for idx, element in enumerate(yTrain_adversarial) if element == LABEL_ADVER]
    print("###### Class occurrences after indexing adversarial samples in yData ######")
    print('P : ', len(indexes_phish), '\t L : ', len(indexes_legit), '\t A : ', len(indexes_adver))

    print("T###### Starting to Train and Fit on Reducer to draw distribution ######")

    reducer.doTrainFitBoth(
        xTrain=xTrain,
        yTrain_1=yTrain_original,
        save_filename_1=pl_file_path,
        yTrain_2=yTrain_adversarial,
        save_filename_2=pla_file_path
    )
    print("Done")


def Process(args):
    print(args)

    data_folder = input_directory + args.directory.value
    train_feature_file = data_folder + '/train.csv'
    valid_feature_file = data_folder + '/valid.csv'

    pl_file_path = data_folder + '/training_data_distribution_PL_epoch_' + str(args.epoch) + '_nn_' + str(args.neighbors) + '.png'
    pla_file_path = data_folder + '/training_data_distribution_PLA_epoch_' + str(args.epoch) + '_nn_' + str(args.neighbors) + '.png'
    # StartMapping(train_feature_file, args.training_data.value, adversarial_only_train_file, pl_file_path, pla_file_path)

    pl_file_path = data_folder + '/validation_data_distribution_PL_epoch_' + str(args.epoch) + '_nn_' + str(args.neighbors) + '.png'
    pla_file_path = data_folder + '/validation_data_distribution_PLA_epoch_' + str(args.epoch) + '_nn_' + str(args.neighbors) + '.png'
    StartMapping(valid_feature_file, args.adv_mode.value, adversarial_only_valid_file, pl_file_path, pla_file_path, args.epoch, args.neighbors)


if __name__ == "__main__":
    opt = get_args()
    Process(opt)
    #
    # opt.epoch = 50
    # opt.neighbors = 5
    # Process(opt)
    # opt.neighbors = 10
    # Process(opt)
    # opt.neighbors = 15
    # Process(opt)
    # opt.neighbors = 30
    # Process(opt)
    # opt.neighbors = 100
    # Process(opt)
    #
    # ######################
    # opt.epoch = 200
    # opt.neighbors = 5
    # Process(opt)
    # opt.neighbors = 10
    # Process(opt)
    # opt.neighbors = 15
    # Process(opt)
    # opt.neighbors = 30
    # Process(opt)
    # opt.neighbors = 100

