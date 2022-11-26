import argparse
import os

import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Utils.DataUtils import GetDataAndLabelsFromFiles, DatasetOptions, check_dir
from tld import get_tld
import random
import csv

LABEL_PHISH = 1
LABEL_LEGIT = -1
LABEL_ADVE = 9

CHAR_PHISH = 'P'
CHAR_LEGIT = 'L'
CHAR_ADVER = 'A'

full_data_path = '../data/grambeddings_Adversarial/test_aug_mode2.csv'
adve_data_path = '../data/grambeddings_Adversarial/TEL_converted.csv'

full_samples, full_labels = GetDataAndLabelsFromFiles(full_data_path, convert_to_array=False, label_legit=LABEL_LEGIT,
                                                      label_phish=LABEL_PHISH)
adve_samples, adve_labels = GetDataAndLabelsFromFiles(adve_data_path, convert_to_array=False, label_legit=LABEL_LEGIT,
                                                      label_phish=LABEL_PHISH, split_char=',')

# for index, sample in enumerate(full_samples):
#     if sample in adve_samples:
#         full_labels[index] = LABEL_ADVE
#
#     if index % 1000 == 0:
#         print("Progress :" , index , " - ", len(full_labels))
#

full_labels = full_labels[:-80000] + [LABEL_ADVE] * 80000
output_path = full_data_path.replace('.csv', '_adv_labeled.csv')
with open(output_path, 'w', encoding='utf-8', newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    for label, url in zip(full_labels, full_samples):
        line = [label, url]
        writer.writerow(line)
