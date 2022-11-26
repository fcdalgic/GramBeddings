import numpy as np

def ReadPDRCNNDataset():
    legit_samples, legit_labels = ReadLegitimates()
    phish_samples, phish_labels = ReadPhishes()

    legit_samples = np.array(legit_samples)
    legit_labels = np.array(legit_labels)

    phish_samples = np.array(phish_samples)
    phish_labels = np.array(phish_labels)

    selection_amount = 10000

    legit_samples = np.random.choice(legit_samples, selection_amount)
    legit_labels = legit_labels[:10000]

    phish_samples = np.random.choice(phish_samples, selection_amount)
    phish_labels = phish_labels[:10000]

    phish_labels = list(phish_labels)
    legit_labels = list(legit_labels)
    phish_samples = list(phish_samples)
    legit_samples = list(legit_samples)

    total_samples = legit_samples + phish_samples
    total_labels = legit_labels + phish_labels

    total_output = list(zip(total_labels, total_samples))
    import random
    random.shuffle(total_output)

    train_amount = int(len(total_output) * 0.8)
    train_output = total_output[:train_amount]

    valid_amount = len(total_output) - train_amount
    valid_output = total_output[-valid_amount:]

    import csv
    with open('train.csv', 'w', encoding='utf-8') as csv_file:
        for line in train_output:
            row = '\t'.join(line)
            csv_file.write(row)

    with open('test.csv', 'w', encoding='utf-8') as csv_file:
        for line in valid_output:
            row = '\t'.join(line)
            csv_file.write(row)



def ReadLegitimates():
    file_path = 'Benign_list_big_final.csv'
    legitimates = list()
    labels = list()
    with open(file_path, encoding="utf8", errors='ignore') as fp:
        line = fp.readline()# To skip the header
        line = fp.readline()# Read the first data
        while line:
            if len(line) < 7:
                line = fp.readline()
                continue

            legitimates.append(line)
            labels.append('-1')
            line = fp.readline()

    return legitimates, labels

def ReadPhishes():
    file_path = 'phishing_dataset.csv'
    phishes = list()
    labels = list()
    with open(file_path, encoding="utf8", errors='ignore') as fp:
        line = fp.readline()
        while line:
            if len(line) < 7:
                line = fp.readline()
                continue
            phishes.append(line)
            labels.append('1')
            line = fp.readline()

    return phishes, labels


ReadPDRCNNDataset()