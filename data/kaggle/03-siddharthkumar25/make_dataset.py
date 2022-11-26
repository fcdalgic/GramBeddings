
def ProcessDataset():
    total_samples, total_labels = ReadUrlAndLabels()

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
            row = '\t'.join(line) + '\n'
            csv_file.write(row)

    with open('test.csv', 'w', encoding='utf-8') as csv_file:
        for line in valid_output:
            row = '\t'.join(line) + '\n'
            csv_file.write(row)



def ReadUrlAndLabels():
    file_path = 'urldata.csv'
    urls = list()
    labels = list()
    with open(file_path, encoding="utf8", errors='ignore') as fp:
        line = fp.readline()# To skip the header
        line = fp.readline()# Read the first data
        while line:
            parts = line.split(',')
            url = ''.join(parts[1:-2])
            label = parts[len(parts) - 2]

            if len(url) < 7:
                line = fp.readline()
                continue

            urls.append(url)
            if label == 'benign':
                labels.append('-1')
            else:
                labels.append('1')

            line = fp.readline()

    return urls, labels

ProcessDataset();