from Utils.DimensionReduction import DimensionReducer
import csv
import numpy as np

file_path = '../outputs/features/model_features_train.csv'
output_file_name = "best_model_training"
def ReadDataAndLabels(file_path):
    xtrain = list()
    ytrain = list()

    with open(file_path, encoding="utf8", errors='ignore') as fp:
        line = fp.readline()
        while line:
            if line.__contains__('\n'):
                line = line.replace('\n', '')

            parts = line.split('\t')
            label = parts[len(parts) - 1]
            weights = parts[1:len(parts) - 1]

            if label == 'highly phish':
                label = 0
            elif label == 'medium phish':
                label = 1
            elif label == 'not significant':
                label = 5
            elif label == 'medium legit':
                label = 6
            elif label == 'highly legit':
                label = 7
            else:
                label = 2

            xtrain.append(weights)
            ytrain.append(label)

            line = fp.readline()

    return xtrain, ytrain

def ReadDataAndLabelsFromCSV(file_path):
    xtrain = list()
    ytrain = list()

    with open(file_path, 'r') as read_file:
        data_reader = csv.reader(read_file)
        for line in data_reader:
            label = line[len(line) - 1]
            weights = line[1:len(line) - 1]

            if label == 'highly phish':
                label = 0
            elif label == 'medium phish':
                label = 1
            elif label == 'not significant':
                label = 5
            elif label == 'medium legit':
                label = 6
            elif label == 'highly legit':
                label = 7
            else:
                label = 2

            xtrain.append(weights)
            ytrain.append(label)

    return xtrain, ytrain


def ReadDataAndLabelsFromFeaturesCSV(file_path):
    xtrain = list()
    ytrain = list()

    with open(file_path, 'r') as read_file:
        data_reader = csv.reader(read_file)
        for line in data_reader:
            label = line[len(line) - 1]
            weights = line[0:len(line) - 1]

            label = int(float(label))
            xtrain.append(weights)
            ytrain.append(label)

    return xtrain, ytrain



print("Loading dataset....")
from pandas import read_csv
df = read_csv(file_path)
print("Loading dataset....")

yTrain = df.values[:,df.values.shape[1]-1]
xTrain = df.values[:,0:df.values.shape[1]-1]
print("Dataset is loaded.")

print("Converting input embedding list into numpy array")
xTrain = np.array(xTrain, dtype='float32')
yTrain = np.array(yTrain, dtype=np.int)
yTrain = np.where(yTrain == 0, -1, 1)

print("2D Numpy array is created")

print("Initializing Dimension Reducer")
reducer = DimensionReducer(xTrain, yTrain)


output_dir = "../outputs/features/"

save_filepath = output_dir + output_file_name + ".png"
print("Training and fitting reducer")
reducer.doTrainFit(do_visualisation=True, supervision=False,
                   title='UMap Projection of Best Model Training Features',
                   save_filename=save_filepath,
                   )
