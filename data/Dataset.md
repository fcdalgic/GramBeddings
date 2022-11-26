![GRAM](https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/assets/images/logo.png "GRAM")
# Dataset Structure
he dataset files are stored under their unique dataset names.  For each unique (e.g. Grambeddings and ebubekir) our program expects two files which are *test.csv* and *train.csv*.  In detail, the required data format of each csv file is LABEL \t URL where \t is the delimiter and LABELS are -1 for Legitimate and 1 for Phish samples. 

In this folder we represented whole dataset covered in out paper.

In order to benchmar our model with your existring project, it is better to add your train and test files into new folder under this scope. You can check [this](/PDRCNN/make_dataset.py)
 example of converting another dataset file into the expected file format 
