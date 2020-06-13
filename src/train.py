from sklearn.model_selection import train_test_split
from HogModel.Models import Classifier
import numpy as np
import sys



def train(X, Y):
    imgBatch = X[:16]
    classifier = Classifier(X, Y, save=True)
    classifier.evalModel(X, Y)
    Classifier.plotPredictions(imgBatch, classifier(imgBatch))
    return


def eval_model(X, Y):
    imgBatch = X[:16]
    classifier = Classifier()
    classifier.evalModel(X, Y)
    Classifier.plotPredictions(imgBatch, classifier(imgBatch))
    return


def main(argv):
    mode = argv[1]
    DATA_PATH = "../data/classification_data/train_data/"
    dataset = np.load(DATA_PATH+"train_dataset.npy")
    labels = np.load(DATA_PATH+"train_labels.npy")
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset, labels, test_size=0.2)

    if mode == "eval":
        eval_model(Xtest, Ytest)
    else:
        train(Xtrain, Ytrain)


if __name__ == "__main__":
    main(sys.argv)    

