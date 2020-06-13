from sklearn import svm, metrics, linear_model
from sklearn import pipeline, preprocessing
from HogModel.Models import FeatureDescriptor
from HogModel.utils import profiling
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Classifier:
    def __init__(self, X=None, Y=None, save=False, model=None, **model_params):
        self.save_path = "HogModel/saved_models/"
        self.scaler = preprocessing.StandardScaler()
        self.descriptor = FeatureDescriptor()
        self.model = None
        self.model = model
        
        if not model:
            self.model = linear_model.LogisticRegression(**model_params)
            # self.model = svm.SVC(kernel="linear", probability=True, **model_params)

        clf_pipeline = pipeline.Pipeline([
            ("descriptor", self.descriptor),
            ("std_scaler", self.scaler),
            ("classifier", self.model)
        ])

        if isinstance(X, (np.ndarray, list)):
            self.model = clf_pipeline.fit(X, Y)
            if save:
                self.save_model()
        else:
            try:
                self.load_model()
            except Exception as e:
                print("ERROR MSG: ", e)
                print("No model fitted is saved for this input data")

    @profiling.timer
    def __call__(self, X):
        Ypred = self.model.predict_proba(X)
        return Ypred

    def save_model(self, fpath=None):
        if not fpath:
            fpath = self.save_path+"trained_model.sav"
        with open(fpath, "wb") as wf:
            pickle.dump(self.model, wf)
        print(f"trained-model saved at: {fpath}")

    def load_model(self, fpath=None):
        if not fpath:
            fpath = self.save_path+"trained_model.sav"
        try:
            with open(fpath, "rb") as rf:
                self.model = pickle.load(rf)
                print(f"trained-model loaded successfully: {fpath}")
                print("="*55)
        except Exception:
            pass

    def evalModel(self, X, Y):
        Ypred = self.model.predict(X)
        accuracy = metrics.accuracy_score(Y, Ypred)
        precision = metrics.precision_score(Y, Ypred)
        recall = metrics.recall_score(Y, Ypred)
        f1_score = metrics.f1_score(Y, Ypred)
        roc_curve = metrics.roc_curve(Y, Ypred)
        print("==================================")
        print(" |---> Model Testing Scores <---|")
        print("==================================")
        print("[-]---------> Accuracy: ", round(accuracy, 3))
        print("[-]---------> Precision: ", round(precision, 3))
        print("[-]---------> Recall: ", round(recall, 3))
        print("[-]---------> F1-Score: ", round(f1_score, 3))
        # plt.plot(roc_curve)
        # plt.show()

    @staticmethod
    def plotPredictions(X, Y=None, figSize=(7, 7), dpi=100, cells_per_col=4):
        m = len(X)
        fig = plt.figure(figsize=figSize, dpi=dpi)
        # fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        for i in range(m):
            class_score = Y[i].max()
            class_id = Y[i].argmax(axis=-1)
            class_name = "Car" if class_id == 1 else "BKG"
            axis = fig.add_subplot(round(m/cells_per_col), cells_per_col, i+1)
            axis.imshow(X[i])
            axis.set_title(class_name)
            # axis.set_axis_off()
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(f"score: {class_score:.2f}")
            axis.set_ylabel(f"class: {class_id}")
        fig.tight_layout(pad=1.5)
        plt.show()

