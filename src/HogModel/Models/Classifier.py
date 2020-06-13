from sklearn import svm, metrics, linear_model
from sklearn import pipeline, preprocessing
from HogModel.Models import FeatureDescriptor
from HogModel.utils import profiling
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Classifier:
    """
    Linear RGB Image Multi-Object Classifier

    Attributes
    ----------
    save_path : str
        default directory for the class
    scaler : StandardScaler object
        normalize and standardize input batch
    descriptor : FeatureDescriptor
        custom preprocessor extracts features_vector for given input batch
    model : Pipeline object
        stacks preprocessors and classification process sequentially

    Methods
    -------
    save_model(fpath=None)
        saves the fitted model to the default path
    load_model(fpath=None)
        loads the fitted model from the default path
    __call__(X)
        calls the fitted model to predict the input image batch
    eval_model(X, Y)
        evaluates fitted model
    plot_predictions(X, Y=None, figSize=(7, 7), dpi=100, cells_per_col=4)
        plotes random small batch of images and visualize prediction results
    """
    def __init__(self, X=None, Y=None, save=False, model=None, **model_params):
        """
        Parameters
        ----------
        X : np.ndarray
            batch of images to train the classifer on
        Y : np.ndarray
            batch of labels associated with the image batch
        save : bool
            saves the fitted model
        model : sklearn supported classifier object
            externally fitted classifier
        model_params : dict
            optional parameters to feed the sklearn classifier with
        """
        self.save_path = "HogModel/saved_models/"
        self.scaler = preprocessing.StandardScaler()
        self.descriptor = FeatureDescriptor()
        self.model = model
        
        if not model:
            self.model = linear_model.LogisticRegression(**model_params)
            # self.model = svm.SVC(kernel="linear", probability=True, **model_params)
        else:
            self.model = model

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

    def save_model(self, fpath=None):
        """
        Parameters
        ----------
        fpath : str
            saving model to custom path
        """
        if not fpath:
            fpath = self.save_path+"trained_model.sav"
        with open(fpath, "wb") as wf:
            pickle.dump(self.model, wf)
        print(f"trained-model saved at: {fpath}")

    def load_model(self, fpath=None):
        """
        Parameters
        ----------
        fpath : str
            loading model from custom path otherwise default path
        """
        if not fpath:
            fpath = self.save_path+"trained_model.sav"
        try:
            with open(fpath, "rb") as rf:
                self.model = pickle.load(rf)
                print(f"trained-model loaded successfully: {fpath}")
                print("="*55)
        except Exception:
            pass

    @profiling.timer
    def __call__(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            batch of images to classify
        
        Returns
        -------
        Ypred : np.ndarray
            prediction score for each class
        """
        Ypred = self.model.predict_proba(X)
        return Ypred

    def eval_model(self, X, Y):
        """
        Parameters
        ----------
        X : np.ndarray
            batch of images to evaluate fitted model over
        Y : np.ndarray
            batch of labels to evaluate fitted model over
        """
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
    def plot_predictions(X, Y=None, figSize=(7, 7), dpi=100, cells_per_col=4):
        """
        Parameters
        ----------
        X : np.ndarray
            batch of images to classify and plot
        Y : np.ndarray
            batch of predicted labels
        figSize : tuple
            initialize the figure size
        dpi : int
            increase/decrease zoom of subplots
        cells_per_col : int
            max number of columns in the figure
        """
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

