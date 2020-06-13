from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from HogModel.utils import profiling
import numpy as np
import pickle
import cv2


class FeatureDescriptor(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.scaler = preprocessing.StandardScaler()
        return self 

    @profiling.timer
    def transform(self, X, y=None):
        if (len(X.shape) == 3) and (X.shape[2] <= 3):
            return self._preprocessImg(X)
        else:
            try:    # will be threaded later
                return np.concatenate([self._preprocessImg(x) for x in X], axis=0)
            except Exception as e:
                print("ERROR MSG: ", e)
                print("Incorrect image shape should be (64, 64, 3), or (-1, 64, 64, 3)")

    @staticmethod
    # @profiling.timer
    def _preprocessImg(X):
        hls = cv2.cvtColor(X, cv2.COLOR_RGB2HLS)
        ycbcr = cv2.cvtColor(X, cv2.COLOR_RGB2YCrCb)
        gray = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 33, 75, L2gradient=True)

        edgesFeatures = FeatureDescriptor._getColorFeatures(edges)
        hlsHistFeatures = FeatureDescriptor._getColorHistFeatures(hls)
        ycbcrColorFeatures = FeatureDescriptor._getColorFeatures(ycbcr)
        hogFeatures = FeatureDescriptor._getHogFeatures(gray)

        featuresVector = np.concatenate([
        hogFeatures,
        edgesFeatures,
        hlsHistFeatures,
        ycbcrColorFeatures
        ])
        return featuresVector.reshape((1, -1))


    @staticmethod
    def _getHogFeatures(img, win_size=(64, 64), block_size=(16, 16),
            block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hogFeatures = hog.compute(img).squeeze()
        return hogFeatures

    @staticmethod
    # @profiling.timeit
    def _getColorFeatures(img, downSize=(32, 32)):
        colorFeatures = cv2.resize(img, downSize).ravel()
        return colorFeatures

    @staticmethod
    # @profiling.timer
    def _getColorHistFeatures(img):
        hists = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
        featuresHistogram = np.array(hists).ravel()
        return featuresHistogram

