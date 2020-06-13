from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from HogModel.utils import profiling
import numpy as np
import pickle
import cv2


class FeatureDescriptor(BaseEstimator, TransformerMixin):
    """
    Custom Preprocessor Extracts Features Vector from Batch of Images

    Attributes
    ----------

    Methods
    -------
    fit(self, X, y=None)
    transform(self, X, y=None)
    _preprocess_img(X)
    _get_color_hist_features(img)
    _get_color_features(img, downSize=(32, 32))
    _get_hog_features(img, win_size=(64, 64), block_size=(16, 16),
        block_stride=(8, 8), cell_size=(8, 8), nbins=9)
    """
    def __init__(self):
        """
        """
        pass

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : np.ndarray
            batch of images to fit preprocessor over
        y : np.ndarray
            batch of labels associated with images batch

        Returns
        -------
        self : FeatureDescriptor object
            fitted preprocessor object
        """
        self.scaler = preprocessing.StandardScaler()
        return self 

    @profiling.timer
    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : np.ndarray
            batch of images to extract features from
        y : np.ndarray
            batch of labels associated with images batch

        Returns
        -------
        features : np.ndarray
            extracted features vector
        """
        if (len(X.shape) == 3) and (X.shape[2] <= 3):
            return self._preprocess_img(X)
        else:
            try:    # will be threaded later
                return np.concatenate([self._preprocess_img(x) for x in X], axis=0)
            except Exception as e:
                print("ERROR MSG: ", e)
                print("Incorrect image shape should be (64, 64, 3), or (-1, 64, 64, 3)")

    @staticmethod
    # @profiling.timer
    def _preprocess_img(X):
        """
        Parameters
        ----------
        X : np.ndarray
            rgb image to extract features vector from 

        Returns
        -------
        featuresVector : np.ndarray
            extracted features vector
        """
        hls = cv2.cvtColor(X, cv2.COLOR_RGB2HLS)
        ycbcr = cv2.cvtColor(X, cv2.COLOR_RGB2YCrCb)
        gray = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 33, 75, L2gradient=True)

        edgesFeatures = FeatureDescriptor._get_color_features(edges)
        hlsHistFeatures = FeatureDescriptor._get_color_hist_features(hls)
        ycbcrColorFeatures = FeatureDescriptor._get_color_features(ycbcr)
        hogFeatures = FeatureDescriptor._get_hog_features(gray)

        featuresVector = np.concatenate([
        hogFeatures,
        edgesFeatures,
        hlsHistFeatures,
        ycbcrColorFeatures
        ])
        return featuresVector.reshape((1, -1))


    @staticmethod
    def _get_hog_features(img, win_size=(64, 64), block_size=(16, 16),
            block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        """
        img : np.ndarray
            gray-scale image to extract hog features vector from 

        Returns
        -------
        hogFeatures : np.ndarray
            extracted hog features vector
        """
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hogFeatures = hog.compute(img).squeeze()
        return hogFeatures

    @staticmethod
    # @profiling.timeit
    def _get_color_features(img, downSize=(32, 32)):
        """
        img : np.ndarray
            rgb image to extract color features vector from 

        Returns
        -------
        colorFeatures : np.ndarray
            extracted color features vector
        """
        colorFeatures = cv2.resize(img, downSize).ravel()
        return colorFeatures

    @staticmethod
    # @profiling.timer
    def _get_color_hist_features(img):
        """
        img : np.ndarray
            rgb image to extract color histogram features vector from 

        Returns
        -------
        featuresHistogram : np.ndarray
            extracted color histogram features vector
        """
        hists = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
        featuresHistogram = np.array(hists).ravel()
        return featuresHistogram

