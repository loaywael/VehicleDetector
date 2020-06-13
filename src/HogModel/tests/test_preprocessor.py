from HogModel.Models import FeatureDescriptor
import matplotlib.pyplot as plt
from unittest import TestCase
import numpy as np 
import cv2



class TestPreprocessor(TestCase):

    def setUp(self):
        self.test_img = cv2.imread("../data/detection_data/test_data/test1.jpg")
        self.test_img = cv2.resize(self.test_img, (64, 64))
        self.test_batch = np.array([self.test_img]*5)
        self.preprocessor_obj = FeatureDescriptor()

    def test_getHogFeatures(self):
        error_test_msg = "hog_features has incorrect shape"
        hog_features = self.preprocessor_obj._getHogFeatures(self.test_img)
        self.assertEqual(hog_features.shape, (1764,), msg=error_test_msg)

    def test_process_img(self):
        error_test_msg = "features_vector has incorrect shape"
        features = self.preprocessor_obj._preprocessImg(self.test_img)
        self.assertEqual(features.shape, (1, 1764), msg=error_test_msg)

    def test_transform(self):
        error_test_msg = "features_vector has incorrect shape"
        features = self.preprocessor_obj.transform(self.test_img)
        self.assertEqual(features.shape, (1, 1764), msg=error_test_msg)
        features = self.preprocessor_obj.transform(self.test_batch)
        self.assertEqual(features.shape, (5, 1764), msg=error_test_msg)

