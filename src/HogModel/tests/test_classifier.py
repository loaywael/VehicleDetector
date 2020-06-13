from HogModel.Models import Classifier
import matplotlib.pyplot as plt
from unittest import TestCase
import numpy as np 
import cv2


class TestClassifier(TestCase):
    
    def setUp(self):
        self.test_img = cv2.imread("../data/detection_data/test_data/test1.jpg")
        self.test_img = cv2.resize(self.test_img, (64, 64))
        self.test_Xbatch = np.array([self.test_img]*1111)
        self.test_Ybatch = np.random.randint(0, 2, (1111))
        self.classifier_obj = Classifier()

    def test_call(self):
        error_test_msg = "prediction shape missmatch"
        self.classifier_obj.load_model()
        preds = self.classifier_obj(self.test_img)
        self.assertEqual(preds.shape, (1, 2), msg=error_test_msg)
        preds = self.classifier_obj(self.test_Xbatch)
        self.assertEqual(preds.shape, (1111, 2), msg=error_test_msg)

