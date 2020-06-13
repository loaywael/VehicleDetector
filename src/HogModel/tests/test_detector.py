from HogModel.Models import Detector
import matplotlib.pyplot as plt
from unittest import TestCase
import numpy as np 
import cv2


class TestDetector(TestCase):

    def setUp(self):
        self.test_boxes = np.array([
        [90, 75, 64, 64, 0.8],
        [0, 0, 64, 64, 0.33],
        [5, 7, 64, 64, 0.55],
        [10, 10, 64, 64, 0.67],
        [88, 69, 64, 64, 0.95],
        [200, 280, 64, 64, 0.79],
        [25, 40, 64, 64, 0.53],
        [15, 6, 128, 128, 0.61],
        [190, 273, 64, 64, 0.83] 
    ])
        self.test_img = cv2.imread("../data/detection_data/test_data/test1.jpg")
        self.test_img = cv2.resize(self.test_img, (1280, 720))
        self.detector_obj = Detector((64, 64), 32, conf_thresh=0.80, iou_thresh=0.50)
        self.search_limits = int(self.test_img.shape[0]/2), int(self.test_img.shape[0]/1.1)

    def test_get_roi_boxes(self):
        windows, bboxes = self.detector_obj._getROIBoxes(self.test_img, limits=self.search_limits)
        error_test_msg = "windows is not instance of numpy array"
        self.assertTrue(isinstance(windows, (np.ndarray)))
        error_test_msg = "bboxes is not instance of numpy array"
        self.assertTrue(isinstance(bboxes, (np.ndarray)))
        error_test_msg = "windows shape missmatch"
        self.assertEqual(windows.shape, (312, 64, 64, 3), msg=error_test_msg)
        error_test_msg = "bounding boxes shape missmatch"
        self.assertEqual(bboxes.shape, (312, 4), msg=error_test_msg)
        error_test_msg = "windows and bounding boxes has inconsistent length"
        self.assertEqual(len(windows), len(bboxes))

    def test_get_iou_score(self):
        error_test_msg = "IoU score is incorrect"
        gnd_truth = self.test_boxes[1][:4]
        pred_box = self.test_boxes[2][:4]
        iou_score = self.detector_obj._getIoUScore(gnd_truth, pred_box)
        self.assertAlmostEqual(iou_score, 0.70, msg=error_test_msg, places=3)

    def test_non_max_suppression(self):
        filtered_boxes = self.detector_obj._nonMaxSuppression(self.test_boxes)
        print(filtered_boxes)


    def test_draw_bboxes(self):
        error_test_msg = "detected image has incorrect shape given bounding boxes"
        detection_img = self.detector_obj.drawBBoxes(self.test_img, self.test_boxes[:3])
        self.assertEqual(detection_img.shape, self.test_img.shape, msg=error_test_msg)
        error_test_msg = "detected image has incorrect shape given ** NO ** bounding boxes"
        detection_img = self.detector_obj.drawBBoxes(self.test_img, [])
        self.assertEqual(detection_img.shape, self.test_img.shape, msg=error_test_msg)

    def test_call(self):
        bboxes = self.detector_obj(self.test_img)
        print(bboxes)
        error_test_msg = "bboxes is not instance of numpy array"
        self.assertTrue(isinstance(bboxes, (np.ndarray)), msg=error_test_msg)
        error_test_msg = "0/2 )---> detcted objects in the test img"
        self.assertGreaterEqual(len(bboxes), 1, msg=error_test_msg)
        error_test_msg = "1/2 )---> detcted objects in the test img"
        self.assertGreaterEqual(len(bboxes), 2, msg=error_test_msg)
        error_test_msg = "+2/2 )---> Detection includes FALSE POSITIVES"
        self.assertGreaterEqual(len(bboxes), 2, msg=error_test_msg)
        
        
        