from scipy.ndimage.measurements import label
from HogModel.Models import Classifier
from HogModel.utils import profiling
import matplotlib.pyplot as plt
from time import time
import numpy as np
import cv2


class Detector:

    def __init__(self, windowSize, steps=2, conf_thresh=0.55):
        self.nmsThreshold = conf_thresh
        self.classifier = Classifier()
        self.w, self.h = windowSize
        self.steps = steps

    # @profiling.timer
    def _getROIBoxes(self, X, visualize=False, fps=10, limits=None):
        x, y = 0, 0
        if limits:
            yStart, yEnd = limits
            y = yStart
        else:
            yStart, yEnd = None, None
        steps_per_width = int((X.shape[1] - self.w) / self.steps) + 1
        steps_per_height = int((X.shape[0] - self.h) / self.steps) + 1
        patches = []
        bBoxes = []
        for yi in range(1, steps_per_height+1):
            if yStart and y+self.h >= yEnd:
                break
            for xi in range(1, steps_per_width+1):
                bBox = np.array([x, y, self.w, self.h])
                patch = X[y: y + self.h, x:x + self.w]
                if visualize:
                    clone = X.copy()
                    cv2.rectangle(clone, (x, y), (x+self.w, y+self.h), (0, 255, 0), 3)
                    cv2.imshow("clone", clone)
                    cv2.waitKey(fps)
                patches.append(patch)
                bBoxes.append(bBox)
                x += self.steps
            x = 0
            y += self.steps
        patches = np.array(patches)
        bBoxes = np.array(bBoxes)
        return patches, bBoxes

    @staticmethod
    def get_height_limits(X):
        bottom_half_start = int(0.5 * X.shape[0])
        bottom_half_end = int(0.9 * X.shape[0])
        return bottom_half_start, bottom_half_end

    @staticmethod
    # @profiling.timer
    def drawBBoxes(X, bBoxes):
        img = X.copy()
        if len(bBoxes) > 0:
            for x, y, w, h, p in bBoxes:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(img, (x, y), (x+32, y+16), (0, 255, 0), -1)
                cv2.putText(img, f"{p:.2f}", (x + 2, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img

    def get_heat_map(self, img, bboxes, threshold, visualize=False):
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        for box in bboxes:
            x, y, w, h = box[:4].astype("int")
            heatmap[y:y+h, x:x+w] += 1
        heatmap[heatmap <= threshold] = 0
        locations, ncars = label(heatmap)
        for i in range(1, ncars+1):
            nonzero_vals = (locations == i).nonzero()
            nonzeroY = np.array(nonzero_vals[0])
            nonzeroX = np.array(nonzero_vals[1])
            top_left = (np.min(nonzeroX), np.min(nonzeroY))
            bottom_right = (np.max(nonzeroX), np.max(nonzeroY))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        print("detected cars in frame: ", ncars)
        if visualize:
            plt.imshow(heatmap)
            plt.show()
        return img, ncars

    @profiling.timer
    def detectMultiScale(self, X, levels=3, threshold=5, 
        scale=0.50, visualize=False, height_limits=None):
        """
        """
        steps = self.steps
        downSampled = X.copy()
        limits = None
        all_bBoxes = []
        all_windows = []
        for level in range(1, levels+1):
            if level == 1:
                upscale_factor = 1
            else:
                self.steps = max(1, self.steps//2)
                downSampled = cv2.pyrDown(downSampled)
                # downSampled = cv2.resize(downSampled, None, fx=scale, fy=scale)
                # downSampled = cv2.GaussianBlur(downSampled, (3, 3), 0)
                upscale_factor = (1/scale) * (level - 1)
            if height_limits:
                limits = Detector.get_height_limits(downSampled)
            windows, bBoxes = self._getROIBoxes(downSampled, limits=limits, visualize=False)
            if len(windows) < 1:
                break
            print("level: ", level, "sride: ", self.steps)
            print("scaned img size: ", downSampled.shape)
            print("scaned windows dims: ", windows.shape)
            print("-"*25)
            if len(bBoxes) > 0:
                bBoxes = bBoxes * upscale_factor
                all_bBoxes.append(bBoxes)
                all_windows.append(windows)
        self.steps = steps
        all_bBoxes = np.concatenate(all_bBoxes)
        all_windows = np.concatenate(all_windows)
        predictions = self.classifier(all_windows)
        all_bBoxes = np.hstack([all_bBoxes, predictions[:, 1].reshape(-1, 1)])
        filteredBoxes = all_bBoxes[all_bBoxes[:, -1] > self.nmsThreshold]
        detectedImg, _ = self.get_heat_map(X, filteredBoxes, threshold, visualize=True)
        if visualize:
            # detectedImg = self.drawBBoxes(X, filteredBoxes)
            return detectedImg
        return filteredBoxes
