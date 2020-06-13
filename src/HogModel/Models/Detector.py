from scipy.ndimage.measurements import label
from HogModel.Models import Classifier
from HogModel.utils import profiling
import matplotlib.pyplot as plt
from time import time
import numpy as np
import cv2


class Detector:
    """
    HogDetector using sliding window and image pyramids

    Attributes
    ----------
    nmsThreshold : float
        non max threshold to ignore proposals with low confidence score
    classifier : Classifier object
        linear classifier to classify scaned windows
    w : int
        sliding window width
    h : int
        sliding window height
    steps: int
        stride of the sliding window

    Methods
    -------
    _get_roi_patches(X, visualize=False, delay=10, limits=None)
        applies sliding window for a given scale
    _get_height_limits(X)
        limits the sliding window to certain height from : to
    _drawBBoxes(X, bBoxes)
        draws bounding boxes and other infro {score, etc} to img
    _get_heat_map(self, img, bboxes, threshold, visualize=False)
        threshold multiple proposals to remove false positives
    detectMultiScale(self, X, levels=3, threshold=5, visualize=False, height_limits=None)
        detect objects at multi-scales using image pyramids
    """
    def __init__(self, windowSize, steps=16, conf_thresh=0.55):
        """
        Parameters
        ----------
        windowSize : tuple
            sliding window (width, height)
        steps : int
            sliding window stride (default is 16)
        conf_thresh : float
            non max proposal thresholding (default is 0.55)
        """
        self.nmsThreshold = conf_thresh
        self.classifier = Classifier()
        self.w, self.h = windowSize
        self.steps = steps

    # @profiling.timer
    def _get_roi_patches(self, X, visualize=False, delay=5, limits=None):
        """
        Parameters
        ----------
        X : np.ndarray
            img scale to scane with sliding window
        visualize : bool
            visualizing the scan process (default is False)
        delay : int
            controls the visualization scan speed (default is 5) 
        limits : tuple
            limits the sliding window scanning height (default is None)
        
        Returns
        -------
        patches : np.ndarray
            ROI patches >> (-1, 64, 64, -1)
        bBoxes : np.ndarray
            Bounding Boxes locations of all windows >> (-1, x, y, w, h)
        """
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
                    cv2.waitKey(delay)
                patches.append(patch)
                bBoxes.append(bBox)
                x += self.steps
            x = 0
            y += self.steps
            
        patches = np.array(patches)
        bBoxes = np.array(bBoxes)
        return patches, bBoxes

    @staticmethod
    def _get_height_limits(X):
        """
        Parameters
        ----------
        X : np.ndarray
            img scale to scane with sliding window
       
        Returns
        -------
        limits : tuple
            height limits for the given image scale
        """
        bottom_half_start = int(0.5 * X.shape[0])
        bottom_half_end = int(0.9 * X.shape[0])
        limits = bottom_half_start, bottom_half_end
        return limits

    @staticmethod
    # @profiling.timer
    def _drawBBoxes(X, bBoxes):
        """
        Parameters
        ----------
        X : np.ndarray
            output image with original shape
        bBoxes : np.ndarray
            propasal boxes locations >> (-1, x, y, w, h, p)

        Returns
        -------
        X : np.ndarray
            detected image with drawings of the proposal boxes
        """
        img = X.copy()
        if len(bBoxes) > 0:
            for x, y, w, h, p in bBoxes:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(img, (x, y), (x+32, y+16), (0, 255, 0), -1)
                cv2.putText(img, f"{p:.2f}", (x + 2, y+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img

    def _get_heat_map(self, img, bboxes, threshold, visualize=False):
        """
        Parameters
        ----------
        X : np.ndarray
            output image with original shape
        bboxes : np.ndarray
            filtered propasal boxes locations >> (-1, x, y, w, h, p)
        threshold : int
            unify proposal that hass above or equal occurences than threshold
            and removes others below it
        visualize : bool
            visualizing heatmap grouping result

        Returns
        -------
        X : np.ndarray
            detected image with drawings of the proposal boxes
        """
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
    def detectMultiScale(self, X, levels=6, threshold=5, 
        visualize=True, height_limits=None):
        """
        Parameters
        ----------
        X : np.ndarray
            img src to detect object in
        levels : int
            max number of pyramid levels to scan (default is 6)
        threshold : int
            unify proposal that hass above or equal occurences than threshold
            and removes others below it (default is 5)
        visualize : bool
            visualizing the detection as image or return proposal boxes (default is True)
        height_limits : bool
            limits the sliding window scanning height (default is None)
        
        Returns
        -------
        X : np.ndarray
            detected image with drawings of the proposal boxes
        or
        bBoxes : np.ndarray
            filtered proposal boxes >> (-1, x, y, w, h, p)
        """
        scale = 0.50
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
                limits = Detector._get_height_limits(downSampled)
            windows, bBoxes = self._get_roi_patches(downSampled, limits=limits, visualize=False)
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
        detectedImg, _ = self._get_heat_map(X, filteredBoxes, threshold, visualize=True)
        if visualize:
            # detectedImg = self._drawBBoxes(X, filteredBoxes)
            return detectedImg
        return filteredBoxes
