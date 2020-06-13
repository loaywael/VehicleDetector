import matplotlib.pyplot as plt 
import numpy as np
import math
import cv2



def visualize_all_scales(scale, levels=1):
    def detect_all_scales(func):
        def wrapper(self, *args, **kwargs):
            X = args[0]
            print("first image scale: ", X.shape)
            fig = plt.figure(figsize=(7, 7), dpi=150)
            scaled_img = X.copy()
            for level in range(levels):
                print("---> scale: ", level)
                if level > 0:
                    scaled_img = cv2.resize(scaled_img, None, fx=0.75, fy=0.75)
                    scaled_img = cv2.GaussianBlur(scaled_img, (3, 3), 0)
                axis = fig.add_subplot(math.ceil(levels/2), 2, level+1)
                axis.set_title(f"level: {level}, size: {scaled_img.shape}")
                height_limits = self.get_height_limits(scaled_img, level)
                detectec_img = func(self, scaled_img, visualize=True, height_limits=height_limits)
                axis.imshow(detectec_img)
                axis.set_axis_off()
                fig.tight_layout(pad=1.5)
            plt.show()
            return
        return wrapper
    return detect_all_scales

