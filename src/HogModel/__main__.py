from HogModel.Models import Detector
import matplotlib.pyplot as plt
import time
import cv2
import sys
import os


def main(arg_vars):
    if len(arg_vars) == 2:
        media_path = arg_vars[1]
        threshold = 3
        height_limits = None
    elif len(arg_vars) == 3:
        media_path, threshold = arg_vars[1:3]
        threshold = eval(threshold)
        height_limits = None
    elif len(arg_vars) == 4:
        media_path, threshold, height_limits = arg_vars[1:4]
        threshold = eval(threshold)
        height_limits = eval(height_limits)
    else:
        print("ERROR: invalid argument list")
        print("|-----> please enter image, video path with supported arguments")
        return

    print("media", media_path)
    media_extension = os.path.splitext(media_path)[-1][1:]
    print("extension: ", media_extension)
    supported_imgs = ["jpg", "png", "jpeg"]
    supported_videos = ["mp4"]
    detector = Detector((64, 64), 16, conf_thresh=0.70)

    if media_extension in supported_imgs:
        img = plt.imread(media_path)
        # img = cv2.resize(img, None, fx=0.50, fy=0.50)
        detectedImg = detector.detectMultiScale(
            img, 6, visualize=True, 
            threshold=int(threshold),
            height_limits=height_limits
        )
        cv2.imshow("detection", detectedImg)
        cv2.waitKey(0)

    elif media_extension in supported_videos:
        cap = cv2.VideoCapture(media_path)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.50, fy=0.50)
            if ret:
                t1 = time.time()
                detectedImg = detector.detectMultiScale(
                    frame, 6, visualize=True, 
                    threshold=int(threshold),
                    height_limits=height_limits
                )
                t2 = time.time()
                cv2.putText(detectedImg, f"FPS: {int(1.0/(t2-t1))}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("detection", detectedImg)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    break

    else:
        print("ERROR: this file is not supported!")


if __name__ == "__main__":
    main(sys.argv)
    cv2.destroyAllWindows()
