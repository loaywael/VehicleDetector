# VehicleDetector
  Detecting urban vehicles in images and videos using Hog Detector

- ## Project Goal   
    > Applying what I have learned in classical computer vision to build something useful   
    > that let me reverse engineer object detection algorithms.   
 
- ## Project Objectives:
	- [x] Feature Extraction Pipeline: {Hog Descriptor, Colors Histogram, and Edges}
	- [x] Sliding Window: Extracting windows of 64x64x3 to be classified
	- [x] Linear Classifier: Training SGD over [KITTY](http://www.cvlibs.net/datasets/kitti/) & [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html)  datasets
	- [x] Image Pyramids: Scaling the image to detect size variant objects
	- [x]  Non Max Suppression: Removing low confidence score of detections
	- [x]  Heatmap Filtering: Grouping multiple detection and removing False Positives **`FP`**
	- [ ] Run in realtime for videos: Analyze more than 5 **`FPS`**


- ## Project Setup and Requirements   
  **`use python3.5`** or newer versions to install and run the package  
  ```bash
  $ git clone https://github.com/loaywael/VehicleDetector.git   
  $ pip3 install -r requirements.txt      
  ```   
  
  alternatively install dependencies in virtualenv `recommended`   
  ```bash
  $ pip3 install virtualenv   
  $ python3 -m virtualenv venv   
  $ pip3 install -r requirements.txt   
  ```   
  
  
- ## How to Use   
   `HogModel` is excutable package can be run given command arguments   
   
   supported arguments:   
	- **`path`**: the image/video to be analyzed   
	- **`threshold`** : heatmap reduces false positives and improves detection   
	- **`height_limits`**: limits the sliding window to scan street level cars or no bool      
   </br>   
   
   ```bash  
   $ cd ./src     
   $ python3 -m HogModel ../data/detection_data/test_data/test2.jpg 5 True     
   ```
    
   </br></br></br>  
   <h3 align=center>Detected Images 2, 4, 6</h3>
   <img src="/assets/test2_detected.png" alt="test2 2 cars should be detected">
   <table><tr>
   <td><img src="/assests/test4_detected.png" alt="test4 2 cars should be detected" style="width: 25%;"/></td>
   <td><img src="/assests/test6_detected.png" alt="test6 2 cars should be detected" style="width: 25%;"/></td>
   </tr></table>
   </br></br>
   
***

# REFERENCES
> Thanks for these references I was able to get over problems I've faced during implementation.   

	https://github.com/udacity/CarND-Vehicle-Detection
	https://www.cs.utoronto.ca/~fidler/slides/CSC420/lecture17.pdf
	https://class.inrialpes.fr/tutorials/triggs-icvss1.pdf
	https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
	https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	https://towardsdatascience.com/pedestrian-detection-using-non-maximum-suppression-b55b89cefc6
	https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
	https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
	https://joblib.readthedocs.io/en/latest/parallel.html

