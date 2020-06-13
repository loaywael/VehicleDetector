# VehicleDetector
  Detecting urban vehicles in images and videos using Hog Detector

- ## Project Goal   
    > Applying what I have learned in classical computer vision to build something usefull   
    > that let me reverse engineer object detection algorithims.   
 
- ## Project Objectives:
	- [x] Feature Extraction Pipeline: {Hog Descriptor, ColorsHistogram, and Edges}
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
	- **`limits`**: limits the sliding window to scan street level cars or no bool      
   </br>   
   
   ```bash  
   $ cd ./src     
   $ python3 -m HogModel ../data/detection_data/test_data/test2.jpg 5 True     
   ```
   </br>   
   </br>   
   </br>   
   
   <h3 align=center>Detected image: test2.png</h3>
   <img src="/assets/test2_detected.png" alt="2 cars should be detected">
   </br>
   </br>
   
***
**`version`**: https://git-lfs.github.com/spec/v1   
**`oid`**: sha256:dbcf7a95d663d2cd7b9d38eb72f11cd29dc8bc71f94977cdaff445b83d118ad5   
**`size`**: 619   

