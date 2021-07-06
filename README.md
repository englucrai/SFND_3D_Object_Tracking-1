# SFND 3D Object Tracking

To accomplish this project it is necessary solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, to know how to detect objects in an image using the YOLO deep-learning framework. And finally, to know how to associate regions in a camera image with Lidar points in 3D space.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks:

The four principal tasks to this project are:
1. Develop a way to match 3D objects over time by using keypoint correspondences. 
2. Compute the TTC (time to colision) based on Lidar measurements. 
3. Associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. Conduct various tests with the framework. The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor.

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

---

## 1. Match 3D Objects

To accomplish successful 3D object matching, the **"matchingBoundingBoxes"** method was implemented. It takes as input both previous and current data frames and provides as output the matched regions of interests' ids. To choose the best matches it was performed a search for the highest key points correspondences.

---

## 2. Compute Lidar based TTC

Now we must compute the time to collision for all 3D objects based on Lidar measurements. It is important to filter the lidar data so the results may not be affected by outliers, preventing wrong estimates of time to colision.

---

## 3. Associate Keypoint Correspondences with Bounding Boxes

Before calculating the time to collision, it is necessary to find all key point matches, checking if the corresponding key points are within the region of interest in the camera image. To avoid outliers among the matches, a mean value is calculated using the Euclidean distance between keypoint matches. Those that are far from the mean value are then excluded from the results.

---

## 4. Compute Camera-based TTC

Once keypoint matches have been added to the bounding boxes, the next step is to compute the TTC estimate. The code is implemented in a way that it is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors.

