# SFND 3D Object Tracking

To accomplish this project it is necessary solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, to know how to detect objects in an image using the YOLO deep-learning framework. And finally, to know how to associate regions in a camera image with Lidar points in 3D space.

<p align="center">
  <img src="https://github.com/englucrai/SFND_3D_object_tracking/blob/master/images/diagram.png"/>
</p>

In this final project, the focus will be the final part showed above inside the blue block. The four principal tasks to this project are:

1. Develop a way to match 3D objects over time by using keypoint correspondences.

2. Compute the TTC (time to colision) based on Lidar measurements. 

3. Associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 

4. Conduct various tests with the framework. The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor.

<p align="center">
  <img src="https://github.com/englucrai/SFND_3D_object_tracking/blob/master/images/LidarClustering.gif">
</p>

<p align="center">
  <img src="https://github.com/englucrai/SFND_3D_object_tracking/blob/master/images/ClusteringLimitingToFirstBBox.gif">
</p>

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

```C++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, 
                        DataFrame &prevFrame, DataFrame &currFrame)
{ 
 int prevBoxSize = prevFrame.boundingBoxes.size();
    int currBoxSize = currFrame.boundingBoxes.size();
    int prevCurrBoxScore [prevBoxSize][currBoxSize] = {};

    // iterate point matches and count matching score
    for (auto it = matches.begin(); it != matches.end()-1; it++)
    {
        // previous point
        cv::KeyPoint prev_key_point = prevFrame.keypoints[it->queryIdx];
        cv::Point prevPoint = cv::Point(prev_key_point.pt.x, prev_key_point.pt.y);

        // current point
        cv::KeyPoint curr_key_point = currFrame.keypoints[it->trainIdx];
        cv::Point currPoint = cv::Point(curr_key_point.pt.x, curr_key_point.pt.y);

        // get corresponding box
        std::vector<int> prev_box_id_list, curr_box_id_list;
        for (int i = 0; i < prevBoxSize; i++)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(prevPoint))
            {
                prev_box_id_list.push_back(i);
            }
        }

        for(int j = 0; j < currBoxSize; j++)
        {
            if (currFrame.boundingBoxes[j].roi.contains(currPoint))
            {
                curr_box_id_list.push_back(j);
            }
        }

        // add count to box score
        for (int i:prev_box_id_list)
        {
            for (int j:curr_box_id_list)
            {   
                prevCurrBoxScore[i][j] += 1;
            }
        }
    }

    // for each box find the one with highest score 
    for (int i  = 0; i < prevBoxSize; i++)
    {
        int max_score = 0;
        int best_idx = 0;

        for (int j = 0; j < currBoxSize; j++)
        {
            if (prevCurrBoxScore[i][j] > max_score)
            {
                max_score = prevCurrBoxScore [i][j];
                best_idx = j;
            }
        }

        bbBestMatches[i] = best_idx;
    }
}
```

The **matchingBoundingBoxes** method receives as input the current and previous frame. A loop is created to go through the matches and to evalute the matching socre. First it is created an cv::Point for both frames containing the keypoints. Then we find the corresponding bounding box, count the score and pick the one with highest score (most matches).

## 2. Compute Lidar based TTC

Now we must compute the time to collision for all 3D objects based on Lidar measurements. It is important to filter the lidar data so the results may not be affected by outliers, preventing wrong estimates of time to colision. This is done by implementing the **computeTTCLidar** method.

```C++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // aux variables
    double dT = 1.0 / frameRate; // time between two measurements [s]

    // find closest distance to Lidar points
    double minXPrev = 1e9, minXCurr = 1e9;

    vector<double> prev_vector;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
    {
        prev_vector.push_back(it->x);
    }
    sort(prev_vector.begin(), prev_vector.end());
    minXPrev = prev_vector[prev_vector.size()*1/5];

    vector<double> curr_vector;
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
    {
        curr_vector.push_back(it->x);
    }
    sort(curr_vector.begin(), curr_vector.end());
    minXCurr = curr_vector[curr_vector.size()*1/5];

    // Compute TTC
    TTC = minXCurr* dT / (minXPrev-minXCurr);

    cout << "-------------------------------" << endl;
    cout << "LIDAR TTC" << endl;
    cout << "Lidar time to colision: " << TTC << " s" << endl;
    cout << "minXPrev: " << minXPrev << " m" << endl;
    cout << "minXCurr: " << minXCurr << " m" << endl;
    cout << "-------------------------------" << endl;
}
```

## 3. Associate Keypoint Correspondences with Bounding Boxes

Before calculating the time to collision, it is necessary to find all key point matches, checking if the corresponding key points are within the region of interest in the camera image. To avoid outliers among the matches, a mean value is calculated using the Euclidean distance between keypoint matches. Those that are far from the mean value are then excluded from the results. This is done by implementing the **clusterKptMatchesWithROI** method.

```C++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, 
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double distance_mean = 0.0;
    double size = 0.0;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        cv::KeyPoint prev_point = kptsPrev[it->queryIdx];
        cv::KeyPoint curr_point = kptsCurr[it->trainIdx];

        if (boundingBox.roi.contains(curr_point.pt))
        {
            distance_mean += cv::norm(curr_point.pt - prev_point.pt);
            size += 1;
        }
    }
    distance_mean = distance_mean / size;

    // Filtering
    for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        cv::KeyPoint prev_point = kptsPrev[it->queryIdx];
        cv::KeyPoint curr_point = kptsCurr[it->trainIdx];

        if(boundingBox.roi.contains(curr_point.pt))
        {
            double curr_distance = cv::norm(curr_point.pt - prev_point.pt);

            if (curr_distance < distance_mean * 1.3)
            {
                boundingBox.keypoints.push_back(curr_point);
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }
}
```

## 4. Compute Camera-based TTC

Once keypoint matches have been added to the bounding boxes, the next step is to compute the TTC estimate. The code is implemented in a way that it is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors. This is done by implementing the **computeTTCCamera** method.

```C++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
     // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);

    cout << "-------------------------------" << endl;
    cout << "CAMERA TTC" << endl;
    cout << "Camera time to colision: " << TTC << " s" << endl;
    //cout << "Dist curr: " << distCurr << " m" << endl;
    //cout << "Dist prev: " << distPrev << " m" << endl;
    cout << "-------------------------------" << endl;
}
```

## Performance evaluation

The goal now is to conduct tests with the final project code, especially with regard to the Lidar part. It was looked for several examples where the Lidar-based TTC estimate is way off the expected values. The, we will be running the different detector/descriptor combinations and looking at the differences in TTC estimation.

The detector and drescription combination used are dictated by the results of the Midterm Project (https://github.com/englucrai/feature_tracking);

So, the combination used for the test are:

1. FAST detector and BRIEF descriptor;

2. FAST detector and ORB descriptor;

3. FAST detector and BRISK descriptor.

It is importante to say that we expect better results from the FAST-BRIEF combination, since this was the best possible combination reagrading time of processing and best matches of the ego vehicle between the frames.

The following Figure shows the Lidar based time to collision. We can clearly see that there are some values with a lot of error, leading to inconsistent estimates. The most incorrect estimate occurs in frame 3.

<p align="center">
  <img src="https://github.com/englucrai/SFND_3D_object_tracking/blob/master/images/ttclidar.png"/>
</p>

The following Table provides the tests results in seconds [s]:

| Lidar ttc | Camera ttc  (FAST-BRIEF) | Camera ttc (FAST-ORB) | Camera ttc (FAST-BRISK) |
|-----------|--------------------------|-----------------------|-------------------------|
| 13,2518   | 10,8026                  | 11,0105               | 12,3379                 |
| 12,5206   | 11,0063                  | 10,7587               | 12,5319                 |
| 31,4519   | 14,1559                  | 11,4167               | 14,639                  |
| 14,4611   | 14,3886                  | 12,8498               | 12,6071                 |
| 10,175    | 19,9511                  | 17,8195               | 34,7028                 |
| 13,96     | 13,293                   | 12,9991               | 12,435                  |
| 11,3597   | 12,2182                  | 11,6025               | 18,9204                 |
| 14,8235   | 12,7596                  | 11,1687               | 11,3076                 |
| 13,1631   | 12,6                     | 12,1119               | 13,2023                 |
| 15,2123   | 13,4637                  | 13,3473               | 12,5274                 |
| 11,9226   | 13,6717                  | 13,778                | 14,2506                 |
| 9,6262    | 10,9087                  | 10,8955               | 11,4004                 |
| 8,9321    | 12,3705                  | 12,0411               | 12,2419                 |
| 9,54663   | 11,2431                  | 10,7303               | 12,1348                 |
| 7,68621   | 11,8747                  | 11,2058               | 12,0853                 |
| 9,2       | 11,8398                  | 11,1948               | 12,1722                 |
| 11,7508   | 7,9201                   | 7,869                 | 8,51608                 |
| 10,4045   | 11,554                   | 10,6099               | 11,5441                 |
 
And the following Figures shows the results displayed as a function of the number of frames used to calculate the TTC.

The FAST-BRIEF combination results in a good camera TTC result, since there are small error percentage and none of the values is way off. This happens because the keypoint detection and description succesfully focusing on keypoints belonging to the vehicle on the ego lane. The FAST-ORB combination also performed well.

<p align="center">
  <img src="https://github.com/englucrai/SFND_3D_object_tracking/blob/master/images/detfast_descbrief.png"/>
</p>

<p align="center">
  <img src="https://github.com/englucrai/SFND_3D_object_tracking/blob/master/images/detfast_descorb.png"/>
</p>

The FAST-BRISK combination results in values way off for the camera based time to collision, presenting a big error on the estimates. This happens because of the keypoint correspondence between frames, where some keypoints used on the calculation does not belong to the vehicle on the ego lane, leading to the inconsistent results.

<p align="center">
  <img src="https://github.com/englucrai/SFND_3D_object_tracking/blob/master/images/detfast_descbrisk.png"/>
</p>

