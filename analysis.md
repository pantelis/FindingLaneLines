# **Finding Lane Lines on the Road**

Pantelis Monogioudis

NOKIA

**Summary**

The goal of this project is to demonstrate a pipeline that detects road lanes from monocular video camera streams.

---


## Pipeline
The pipeline consists of 5 stages.

1. First, the images are converted to grayscale using the corresponding
   openCV function.
2. Then noise is spatially averaged out using a symmetric Gaussian
   kernel. The standard deviation of the Gaussian kernel in the x and y
   direction was kept the same in the initial experimentation.
3. A Canny edge detection algorithm was then used to processed the
   blurred image.
4. A probabilistic Hough transform processed all Canny-detected edge
   pixels in the quadrilateral region of interest.
5. The identified edge lines were then classified as belonging to a
   right or left lane. The classification was based on a simple slope
   criterion.
6. The points belonging to the classified to each lane where fit using a
   RANSAC regressor and the produced linear model was superposed on the
   image.

## Performance
The figures below demonstrate the end result superposing the identified
lane lines on still images:

![out_solidWhiteCurve](/test_images/out_solidWhiteCurve.jpg)

_Detecting lanes in an image containing a solid white curve._

![out_solidWhiteRight](/test_images/out_solidWhiteRight.jpg)

_Detecting lanes in an image containing a right solid white line._

![out_solidYellowCurve](/test_images/out_solidYellowCurve.jpg)

_Detecting lanes in an image containing a solid yellow curve._


![out_solidYellowCurve2](/test_images/out_solidYellowCurve2.jpg)

_Detecting lanes in an image containing another solid yellow curve._


The developed simple pipeline was also tested in video scenes with both
white and yellow lines.


[![white_lane_video](https://img.youtube.com/vi/pMvNcu8Pzt8/0.jpg)](https://www.youtube.com/watch?v=pMvNcu8Pzt8)

_Detecting lanes in a video containing a white solid lane._

[![yellow_lane_video](https://img.youtube.com/vi/3aiFcOuQkMM/0.jpg)](https://www.youtube.com/watch?v=3aiFcOuQkMM)

_Detecting lanes in a video containing a yellow left lane._


* Despite its simplicity, the pipeline performed well and was able to
  identify both white and yellow lanes.
* We noticed that in some specific scenes as other vehicles come close
  to the lanes, the RANSAC regressor will get confused and create
  spurious lines. This can be observed instantaneously on the yellow.mp4
  video. Effectively the outliers created by the other car can throw
  instantaneously the RANSAC regressor off.
* The pipeline, however, performed poorly on the challenge.mp4 video scene
  that contains shadows from trees, nearby to the left lane concrete
  media barriers and other objects that are picked up by
  the Canny detector and are not filtered adequately by the Hough
  transform.


## Possible improvements
1. There is far more work that is required to have a successful outcome
   in the challenging extra.mp4 video. The work is particularly
   challenging as we need labelled video datasets where
   models can be trained on. Without such datasets, parameter tuning was almost ad hoc
   (trial and error). One source of improvement is therefore the
   improvement of the test setup that will allow engineers to tune their
   algorithms.

2. Another improvement can be achieved by stabilization algorithms to the video itself.

3. Algorithmically, after a literature survey on the subject, we can
   summarise the following ideas that almost certainly lead to
   performance improvements: a. The Gaussian kernel applied during the
   blurring step can be asymmetric with the standard deviation in the
   y-direction larger than the x-direction. The reason is that we are
   trying to detect vertical lines whose length is larger than their
   width. b. The parameters of the RANSAC regressor can be tuned.
   Especially the parameter residual_threshold that can tune the maximum
   residual for a data sample to be classified as an inlier. This will
   probably eliminate the aforementioned in the performance section
   RANSAC issues with nearby objects. c. We can an Inverse Perspective
   Mapping (IPM) which generates a bird's eye view of the road. In this
   view, the nearby to the left yellow line objects or concrete medians
   can be filtered out much better by selecting an appropriate region of
   interest.
