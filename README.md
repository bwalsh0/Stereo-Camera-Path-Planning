# Stereo Camera Path-Planning
Receives a stream of image pairs from two identical, calibrated cameras to compute a non-obstructed route.

#### Algorithm Procedure:
The input images are first rectified from camera calibration parameters before using OpenCV's StereoSGBM algorithm to compute the disparity map. The disparity map is then reprojected to a 3D space in the world coordinate system for obstacle filtering. Obstacles located above the floor plane are used to construct an occupancy grid. The A* pathfinding algorithm operates on the occupancy grid, with the end node determined by the furthest depth value with the shortest Euclidian distance from unified camera's principal point. The computed A* path is then back-projected to 2D image coordinates and overlaid on the unrectified left camera image.

## Results:
#### Video demonstration: 
https://youtu.be/m3qOAA8Q0Sc


#### Screenshots:
![Fig. 1](example_results/figure_01.png?raw=true)
![Fig. 2](example_results/figure_02.png?raw=true)


## Core Dependencies:
* [Python 3.9.1](https://www.python.org/)
* [OpenCV-contrib Python Bindings](https://github.com/opencv/opencv_contrib/tree/master)
* [Pathfinding Algorithm Implementations](https://github.com/brean/python-pathfinding)
