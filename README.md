# Stereo Camera Path-Planning

#### Description:
This program is for a stereo camera device, and enables real-time path planning and obstacle avoidance for robotic vehicles. The code implements OpenCV with Python for camera calibration, stereo correspondence, 3D reconstruction, and pathfinding.

#### Algorithm Procedure:
The input images are first rectified from camera calibration parameters before using OpenCV's StereoSGBM algorithm to compute the disparity map. The disparity map is then reprojected to a 3D space in the world coordinate system for obstacle filtering. Obstacles located above the floor plane are used to construct an occupancy grid. The A* pathfinding algorithm operates on the occupancy grid, with the end node determined by the furthest depth value with the shortest Euclidian distance from unified camera's principal point. The computed A* path is then back-projected to 2D image coordinates and overlaid on the unrectified left camera image.

## Results:
#### Video demonstration: 
https://youtu.be/Hd16ineBsT8


#### Screenshots:
![Fig. 0](example_results/figure_00.png?raw=true)

##### 3D Scene Projection with Planned Path:
![Fig. 4](example_results/figure_04.png?raw=true)

##### Image of Stereo Camera Device:
![Device Photo](example_results/device_photo.png?raw=true)

##### System Diagram:
![SBD](example_results/system_block_diagram.png?raw=true)


## Core Dependencies:
* [Python 3.9.1](https://www.python.org/)
* [OpenCV-contrib Python Bindings](https://github.com/opencv/opencv_contrib/tree/master)
* [Pathfinding Algorithm Implementations](https://github.com/brean/python-pathfinding)
