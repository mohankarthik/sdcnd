# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

---

## Write-up

The write-up for this project is present in this [Medium post](https://medium.com/@mohankarthik/path-planning-in-highways-for-an-autonomous-vehicle-242b91e6387d)

The major portions of the code are
1. PathPlanner class in main.cpp takes care of the entire software implementation
2. PathPlanner::Plan is the higher level interface API that does all the planning
3. PathPlanner::HandleFirstCycle takes care of the first cycle, by essentially setting up a safe speed trajectory from 0 to desired velocity.
4. PathPlanner::HandleGenericCycle takes care of the generic case of all cycles, uses previous path, desired speed and lane to compute effective trajectories.
5. PathPlanner::BehaviourPlanner plans the behaviour of the car by considering sensorFusion data and planning to stay of change lanes using scoring. More details are there in the medium post.

---

## Infrastructure

You can download the [Term3 Simulator v1.2](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).

---

## Basic Build Instructions

1. Make a build directory: `mkdir build && cd build`
2. Compile: `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make`
3. Run it: `./path_planning`.

(or)
1. Run it using ./run.sh

---

## Basic Execution Instructions

1. The project is meant to be run on a powerful PC, enough to predict 40 points within 0.2ms and not miss too many cycles. The car's performance could suffer is sufficient processing power is not available.
2. The project prints out a set of data to aid in understanding it's choices. This uses `system("clear")`. If you are not running this on bash, this statement could crash. Please turn this off at Line No: 609.

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
---

## References
* [Source Respository](https://github.com/udacity/CarND-Path-Planning-Project)
* [Simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2)
* [Write Up](https://medium.com/@mohankarthik/path-planning-in-highways-for-an-autonomous-vehicle-242b91e6387d)
* [Video](https://youtu.be/PqbAUjUfMCo)

---
