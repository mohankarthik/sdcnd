# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

## Data Manipulations
There are two important things to do with the incoming data.
1. Convert it to vehicle co-ordinates (so that straight lines don't cause numerical issues)
2. Re-evaluate the state considering the **100ms** latency (induced in the simulator)
3. This data is then used to fit a **polynomial** of **order 3**, which reasonably captured the waypoint information.

## The Model
Once the data is ready, the model can be formed. The model is based on Model Predictive Control: given a reference trajectory, using lpopt optimizer to optimize a tracjectory/control(steering and throttling) parameters to achive a lowest defined cost function value. The state used in the project includes:
1. The vehicle position (x & y)
2. The heading of the vehicle (psi)
3. The velocity of the vehicle (v)
4. The cross track error (i.e. offset from center of the road) (cte)
5. The heading error (i.e. the difference between ideal heading and actual heading) (epsi)

The actuators are the steering parameter and the throttle parameter that get to sent to the car control as the model output. The updates are performed every moment live with a forced latency of 100ms.

##  Hyper Parameters
The hyper parmeters (the weights of the cost function, N and dt) were chosen like this

### Wetights
<pre>
#define W_CTE           (1000.0) /*!< CTE and EPSI is very important */
#define W_EPSI          (1000.0)
#define W_V             (0.01)   /*!< Increase this for more reckless behaviour :P */
#define W_STEER         (1.0)    /*!< These are all equally important */
#define W_THROTTLE      (1.0)  
#define W_STEER_DIFF    (1.0)
#define W_THROTTLE_DIFF (1.0)
</pre>

### Reference values
<pre>
#define REF_CTE         (0.0)
#define REF_EPSI        (0.0)
#define REF_V           (200.0) /*!< Aim for the maximum speed possible */
</pre>

### MPC values
<pre>
#define N               (10)   /*!< Predict 1 second in future */
#define DT              (0.10) /*!< Every 100ms */
#define NUM_VARS        (6)
</pre>
1. I tried a smaller value of DT (0.01 -> 0.03 -> 0.10), but the model quickly became unstable, so 0.10 was the smallest DT that worked (probably because of machine dependance)
2. I did not see much improvement increase N further(10 -> 15 -> 20). So for the sake of optimization, I kept N at 10 which seems to almost match the amount of waypoints that we get as well (visually). So I thought that this was a good idea. In places where N * dt exceeds the waypoint input, we can see that the MPC outputs are sometimes wrong, which also makes sense, because it's just wildly guessing there. 

## Constraint Equation
Used these equations for the model:
<pre>
const auto dMultiplier = (steer0 / LF);
fg[2 + X_START + i] = x1 - (x0 + (v0 * CppAD::cos(psi0) * DT));
fg[2 + Y_START + i] = y1 - (y0 + (v0 * CppAD::sin(psi0) * DT));
fg[2 + PSI_START + i] = psi1 - (psi0 - (v0 * dMultiplier * DT));
fg[2 + V_START + i] = v1 - (v0 + (throttle0 * DT));
fg[2 + CTE_START + i] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * DT));
fg[2 + EPSI_START + i] = epsi1 - ((psi0 - psides0) - (v0 * dMultiplier * DT));
</pre>

## Playing with the solution
You can modify the W_V parameter in `mpc.cpp` anywhere from 0.1 to 0.01 to influence the driving from reckless to cautious. The performance also depends on your machine, for me 1.0 reaches about 120mph but after a couple of laps becomes unstable. And 0.01 reaches a max of 80mph, is very cautious on the curves and is very stable.
The 0.1 run on my machine is captured here (https://youtu.be/ndK_wXyNAYc)

## Thoughts
One interesting idea that I had was, if we had a racing tragectory, you know how racing cars go to the right curb when they've a hard left turn and then turn through the curb, then we would be able to reach much higher speeds and the car would be much more stable too. Making the car drive through the center while trying to race isn't that very optimal :smile:

# Original README
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
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) == 0.14, but the master branch will probably work just fine
  * Follow the instructions in the [uWebSockets README](https://github.com/uWebSockets/uWebSockets/blob/master/README.md) to get setup for your platform. You can download the zip of the appropriate version from the [releases page](https://github.com/uWebSockets/uWebSockets/releases). Here's a link to the [v0.14 zip](https://github.com/uWebSockets/uWebSockets/archive/v0.14.0.zip).
  * If you have MacOS and have [Homebrew](https://brew.sh/) installed you can just run the ./install-mac.sh script to install this.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt --with-openblas`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/CarND-MPC-Project/releases).



## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./
