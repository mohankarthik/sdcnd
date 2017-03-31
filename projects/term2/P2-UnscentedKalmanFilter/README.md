# Unscented Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

---

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## Basic Build Instructions

1. Clone this repo.
2. Switch to the build directory: `cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF path/to/input.txt path/to/output.txt radar_state lidar_state debug_enabled`. You can find
   some sample inputs in 'data/'.
    - eg. `./UnscentedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt true true false`
5. There is a small shell script that automates the execution of all the combinations in `build/run_all.sh`. This will create
   all the relevant output files and debug files as `out_<comb>_<#>.txt` and `dbg_<comb>_<#>.txt`. 
     - where `<comb>` can be both, radar, lidar or none.
     - `<#>` is either 1 or 2 depending on the dataset used.
     - Ensure that the shell script has execute permissions.


## Results

The results achieved at my PC are for the 
**1st dataset**
```
Accuracy - RMSE:
0.0726209
0.074172
0.605913
0.573694
Done!
```

**2nd dataset**
```
Accuracy - RMSE:
0.192507
0.18681
0.303473
0.404276
Done!
```

## Visualizations

The visualizations of all the combinations are present in the Docs folder. Please refer to them for how the algorithm has performed.

## Issues

1. Radar performance in general seems to be more problematic when compared to Lidar. This is seen both in it's estimates, RMSE values and the NIS values. Not sure if there is a bug in the code.
  - This is especially evident in dataset 2 and can be seen in the visualizations. It's just completely off. But along with Lidar data, the algorithm is able to converge.
2. There are possible divide by 0 issues in the radar measurement update (specifically in the Zsig calculations when dividing by the `sqrt_term`. But I'm unable to think of an alternative to place when it does encounter a divide by 0, so I've left it as it is.


