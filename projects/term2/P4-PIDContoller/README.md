# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## PID Tuning approach
My approach to PID tuning was to initially set all the 3 gains Kp, Ki and Kd to 0 and check that the car drives straight. Once I ensured that the framework is running correctly, I increased Kp in small amounts, until the car locked in the track with a small throttle (0.1). Then I increased Kd till it achieved a stable drive. I've left Ki to be 0, since there does not seem to be any significant steering drift in the simulator (but in a real car, this would be the next parameter I tune).

Now that I've stable parameters for a throttle of 0.1, I started increasing the throttle (in steps of 0.1), while repeating the earlier process. I was able to reach only upto 0.3 after which no matter what I tune, the car went out the track surface.

Here is where I added a control for the throttle based on the steering. Which is what you see in 82+ in pid.cpp. With this, I was able to get the speed up to 50mph while ensuring that the car remains in the track.

## Final PID values
Kp = 0.14
Ki = 0
Kd = 1.5

## Results
The video can be seen here (https://youtu.be/gPHpdfAvfvU).

# Future work
While twiddle looks great, from discussions on slack, etc. it looks like the car takes a really long time to drive and learn how to drive. I definitely want to try this once, but I also want to take a machine learning approach and see if I can implement maybe a simple linear regression for the car to figure out the parameters.