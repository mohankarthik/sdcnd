from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        # Initialize the steering controller using the YawController
        self.steer_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # Initialize the throttle/break controller using PID and lowpass
        self.throttle_controller = PID(0.0, 0.0, 0.0)

        # Set up low pass filters
        self.linear_filt = LowPassFilter(0.9, 0.1)  # 90% to prev. value & 10% to current value
        self.angular_filt = LowPassFilter(0.9, 0.1) # 90% to prev. value & 10% to current value

        # Other variables
        self.time = 0.0

        pass

    def control(self, time, linear_velocity, angular_velocity, current_velocity):
        # Preconditions
        if self.time != 0:
            assert time >= self.time, 'Time is going in reverse!! {}:{}'.format(time, self.time)

        # Pass the values through a LPF
        prop_lin_vel = self.linear_filt.filt(linear_velocity)
        prop_ang_vel = self.angular_filt.filt(angular_velocity)

        # Compute the error
        error = prop_lin_vel - current_velocity

        # Compute and return the throttle, break and steering
        throttle = 1.0
        brek = 0.0

        # If this is not the first time instant, run the PID
        if time > 0.0 and (time - self.time) > 0:
            throttle = self.throttle_controller.step(error, (time - self.time))

        # Normalize the values and set break / throttle correctly
        if throttle < 0.0:
            brek = -throttle
            throttle = 0.0

        # Compute the steering
        steer = self.steer_controller.get_steering(prop_lin_vel, prop_ang_vel, current_velocity)

        # Save the time
        self.time = time

        # Return the values back
        return throttle, brek, steer

    def reset(self):
        """
        Resets the controller, typically used when there is a fatal error, or when a safety driver takes over

        :return: None
        """
        self.throttle_controller.reset()
        self.angular_filt.reset()
        self.linear_filt.reset()
        self.time = 0.0


if __name__ == '__main__':
    Controller(arg1='10')