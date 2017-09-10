from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, throttle_params, brake_params, steer_params):
        # Initialize all the PIDs
        self.throttle_pid = PID(
            throttle_params['kp'],
            throttle_params['ki'],
            throttle_params['kd'],
            throttle_params['min'],
            throttle_params['max'])
        self.brake_pid = PID(
            brake_params['kp'],
            brake_params['ki'],
            brake_params['kd'],
            brake_params['min'],
            brake_params['max'])
        self.steer_pid = PID(
            steer_params['kp'],
            steer_params['ki'],
            steer_params['kd'],
            steer_params['min'],
            steer_params['max'])

    def control(self, sample_time, lin_vel_err, ang_vel_err):
        # Compute the errors
        throttle = self.throttle_pid.step(lin_vel_err, sample_time)
        brake = self.brake_pid.step(0.0 - lin_vel_err, sample_time)
        steering = self.steer_pid.step(ang_vel_err, sample_time)

        return throttle, brake, steering

    def reset(self):
        self.throttle_pid.reset()
        self.brake_pid.reset()
        self.steer_pid.reset()
