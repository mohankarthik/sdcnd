#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        # Initialize the node
        rospy.init_node('dbw_node')

        # Setup the Constants
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # Setup the global variables
        self.exp_lin_vel = 0.0
        self.exp_ang_vel = 0.0
        self.act_lin_vel = 0.0
        self.act_ang_vel = 0.0

        # Tracking information
        self.time = None

        # Debug
        self.run_cnt = 0
        self.dir = 1

        # Setup the publishers
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # Create `TwistController` object
        throttle_params = {
            'kp': 1.0,
            'ki': 0.0,
            'kd': 0.0,
            'max': float('inf'),
            'min': 0.0,
        }
        brake_params = {
            'kp': 1.0,
            'ki': 0.0,
            'kd': 0.0,
            'max': float('inf'),
            'min': 0.0,
        }
        steer_params = {
            'kp': 3.0,
            'ki': 0.0,
            'kd': 0.0,
            'max': float('inf'),
            'min': float('-inf'),
        }
        self.controller = Controller(throttle_params, brake_params, steer_params)

        # Subscribe to all the topics you need to
        self.sub_twist_cmd = rospy.Subscriber('/twist_cmd', TwistStamped, self.__twist_cb, queue_size=1)
        self.sub_cur_vel = rospy.Subscriber('/current_velocity', TwistStamped, self.__vel_cb, queue_size=1)

        self.loop()

    def loop(self):
        # Lower the rate to avoid performance issues
        # https://carnd.slack.com/archives/C6NVDVAQ3/p1504061507000179
        rate = rospy.Rate(10) # 50Hz
        while not rospy.is_shutdown():
            if self.time is None:
                self.time = rospy.get_time()
                rospy.loginfo(self.time)
            else:
                sample_time = rospy.get_time() - self.time
                lin_err = self.exp_lin_vel - self.act_lin_vel
                ang_err = self.exp_ang_vel - self.act_ang_vel

                rospy.loginfo(sample_time)
                rospy.loginfo(lin_err)
                rospy.loginfo(ang_err)

                # Get predicted throttle, brake, and steering using `twist_controller`
                # You should only publish the control commands if dbw is enabled
                throttle, brake, steering = self.controller.control(sample_time, lin_err, ang_err)
                self.publish(throttle, brake, steering)

            rate.sleep()

    def __twist_cb(self, msg):
        self.exp_lin_vel = msg.twist.linear.x
        self.exp_ang_vel = msg.twist.angular.z

    def __vel_cb(self, msg):
        self.act_lin_vel = msg.twist.linear.x
        self.act_ang_vel = msg.twist.angular.z

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
