#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
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

        # Setup the parameters
        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.decel_limit = rospy.get_param('~decel_limit', -5)
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.cur_vel_cb)

        # Setup the outgoing publishers
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # Create the `Controller` object
        self.controller = Controller(self.wheel_base, self.steer_ratio, 0.0, self.max_lat_accel, self.max_steer_angle)

        # Other variables
        self.time = 0.0
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.cur_vel = 0.0
        self.dbw_enabled = True

        # Go into a loop
        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # Get predicted throttle, brake, and steering using `twist_controller`
            # throttle, brake, steer = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            if not rospy.is_shutdown() and self.dbw_enabled and self.time > 0:
                throttle, brake, steer = self.controller.control(self.time, self.lin_vel, self.ang_vel, self.cur_vel)

                # TODO: You should only publish the control commands if dbw is enabled
                self.publish(throttle, brake, steer)

            rate.sleep()

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

    def twist_cmd_cb(self, msg):
        # Preconditions
        #assert msg.twist.linear.y == 0 and msg.twist.linear.z == 0, 'Y or Z linear velocity is not 0!'
        #assert msg.twist.angular.y == 0 and msg.twist.angular.z == 0, 'Y or Z angular velocity is not 0!'

        # Save the currently aimed angular and linear velocities
        self.lin_vel = msg.twist.linear.x
        self.ang_vel = msg.twist.angular.x

        # Save the time
        self.time = msg.header.stamp.secs + (msg.header.stamp.nsecs / float(1e9))
        print (self.time)

    def dbw_enabled_cb(self, status):
        # Save the status
        self.dbw_enabled = status

        # If Drive-By-Wire is turned off, reset the controller
        if status == False:
            self.controller.reset()

    def cur_vel_cb(self, msg):
        # Save the values
        self.cur_vel = msg.twist.linear.x


if __name__ == '__main__':
    DBWNode()
