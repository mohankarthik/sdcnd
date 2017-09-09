#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint

from math import cos, sin, sqrt
from tf.transformations import euler_from_quaternion

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50  # number of waypoints we will publish. You can change this number
MPH_TO_MPS = 0.44704  # simple conversion macro


class WaypointUpdater(object):
    def __init__(self):
        """
        Initializes the WaypointUpdater class
        """
        # initialize the node with ROS
        rospy.init_node('waypoint_updater')

        # subscribe to all relevant topics
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        # TODO: Subscribe to /traffic_waypoint and /obstacle_waypoint

        # setup the publishers
        self.final_waypoints = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Log
        rospy.logdebug('Initialized Waypoint updater')

        # initialize the globals
        self.time = None
        self.pose = None
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.base_wp = None
        self.desired_vel = 20.0 # TODO: Set this dynamically based on traffic & object states

        # start a permanent spin
        rospy.spin()

    def pose_cb(self, msg):
        """
        Processes the reception of the /current_pose message. This function is also responsible for generating the
        next set of waypoints required for the /final_waypoints and publish the message.

        :param msg: A PoseStamped object
        :return: None
        """
        self.time = msg.header.stamp
        self.pose = msg.pose

        if self.base_wp is not None:
            # get closest waypoint
            index = self.__get_closest_waypoint()

            # make list of n waypoints ahead of vehicle
            lookahead_waypoints = self.__get_next_waypoints(index, LOOKAHEAD_WPS)

            # TODO: Update the desired final velocity based on the traffic & object locations

            # set velocity of all waypoints
            # TODO: Setup a spline from current velocity to desired velocity here
            for waypoint in lookahead_waypoints:
                waypoint.twist.twist.linear.x = self.desired_vel * MPH_TO_MPS

            # make lane data structure to be published
            lane = self.__make_lane(msg.header.frame_id, lookahead_waypoints)

            # publish the subset of waypoints ahead
            self.final_waypoints.publish(lane)

    def waypoints_cb(self, msg):
        """
        Processes the reception of the /base_waypoints message.

        :param msg: A Lane object
        :return: None
        """
        self.base_wp = msg.waypoints

    def velocity_cb(self, msg):
        """
        Processes the reception of the /current_velocity message

        :param msg: A TwistStamped object
        :return: None
        """
        self.lin_vel = msg.twist.linear.x
        self.ang_vel = msg.twist.angular.z

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def __get_closest_waypoint(self):
        """
        :return: Returns the closest waypoint ahead of the car
        """
        best_gap = float('inf')
        best_index = 0
        my_position = self.pose.position

        for i, waypoint in enumerate(self.base_wp):
            other_position = waypoint.pose.pose.position
            gap = self.__distance(my_position, other_position)

            if gap < best_gap:
                best_index, best_gap = i, gap

        is_behind = self.__is_waypoint_behind(self.base_wp[best_index])
        if is_behind:
            best_index += 1
        return best_index

    def __is_waypoint_behind(self, waypoint):
        """
        Checks if a given waypoint is behind the car's current position based on it's orientation

        :param waypoint: The waypoint
        :return: True / False
        """
        _, _, yaw = euler_from_quaternion([self.pose.orientation.x,
                                           self.pose.orientation.y,
                                           self.pose.orientation.z,
                                           self.pose.orientation.w])
        origin_x = self.pose.position.x
        origin_y = self.pose.position.y

        shift_x = waypoint.pose.pose.position.x - origin_x
        shift_y = waypoint.pose.pose.position.y - origin_y

        x = (shift_x * cos(0.0 - yaw)) - (shift_y * sin(0.0 - yaw))

        if x > 0.0:
            return False
        return True

    def __get_next_waypoints(self, i, n):
        """
        Gets the next n waypoints starting from i
        :param i: The start location
        :param n: The number of waypoints to get
        :return: A list of waypoints from base_wp
        """
        m = min(len(self.base_wp), i + n)
        return self.base_wp[i:m]

    @staticmethod
    def __make_lane(frame_id, waypoints):
        """
        Makes a lane object given the frame_id and waypoints

        :param frame_id: The sequence number to be used
        :param waypoints: A set of waypoints to be published
        :return: A Lane object
        """
        lane = Lane()
        lane.header.frame_id = frame_id
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        return lane

    @staticmethod
    def __distance(a, b):
        """
        Returns the distance between two positions
        :param a: Position 1
        :param b: Position 2
        :return: Float32 distance between two positions
        """
        return sqrt(((a.x - b.x) ** 2) + ((a.y - b.y) ** 2))

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
