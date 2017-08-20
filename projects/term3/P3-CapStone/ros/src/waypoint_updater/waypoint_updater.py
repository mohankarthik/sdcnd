#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        # Initialize the node
        rospy.init_node('waypoint_updater')

        # Initialize the incoming topics
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # Initialize the outgoing final waypoints topic
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Other variables
        self.mapX = []  # variable to store waypoint X coordinate from /base_waypoints
        self.mapY = []  # variable to store waypoint Y coordinate from /base_waypoints
        self.map_wp_len = 0  # numper of map waypoints in /base_waypoints
        self.mapWP = []  # array to store map waypoints for full information

        # Spin the node
        rospy.spin()

    def pose_cb(self, msg):
        print 'got pose info'
        pose_x = msg.pose.position.x
        pose_y = msg.pose.position.y
        nearest_waypoint = self.closest_waypoint(pose_x, pose_y, self.mapX, self.mapY)
        # TODO: can return closest waypoint even if it is behind, need to add orientation check

        # Get next LOOKAHEAD_WPS waypoints
        pub_list = []
        for i in range(LOOKAHEAD_WPS):
            pub_list.append(self.mapWP[nearest_waypoint + i])

        # Create correct data structure for publishing
        lane = Lane()
        lane.header.frame_id = msg.header.frame_id
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = pub_list

        # Publish to /final_waypoints
        if not rospy.is_shutdown():
            self.final_waypoints_pub.publish(lane)
        pass

    def waypoints_cb(self, waypoints):
        # Create two lists one with X and other with Y waypoints
        new_mapX = []
        new_mapY = []
        new_mapWP = []
        for waypoint in waypoints.waypoints[:]:
            new_mapX.append(waypoint.pose.pose.position.x)
            new_mapY.append(waypoint.pose.pose.position.y)
            new_mapWP.append(waypoint)

        # Assign to global variables
        self.mapX = new_mapX
        self.mapY = new_mapY
        self.mapWP = new_mapWP
        self.map_wp_len = len(waypoints.waypoints)
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    # Calculate distance between two points
    def dist_two_points(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    # Finds nearest waypoint
    def closest_waypoint(self, x, y, maps_x, maps_y):
        closest_wp_dist = 999999.9
        closest_wp = 0

        for i in range(self.map_wp_len):
            map_x = maps_x[i]
            map_y = maps_y[i]
            dist = self.dist_two_points(x, y, map_x, map_y)
            if (dist < closest_wp_dist):
                closest_wp_dist = dist
                closest_wp = i
        return closest_wp


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
