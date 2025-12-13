#!/usr/bin/env python3
"""Helper utilities for publishing goals to astar_planner."""
import math

from geometry_msgs.msg import PoseStamped


def make_goal_pose(x: float, y: float, yaw: float, frame_id: str = 'map') -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def set_goal_pose(node, goal_tuple, frame_id: str = 'map'):
    """Publish a PoseStamped goal using the provided node.

    Args:
        node: rclpy Node with an existing publisher or the ability to create one.
        goal_tuple: (x, y, yaw)
        frame_id: Frame for the goal pose (default 'map').
    """
    if not hasattr(node, '_goal_publisher'):
        node._goal_publisher = node.create_publisher(PoseStamped, '/goal_pose', 10)

    pose = make_goal_pose(goal_tuple[0], goal_tuple[1], goal_tuple[2], frame_id)
    pose.header.stamp = node.get_clock().now().to_msg()
    node._goal_publisher.publish(pose)
    return pose
