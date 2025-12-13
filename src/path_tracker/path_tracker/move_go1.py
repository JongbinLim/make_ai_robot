#!/usr/bin/env python3

"""
GO1 Interactive Path Tracker Node

This node provides an interactive interface to command the GO1 robot to move to
target positions. It continuously accepts user input for target positions and
generates smooth curved paths using cubic Hermite splines.

The node runs in a loop, allowing you to:
- Send multiple trajectory commands without restarting
- Update the target position while the robot is still moving
- Monitor the robot's progress in real-time

Usage:
    ros2 run path_tracker move_go1.py
    
    Then enter target positions when prompted:
    Enter target (x y yaw): 2.0 1.0 0.0
    Enter target (x y yaw): -1.0 2.0 1.57
    
    Or press Ctrl+C to exit
"""

import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist
from nav_msgs.msg import Path


# Constants
MIN_PATH_POINTS = 60             # Minimum number of waypoints in path
MAX_PATH_POINTS = 100             # Maximum number of waypoints in path
PATH_DENSITY = 20                # Waypoints per meter of path


class MoveGo1(Node):
    """
    Interactive path tracker node that generates smooth paths to target positions.
    
    This node:
    1. Waits for the initial robot pose
    2. Continuously accepts user input for target positions
    3. Generates smooth curved paths using cubic Hermite splines
    4. Publishes paths to the path tracker controller
    5. Allows updating targets while the robot is moving
    """
    def __init__(self):
        """
        Initialize the interactive path tracker node.
        """
        super().__init__('path_tracker')
        
        # State tracking
        self.robot_pose = None           # Current robot pose
        self.current_cmd_vel = None      # Latest velocity command
        
        # Target tracking
        self.target_x = None             # Current target x position
        self.target_y = None             # Current target y position
        self.target_yaw = None           # Current target orientation
        self.has_new_target = False      # Flag indicating new target is ready
        
        # Subscribe to robot pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/go1_pose',
            self.pose_callback,
            10
        )
        
        # Subscribe to velocity commands (to monitor robot motion)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Subscribe to tracking info (cross-track error, target speed, current speed)
        self.tracking_info_sub = self.create_subscription(
            Point,
            '/path_tracker/tracking_info',
            self.tracking_info_callback,
            10
        )
        
        # Publish path for the controller to follow
        self.path_pub = self.create_publisher(
            Path,
            '/local_path',
            10
        )
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Interactive Path Tracker Node Started')
        self.get_logger().info('Waiting for robot pose from /go1_pose...')
        self.get_logger().info('=' * 60)
        
    def pose_callback(self, msg):
        """
        Callback for robot pose updates.
        
        Args:
            msg (PoseStamped): Current robot pose
        """
        was_none = self.robot_pose is None
        self.robot_pose = msg
        
        if was_none:
            self.get_logger().info(
                f'Robot pose received: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
            )
            self.get_logger().info('Ready to accept commands!')
            self.get_logger().info('=' * 60)
    
    def cmd_vel_callback(self, msg):
        """
        Callback for velocity commands.
        
        Args:
            msg (Twist): Velocity command
        """
        self.current_cmd_vel = msg
    
    def tracking_info_callback(self, msg):
        """
        Callback for path tracking information.
        
        Args:
            msg (Point): Tracking info where x=cross_track_error, y=target_speed, z=current_speed
        """
        # Can be used for monitoring if needed
        pass
    
    def quaternion_to_yaw(self, quaternion):
        """
        Convert quaternion to yaw angle.
        
        Extracts the yaw (rotation around z-axis) from a quaternion representation.
        
        Args:
            quaternion: Quaternion with x, y, z, w components
            
        Returns:
            float: Yaw angle in radians
        """
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle to quaternion.
        
        Creates a quaternion representing a rotation around the z-axis.
        
        Args:
            yaw (float): Yaw angle in radians
            
        Returns:
            Quaternion: Quaternion representation of the yaw rotation
        """
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
    
    def generate_smooth_path(self, start_pose, target_x, target_y, target_yaw, num_points=60):
        """
        Generate a smooth curved path from current position to target.
        
        Uses cubic Hermite spline interpolation to create a smooth path that respects
        both initial and final orientations. The path is generated by:
        1. Creating control points based on start and target orientations
        2. Using Hermite basis functions to interpolate positions
        3. Computing orientations from path tangents
        
        Args:
            start_pose (PoseStamped): Current robot pose
            target_x (float): Target x position in meters
            target_y (float): Target y position in meters
            target_yaw (float): Target orientation in radians
            num_points (int): Number of waypoints along the path
            
        Returns:
            Path: ROS Path message containing the smooth trajectory
        """
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        
        # Start position and orientation
        x0 = start_pose.pose.position.x
        y0 = start_pose.pose.position.y
        z0 = start_pose.pose.position.z
        yaw0 = self.quaternion_to_yaw(start_pose.pose.orientation)
        
        # Target position and orientation
        x1 = target_x
        y1 = target_y
        z1 = 0.0  # Keep z at ground level
        yaw1 = target_yaw
        
        # Calculate distance to target
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        # Scale control points based on distance (longer paths = longer tangent vectors)
        # Clamped to maximum of 2.0 meters for stability
        control_scale = min(distance * 0.5, 2.0)
        
        # Start control point: extends in direction of initial orientation
        cx0 = x0 + control_scale * math.cos(yaw0)
        cy0 = y0 + control_scale * math.sin(yaw0)
        
        # End control point: comes from direction of final orientation
        cx1 = x1 - control_scale * math.cos(yaw1)
        cy1 = y1 - control_scale * math.sin(yaw1)
        
        # Generate path waypoints using cubic Hermite interpolation
        for i in range(num_points + 1):
            t = i / num_points  # Interpolation parameter: 0 (start) to 1 (end)
            
            # Cubic Hermite basis functions
            h00 = 2*t**3 - 3*t**2 + 1      # Blends start point
            h10 = t**3 - 2*t**2 + t        # Blends start tangent
            h01 = -2*t**3 + 3*t**2         # Blends end point
            h11 = t**3 - t**2              # Blends end tangent
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            
            # Compute position using Hermite spline formula:
            # P(t) = h00*P0 + h10*T0 + h01*P1 + h11*T1
            # where P0, P1 are start/end points and T0, T1 are tangent vectors
            tangent_x0 = cx0 - x0
            tangent_y0 = cy0 - y0
            tangent_x1 = x1 - cx1
            tangent_y1 = y1 - cy1
            
            pose_stamped.pose.position.x = (h00 * x0 + h10 * tangent_x0 + 
                                           h01 * x1 + h11 * tangent_x1)
            pose_stamped.pose.position.y = (h00 * y0 + h10 * tangent_y0 + 
                                           h01 * y1 + h11 * tangent_y1)
            pose_stamped.pose.position.z = z0 + t * (z1 - z0)
            
            # Compute orientation from path tangent vector
            if i < num_points:
                # Look ahead to next point to compute tangent
                t_next = (i + 1) / num_points
                h00_next = 2*t_next**3 - 3*t_next**2 + 1
                h10_next = t_next**3 - 2*t_next**2 + t_next
                h01_next = -2*t_next**3 + 3*t_next**2
                h11_next = t_next**3 - t_next**2
                
                x_next = (h00_next * x0 + h10_next * tangent_x0 + 
                         h01_next * x1 + h11_next * tangent_x1)
                y_next = (h00_next * y0 + h10_next * tangent_y0 + 
                         h01_next * y1 + h11_next * tangent_y1)
                
                # Tangent direction
                dx = x_next - pose_stamped.pose.position.x
                dy = y_next - pose_stamped.pose.position.y
                tangent_yaw = math.atan2(dy, dx)
            else:
                # Last point uses final orientation
                tangent_yaw = yaw1
            
            pose_stamped.pose.orientation = self.yaw_to_quaternion(tangent_yaw)
            path.poses.append(pose_stamped)
        
        return path
    
    def set_target(self, target_x, target_y, target_yaw):
        """
        Set a new target position and generate path.
        
        Args:
            target_x (float): Target x position in meters
            target_y (float): Target y position in meters
            target_yaw (float): Target orientation in radians
        """
        if self.robot_pose is None:
            self.get_logger().warn('Cannot set target: Robot pose not available yet')
            return False
        
        self.target_x = target_x
        self.target_y = target_y
        self.target_yaw = target_yaw
        
        # Get current position
        x_start = self.robot_pose.pose.position.x
        y_start = self.robot_pose.pose.position.y
        yaw_start = self.quaternion_to_yaw(self.robot_pose.pose.orientation)
        
        # Calculate distance to target
        distance = math.sqrt((target_x - x_start)**2 + (target_y - y_start)**2)
        
        # Determine number of waypoints based on path distance
        num_points = max(MIN_PATH_POINTS, min(MAX_PATH_POINTS, int(distance * PATH_DENSITY)))
        
        # Log path information
        self.get_logger().info('=' * 60)
        self.get_logger().info('Generating new path...')
        self.get_logger().info(f'  From: ({x_start:.3f}, {y_start:.3f}, {math.degrees(yaw_start):.1f}°)')
        self.get_logger().info(f'  To:   ({target_x:.3f}, {target_y:.3f}, {math.degrees(target_yaw):.1f}°)')
        self.get_logger().info(f'  Distance: {distance:.3f}m with {num_points} waypoints')
        
        # Generate smooth path
        path = self.generate_smooth_path(
            self.robot_pose,
            target_x,
            target_y,
            target_yaw,
            num_points=num_points
        )
        
        # Publish the path to the controller
        self.path_pub.publish(path)
        self.get_logger().info(f'Path published with {len(path.poses)} points.')
        self.get_logger().info('=' * 60)
        
        return True


def input_thread(node):
    """
    Thread function to handle user input for target positions.
    
    Continuously prompts the user for target positions and sends them
    to the node for path generation.
    
    Args:
        node: MoveGo1 node instance
    """
    # Wait for ROS node to initialize and display startup messages
    time.sleep(3.0)
    
    print("\n" + "=" * 60)
    print("Interactive Path Tracker")
    print("=" * 60)
    print("Enter target positions in the format: x y yaw")
    print("  x   : Target x position (meters)")
    print("  y   : Target y position (meters)")
    print("  yaw : Target orientation (radians)")
    print()
    print("Examples:")
    print("  2.0 1.0 0.0      # Move to (2.0, 1.0) facing east")
    print("  -1.0 2.0 1.57    # Move to (-1.0, 2.0) facing north")
    print("  0.0 0.0 0.0      # Return to origin")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 60)
    
    while rclpy.ok():
        try:
            # Get user input
            user_input = input("\nEnter target (x y yaw): ").strip()
            
            if not user_input:
                continue
            
            # Parse input
            parts = user_input.split()
            if len(parts) != 3:
                print("Error: Please enter exactly 3 values (x y yaw)")
                continue
            
            try:
                target_x = float(parts[0])
                target_y = float(parts[1])
                target_yaw = float(parts[2])
            except ValueError:
                print("Error: All values must be valid numbers")
                continue
            
            # Send target to node
            node.set_target(target_x, target_y, target_yaw)
            
        except EOFError:
            # Handle Ctrl+D
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            break


def main(args=None):
    """
    Main function to start the interactive path tracker node.
    
    Creates the node and starts both the ROS 2 spin loop and the
    user input thread.
    
    Args:
        args: Command line arguments (optional)
    """
    # Initialize ROS 2
    rclpy.init(args=args)
    
    # Create the node
    node = MoveGo1()
    
    # Start input thread
    input_thread_handle = threading.Thread(target=input_thread, args=(node,), daemon=True)
    input_thread_handle.start()
    
    try:
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()