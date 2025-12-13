#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from goal_utils import set_goal_pose


class ExampleGoalPublisher(Node):
    def __init__(self):
        super().__init__('example_goal_publisher')
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.goal_sent = False

    def timer_callback(self):
        if not self.goal_sent:
            set_goal_pose(self, (2.0, 4.0, 1.5))
            self.get_logger().info('Goal (2.0, 4.0, 1.5) published')
            self.goal_sent = True
        else:
            self.get_logger().info('Goal already sent. Shutting down.')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ExampleGoalPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
