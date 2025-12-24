#!/usr/bin/env python3
"""
ROS2 script to read velocity commands for a turtlebot in turtlesim.

This script demonstrates the basic ROS2 concept:
- Subscriber: This script creates a ROS2 node to listen to velocity commands
- Topic: Subscribes to /turtle1/cmd_vel topic
- Message: Uses geometry_msgs/Twist message type
"""

import rclpy
from rclpy.node import Node

# TODO: import the message type for robot pose
# Reference Answer:
# from turtlesim.msg import Pose


class PoseReader(Node):
    """
    A ROS2 Node that reads Robot Pose.
    
    This class demonstrates:
    1. How to create a Node in ROS2
    2. How to create a Subscriber to receive messages on a Topic
    3. How to process received Message (Twist) using a callback
    """
    
    def __init__(self, turtle_name='turtle1'):
        """
        Initialize the Pose reader node.
        
        Args:
            turtle_name: Name of the turtle to monitor (default: 'turtle1')
        """
        super().__init__('pose_reader')
        
        # TODO: Create a Subscriber for the robot pose Topic
        # Reference Answer:
        # self.subscription = self.create_subscription(Pose, f'/{turtle_name}/pose', self.listener_callback, 10)
        
        # TODO: Print the subscribed topic
        # Reference Answer:
        # self.get_logger().info(f'Subscribed to: /{turtle_name}/pose')
        pass

    def listener_callback(self, msg):
        """
        Callback function called whenever a new message is received.
        
        Args:
            msg: The received Twist message
        """
        # TODO：Print the received linear and angular velocity
        # Reference Answer:
        # self.get_logger().info(f'Pose -> x: {msg.x:.2f}, y: {msg.y:.2f}, theta: {msg.theta:.2f}')
        pass


def main(args=None):
    """
    Main function demonstrating basic subscriber setup.
    """
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the reader node
    pose_reader = PoseReader(turtle_name='turtle1')
    
    try:
        # Spin the node so the callback function can be called
        # This will keep the script running until Ctrl+C is pressed
        rclpy.spin(pose_reader)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pose_reader.get_logger().info('Stopping pose reader...')
    finally:
        # Clean up
        pose_reader.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

