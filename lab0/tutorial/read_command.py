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
from geometry_msgs.msg import Twist


class CommandReader(Node):
    """
    A ROS2 Node that reads velocity commands.
    
    This class demonstrates:
    1. How to create a Node in ROS2
    2. How to create a Subscriber to receive messages on a Topic
    3. How to process received Message (Twist) using a callback
    """
    
    def __init__(self, turtle_name='turtle1'):
        """
        Initialize the command reader node.
        
        Args:
            turtle_name: Name of the turtle to monitor (default: 'turtle1')
        """
        super().__init__('command_reader')
        
        # Create a Subscriber for the velocity command Topic
        # Topic: /turtle1/cmd_vel
        # Message type: Twist
        # Callback: self.listener_callback (function to call when message is received)
        # Queue size: 10
        topic_name = f'/{turtle_name}/cmd_vel'
        self.subscription = self.create_subscription(
            Twist,
            topic_name,
            self.listener_callback,
            10)
        
        self.get_logger().info(f'Subscribed to topic: {topic_name}')

    def listener_callback(self, msg):
        """
        Callback function called whenever a new message is received.
        
        Args:
            msg: The received Twist message
        """
        # Print the received linear and angular velocity
        self.get_logger().info(
            f'Received cmd_vel -> Linear: x={msg.linear.x:.2f}, '
            f'Angular: z={msg.angular.z:.2f}'
        )


def main(args=None):
    """
    Main function demonstrating basic subscriber setup.
    """
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the reader node
    reader = CommandReader(turtle_name='turtle1')
    
    try:
        # Spin the node so the callback function can be called
        # This will keep the script running until Ctrl+C is pressed
        rclpy.spin(reader)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        reader.get_logger().info('Stopping reader...')
    finally:
        # Clean up
        reader.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

