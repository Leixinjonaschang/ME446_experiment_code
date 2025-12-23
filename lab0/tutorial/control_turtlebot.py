#!/usr/bin/env python3
"""
ROS2 script to control a turtlebot in turtlesim.

This script demonstrates the basic ROS2 concepts:
- Node: This script creates a ROS2 node to control the turtle
- Topic: Publishes velocity commands to /turtle1/cmd_vel topic
- Message: Uses geometry_msgs/Twist message type for velocity commands

Make sure turtlesim is running before executing this script:
    ros2 launch turtle_tf2_py turtle_tf2_demo.launch.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class TurtleController(Node):
    """
    A ROS2 Node that controls a turtle robot.
    
    This class demonstrates:
    1. How to create a Node in ROS2 (inherits from Node)
    2. How to create a Publisher to send messages on a Topic
    3. How to publish Message (Twist) to control the robot
    """
    
    def __init__(self, turtle_name='turtle1'):
        """
        Initialize the turtle controller node.
        
        In ROS2:
        - super().__init__() creates a Node with name 'turtle_controller'
        - create_publisher() creates a Publisher for a Topic
        - The Topic name follows pattern: /{turtle_name}/cmd_vel
        - Message type is Twist (geometry_msgs/Twist)
        - Queue size 10 means it buffers up to 10 messages
        
        Args:
            turtle_name: Name of the turtle to control (default: 'turtle1')
        """
        # Initialize the ROS2 Node - this makes this class a ROS2 Node
        super().__init__('turtle_controller')
        
        # Create a Publisher for the velocity command Topic
        # Topic: /turtle1/cmd_vel (communication channel)
        # Message type: Twist (the data structure being sent)
        # Queue size: 10 (buffer size for messages)
        topic_name = f'/{turtle_name}/cmd_vel'
        self.publisher_ = self.create_publisher(Twist, topic_name, 10)
        
        self.get_logger().info(f'Controlling turtle: {turtle_name}')
        self.get_logger().info(f'Publishing to topic: {topic_name}')
        
    def move_forward(self, linear_speed=1.0, duration=2.0):
        """
        Move the turtle forward by publishing velocity commands.
        
        This demonstrates how to:
        - Create a Message (Twist)
        - Set message fields (linear.x for forward velocity)
        - Publish the message repeatedly for a duration
        
        Args:
            linear_speed: Forward speed in m/s (positive = forward)
            duration: How long to move in seconds
        """
        # Create a Twist message - this is the Message type for velocity commands
        twist = Twist()
        # Set linear velocity in x direction (forward/backward)
        # linear.y and linear.z are zero (turtlesim only uses linear.x)
        twist.linear.x = float(linear_speed)
        # Set angular velocity to zero (no rotation)
        twist.angular.z = 0.0
        
        # Publish the message repeatedly for the specified duration
        # This creates a continuous control command
        start_time = time.time()
        rate = self.create_rate(10)  # Publish at 10 Hz (10 times per second)
        
        while (time.time() - start_time) < duration:
            # Publish the message to the Topic
            self.publisher_.publish(twist)
            # Allow ROS2 to process callbacks (spin_once)
            rclpy.spin_once(self, timeout_sec=0.1)
            rate.sleep()
        
        # Stop the turtle after the duration
        self.stop()
        
    def turn_right(self, angular_speed=1.0, duration=1.57):
        """
        Turn the turtle right (clockwise) by publishing angular velocity commands.
        
        Args:
            angular_speed: Angular velocity in rad/s (positive = counterclockwise, 
                          negative = clockwise)
            duration: How long to turn in seconds (1.57 ≈ π/2 ≈ 90 degrees)
        """
        twist = Twist()
        twist.linear.x = 0.0  # No forward movement
        # Negative angular.z rotates clockwise (right turn)
        twist.angular.z = float(-angular_speed)
        
        # Publish continuously for the duration
        start_time = time.time()
        rate = self.create_rate(10)
        
        while (time.time() - start_time) < duration:
            self.publisher_.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
            rate.sleep()
        
        self.stop()
        
    def stop(self):
        """
        Stop the turtle by publishing a zero-velocity command.
        
        This publishes a Twist message with all zeros, which tells
        the turtle to stop moving immediately.
        """
        twist = Twist()  # All fields are zero by default
        self.publisher_.publish(twist)
        self.get_logger().info('Stopped turtle')


def main(args=None):
    """
    Main function demonstrating basic turtle control.
    
    This function shows:
    1. How to initialize ROS2 (rclpy.init)
    2. How to create a Node (TurtleController)
    3. How to use the Node to control the robot
    4. How to properly clean up (destroy_node, shutdown)
    """
    # Initialize ROS2 - must be called before creating any nodes
    rclpy.init(args=args)
    
    # Create a Node instance - this creates a ROS2 Node named 'turtle_controller'
    controller = TurtleController(turtle_name='turtle1')
    
    try:
        # Example: Make the turtle move forward and then turn right
        # This demonstrates basic control commands
        
        controller.get_logger().info('Starting turtle control demo...')
        
        # Move forward for 2 seconds
        controller.get_logger().info('Moving forward...')
        controller.move_forward(linear_speed=1.0, duration=2.0)
        time.sleep(0.5)  # Brief pause between commands
        
        # Turn right ~90 degrees
        controller.get_logger().info('Turning right...')
        controller.turn_right(angular_speed=1.0, duration=1.57)
        
        controller.get_logger().info('Demo completed!')
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        controller.get_logger().info('Interrupted by user')
    finally:
        # Clean up: stop the turtle, destroy the node, and shutdown ROS2
        controller.stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
