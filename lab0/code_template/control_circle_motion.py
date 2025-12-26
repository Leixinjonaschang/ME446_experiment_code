#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TurtleController(Node):
    def __init__(self):
        # Initialize the node with the name 'turtle_controller'
        super().__init__('turtle_controller')
        
        # Create a publisher that publishes Twist messages to '/turtle1/cmd_vel'
        # The queue size is set to 10
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Create a timer that triggers the callback every 0.5 seconds (2 Hz)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('Turtle Controller Node has been started.')

    def timer_callback(self):
        """
        This function is called periodically. 
        It creates a Twist message and publishes it to control the robot.
        """
        msg = Twist()

        # ---------------------------------------------------------
        # TODO: Modify the code below to make the turtle move in a CIRCLE.
        # Hint: A circle requires both linear velocity (forward speed) 
        #       and angular velocity (turning speed) at the same time.
        # ---------------------------------------------------------
        
        # Reference Answer:
        # msg.linear.x = 2.0   # Forward velocity
        # msg.angular.z = 1.0  # Angular velocity
        
        # Linear velocity (m/s)
        # msg.linear.x = 0.0  # Replace 0.0 with your value
        # msg.linear.y = 0.0  # Usually 0 for non-holonomic robots
        # msg.linear.z = 0.0
        
        # Angular velocity (rad/s)
        # msg.angular.x = 0.0
        # msg.angular.y = 0.0
        # msg.angular.z = 0.0  # Replace 0.0 with your value

        # ---------------------------------------------------------
        # Example Solution (Commented out):
        # msg.linear.x = 1.0   # Move forward
        # msg.angular.z = 1.0  # Turn left
        # ---------------------------------------------------------

        # Publish the message
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: Linear X: {msg.linear.x}, Angular Z: {msg.angular.z}')

def main(args=None):
    # Initialize the ROS client library
    rclpy.init(args=args)

    # Create the node instance
    turtle_controller = TurtleController()

    try:
        # Spin the node so the callback function is called
        rclpy.spin(turtle_controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        turtle_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()