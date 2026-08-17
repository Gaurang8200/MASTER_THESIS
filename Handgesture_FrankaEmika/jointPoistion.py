from franky import *

robot = Robot("172.16.0.2")

# Get the current state as `franky.RobotState`. See the documentation for a list of fields.
state = robot.state


# Get the robot's joint state
joint_state = robot.current_joint_state
joint_pos = joint_state.position
joint_vel = joint_state.velocity

print("Current joint positions:", joint_pos)
