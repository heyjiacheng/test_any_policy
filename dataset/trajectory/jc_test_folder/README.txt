# Set frame_id = None, visualize the last grasp 
python convert_gripper_pose.py --viz

# Set frame_id = N , visualize the Nth grasp 
python convert_gripper_pose.py --frame_id k --viz

# Set frame_id = -1, visualize all grasps (trajectory)
python convert_gripper_pose.py --frame_id -1 --viz