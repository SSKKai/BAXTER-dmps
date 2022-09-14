import rospy
from trac_ik_python.trac_ik import IK
from numpy.random import random
import numpy as np
import time


def ikine(x, y, z, qx, qy, qz, qw, seed = None, arm = 'left'):
	# roslaunch pr2_description upload_pr2.launch
    # Needed beforehand

    # Identifying the robot and getting data using URDF
    urdf_str = rospy.get_param('/robot_description')

    # Defining the two frames of reference wanted on the robot
    if arm == 'left':
        end_effector = 'left_gripper'
    else:
        end_effector = 'right_gripper'
          
    ik_solver = IK("base",
                   end_effector, urdf_string=urdf_str)

    lb, up = ik_solver.get_joint_limits() 

    # Setting lower and upper limits to the joints
    ik_solver.set_joint_limits([-1.70168, -2.147, -3.05418, -0.05, -3.059, -1.5708, -3.059], 
    	[1.70168, 1.047, 3.05418, 2.618, 3.059, 2.094, 3.059])

    if seed is None:
        seed_state = [0.0] * ik_solver.number_of_joints
    else:
        seed_state = seed
    #seed_state = [0.0] * ik_solver.number_of_joints
    #print(seed_state)
    #print(type(seed_state))

    # Inserting desired points for the solver to solve
    #sol = np.array(ik_solver.get_ik(seed_state, x, y, z, qx, qy, qz, qw)) # put these in as integers 
    #print(type(sol))
    #print(sol.shape)
    #print('Joint Angles:', sol)
    joint_angles = np.array(ik_solver.get_ik(seed_state, x, y, z, qx, qy, qz, qw,0.001,0.001,0.001,0.01,0.01,0.01))
    return joint_angles
    




if __name__ == '__main__':
    print('-----Testing-----')
    print('Desired position')
    x=0.7
    y=-0.15
    z=0.05
    Qx=-0.02495908
    Qy=0.99964940
    Qz=0.00737916
    Qw=0.00486450
    #seed_state = [0.998940637469,-0.12254586357,-1.38328641688,1.92753659386,0.821642479962,0.471855256714,0.746724780637]
    #print('x, y, z', x, y, z)
    #print('Qx, Qy, Qz, Qw', Qx, Qy, Qz, Qw)

    #for i in range(10):
    #    aaa = ikine(x, y, z, Qx, Qy, Qz, Qw, arm = 'left')
    #    x = x+0.01
    #    print(aaa)
    aaa = ikine(x, y, z, Qx, Qy, Qz, Qw, arm = 'left')
    print(aaa)
    print(type(aaa))
#    print(aaa)
#    bbb = aaa.tolist()
#    bbb = [float('%.8f'%b) for b in bbb]
#    print(bbb)

