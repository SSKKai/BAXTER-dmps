import roslib
import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import time, math




class FKine(object):
    def __init__(self):
        rospy.init_node('baxter_test')
        #arm= RIGHT
        RIGHT=0
        LEFT=1
        self.RIGHT = RIGHT
        self.LEFT = LEFT
        self.limbs= [None,None]
        self.limbs[self.RIGHT]= baxter_interface.Limb(self.LRTostr(self.RIGHT))
        self.limbs[self.LEFT]=  baxter_interface.Limb(self.LRTostr(self.LEFT))
        self.kin= [None,None]
        self.kin[RIGHT]= baxter_kinematics(self.LRTostr(self.RIGHT))
        self.kin[LEFT]=  baxter_kinematics(self.LRTostr(self.LEFT))

        self.joint_names= [[],[]]
        #joint_names[RIGHT]= ['right_'+joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
        #joint_names[LEFT]=  ['left_' +joint for joint in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
        self.joint_names[RIGHT]= self.limbs[RIGHT].joint_names()
        self.joint_names[LEFT]=  self.limbs[LEFT].joint_names()
    
    def LRTostr(self,whicharm):
        if whicharm==self.RIGHT: return 'right'
        if whicharm==self.LEFT:  return 'left'
        return None
    
    def fkine(self, joint_angles, arm = 'left'):
        if arm == 'right':
            angles= {joint:joint_angles[j] for j,joint in enumerate(self.joint_names[0])}  #Deserialize
            x= self.kin[0].forward_position_kinematics(joint_values=angles)
        elif arm == 'left':
            angles= {joint:joint_angles[j] for j,joint in enumerate(self.joint_names[1])}  #Deserialize
            x= self.kin[1].forward_position_kinematics(joint_values=angles)
        else:
            x = 'wrong arm'
        
        #print(x)
        return x





if __name__ == "__main__":
    q0=[ 0.40, 0.02,  0.05, 1.51,  1.05, 0.18, -0.41]
    q1=[1.01030734186,-0.106999307635,-1.3809397897,1.93350718072,0.830992280383,0.457187212415,0.763579943348]
    #q1 = [-0.40, 0.02, -0.05, 1.51, -1.05, 0.18,  0.41]

    fk = FKine()
    pos_r = fk.fkine(joint_angles=q0,arm = 'right')
    pos_l = fk.fkine(joint_angles=q1,arm = 'left')
    print(pos_l)

