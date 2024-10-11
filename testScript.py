# file สำหรับตรวจคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
import HW3_utils as ut
import FRA333_HW3_6556_6558 as m
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np
from math import pi
import random


'''
ชื่อ_รหัส(ธนวัฒน์_6461)
1.อนวัช_6556
2.อนุวิท_6558
3.
'''
d_1 = 0.0892
a_2 = -0.425
a_3 = -0.39243
d_4 = 0.109
d_5 = 0.093
d_6 = 0.082

# make a DH-Parameter
robot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d=d_1, offset=pi), # joint 1
        rtb.RevoluteMDH(alpha=pi/2), # joint 2
        rtb.RevoluteMDH(a=a_2), # joint 3
    ]
    ,tool = SE3.Tx(a_3) @ SE3.Tz(d_4) @ SE3.Rx(90, unit="deg") @ SE3.Tz(d_5) @ SE3.Rz(90, unit="deg") @ SE3.Rx(-90, unit="deg") @ SE3.Tz(d_6) 
    ,name = "3DOF_Robot"
)

#===========================================<ตรวจคำตอบข้อ 1>====================================================#
#code here
def Proof_endEffectorJacobianHW3()->bool:

    # Random each joint's angle
    q1 = random.uniform(0,pi)
    q2 = random.uniform(0,pi)
    q3 = random.uniform(0,pi)

    print(q1,q2,q3)

    q = [q1,q2,q3]
    print(robot.jacob0(q),"\n\n", m.endEffectorJacobianHW3(q))

    # compare two methods with tolerance of 10^-6
    return np.allclose(robot.jacob0(q), m.endEffectorJacobianHW3(q), atol=1e-6)
    
#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 2>====================================================#
#code here
def Proof_checkSingularityHW3():

    # Random each joint's angle
    q1 = random.uniform(0,pi)
    q2 = random.uniform(0,pi)
    q3 = random.uniform(0,pi)

    #Singularity
    q = [0,0,-0.19] 

    #Random (uncomment this to use random value)
    # q = [q1,q2,q3] 
    
    # compare two methods that equal to each other   
    return (np.linalg.norm((np.linalg.det(robot.jacob0(q)[:3]))) < 0.001) ==  m.checkSingularityHW3(q)

#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 3>====================================================#
#code here
def Proof_computeEffortHW3():

    # Random each joint's angle
    q1 = random.uniform(0,pi)
    q2 = random.uniform(0,pi)
    q3 = random.uniform(0,pi)

    print(q1,q2,q3)

    q = [q1,q2,q3]

    # Random wrench 
    w1 = random.uniform(0,pi)
    w2 = random.uniform(0,pi)
    w3 = random.uniform(0,pi)
    w4 = random.uniform(0,pi)
    w5 = random.uniform(0,pi)
    w6 = random.uniform(0,pi)

    print(w1,w2,w3,w4,w5,w6)

    w = [w1,w2,w3,w4,w5,w6]

    # use pay to get wrench at end-effector
    tau_roboticstool = robot.pay(w,J=robot.jacobe(q),frame = 1)
    # tau from 'FRA333_HW3_6556_6558'
    tau = m.computeEffortHW3(q,w)

    #compare
    return np.allclose(tau, -tau_roboticstool, atol=1e-6)

#==============================================================================================================#

#test 
print(Proof_endEffectorJacobianHW3())
print(Proof_checkSingularityHW3())
print(Proof_computeEffortHW3())