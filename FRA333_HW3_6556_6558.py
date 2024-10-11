# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
import HW3_utils as ut
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np
from math import pi

'''
ชื่อ_รหัส(ธนวัฒน์_6461)
1.อนวัช_6556
2.อนุวิท_6558
3.
'''

#=============================================<คำตอบข้อ 1>======================================================#
#code here
def endEffectorJacobianHW3(q:list[float])->list[float]:

    # get Data from Foward Kinematic in HW3_utils.py
    R,P,R_e,p_e = ut.FKHW3(q)

    # get position each frames
    P_01 = P[:,0]
    P_02 = P[:,1]
    P_03 = P[:,2]
    P_0e = P[:,3]

    # get rotation each frames
    R_01 = R[:,:,0]
    R_02 = R[:,:,1]
    R_03 = R[:,:,2]
    R_0e = R[:,:,3]

    # make a z by Rotation dot [0,0,1]
    z1 = R_01 @ np.array([0.0, 0.0, 1.0])
    z2 = R_02 @ np.array([0.0, 0.0, 1.0])
    z3 = R_03 @ np.array([0.0, 0.0, 1.0])

    # Get Velocity by z cross P_0e - P_0x and reshape matrix to 3 x 1
    J_v1 = np.cross(z1, P_0e - P_01).reshape(3, 1)
    J_v2 = np.cross(z2, P_0e - P_02).reshape(3, 1)
    J_v3 = np.cross(z3, P_0e - P_03).reshape(3, 1)

    # Get Angular velocity by reshape matrix z to 3 x 1
    J_w1 = z1.reshape(3, 1)
    J_w2 = z2.reshape(3, 1)
    J_w3 = z3.reshape(3, 1)

    # concatenate J_v and J_w to Jacobian matrix complete 
    J_v = np.concatenate((J_v1, J_v2, J_v3), axis=1)
    J_w = np.concatenate((J_w1, J_w2, J_w3), axis=1)
    J = np.concatenate((J_v, J_w), axis=0)

    # return jacobian
    return J

#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:

    # get J from 'endEffectorJacobianHW3' Function
    J = endEffectorJacobianHW3(q)

    # Get J_v 
    J_v = J[0:3]

    # det Jv
    det_Jv = (
        J_v[0, 0] * (J_v[1, 1]*J_v[2, 2] - J_v[1, 2]*J_v[2, 1])
        - J_v[0, 1] * (J_v[1, 0]*J_v[2, 2] - J_v[1, 2]*J_v[2, 0])
        + J_v[0, 2] * (J_v[1, 0]*J_v[2, 1] - J_v[1, 1]*J_v[2, 0])
    )

    # define tolerance as 10^-3
    tolerance = 1e-3
    
    # return singularity
    return abs(det_Jv) < tolerance

    
#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here

def computeEffortHW3(q: list[float], w: list[float]) -> list[float]:

    # get Data from Foward Kinematic in HW3_utils.py
    R,P,R_e,p_e = ut.FKHW3(q)

    # split wrench into force and moment of end-effector frame  
    force_e = np.array(w[:3])   
    moment_e = np.array(w[3:])  

    # Dot Rotation matrix of end-effector frame with force and moment to get force and moment at frame 0
    force_0 = R_e @ force_e
    moment_0 = R_e @ moment_e

    # and conacate it back to get wrench at frame 0
    w_0 = np.concatenate((force_0, moment_0), axis=0)

    # get J from 'endEffectorJacobianHW3' Function    
    J = endEffectorJacobianHW3(q)

    # Transpose Matrix J
    J_t = np.array(J).T

    # Dot J_t with wrench at frame 0 to get tau    
    return J_t @ w_0

#==============================================================================================================#
# print(endEffectorJacobianHW3([0.0,-pi/2,-0.2]))
# print("Singularity :", checkSingularityHW3([0.0,-pi/2,-0.2]))
# print("Effort",computeEffortHW3([0.0,-pi/2,-0.2], [1,1,5,1,2,1]))
