# FRA333 Homework Assignment 3: Static Force
### **Creator**
> 65340500056 A. ANAWACH

> 65340500058 I. ANUWIT

## Objective
> This homework is designed to enable students to apply their knowledge of differential kinematics of a 3-DOF manipulator.

## 3-DOF Manipulator (RRR Robot) 
<p align="center">
  <img width=max alt="pic1" src="https://github.com/user-attachments/assets/f2f1f785-bab8-4585-afa5-fe5f4903c512">
</p>
  
### Robot Modeling
> The RRR robot can be modeled using Modified Denavit-Hartenberg (MDH) parameters in the Robotics Toolbox for Python.

```
import HW3_utils as ut
import FRA333_HW3_6556_6558 as m
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np
from math import pi

d_1 = 0.0892
a_2 = -0.425
a_3 = -0.39243
d_4 = 0.109
d_5 = 0.093
d_6 = 0.082

T_3_e = SE3.Tx(a_3) @ SE3.Tz(d_4) @ SE3.Rx(90, unit="deg") @ SE3.Tz(d_5) @ SE3.Rz(90, unit="deg") @ SE3.Rx(-90, unit="deg") @ SE3.Tz(d_6)

# make a DH-Parameter
robot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d=d_1, offset=pi), # joint 1
        rtb.RevoluteMDH(alpha=pi/2), # joint 2
        rtb.RevoluteMDH(a=a_2), # joint 3
    ]
    ,tool = T_3_e
    ,name = "3DOF_Robot"
)
```

## Question 1
Please write a function to compute the Jacobian of this robot within the following function.

### Solution
Using a geomatric jacobian:

$$
J_{linear,i} = \hat{Z}_i \times (P_e^0 - P_i^0)
$$

$$
J_{angular,i} = \hat{Z}_i
$$

The Jacobian matrix \( J \) is then given by:

$$
J = \begin{bmatrix}
J_{linear} \\
J_{angular}
\end{bmatrix}
$$

From these equations, the code can be written as follows:

```
def endEffectorJacobianHW3(q:list[float])->list[float]:

    # get Data from Foward Kinematic in HW3_utils.py
    R,P,R_e,p_e = ut.FKHW3(q)

    # get position each frames relative to frame 0
    P_01 = P[:,0]
    P_02 = P[:,1]
    P_03 = P[:,2]
    P_0e = P[:,3]

    # get rotation each frames relative to frame 0
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
```
### Validation
Create a function that generates random joint angles and compares the Jacobian in my own implementation and computed using the Robotics Toolbox in Python:
```
def Proof_endEffectorJacobianHW3():

    # Random each joint's angle
    q1 = random.uniform(0,pi)
    q2 = random.uniform(0,pi)
    q3 = random.uniform(0,pi)

    q = [q1,q2,q3]
    print("Jacobian Matrix from Robotics Toolbox in python :\n", robot.jacob0(q),"\n\n",
          "Jacobian Matrix from FRA333_HW3_6556_6558 :\n", m.endEffectorJacobianHW3(q))

    # compare two methods with tolerance of 10^-6
    print(f"\nq1 = {q1}, q2 = {q2}, q3 = {q3}")
    if (np.allclose(robot.jacob0(q), m.endEffectorJacobianHW3(q), atol=1e-6)):
        print("Jacobian Matrix from Robotics Toolbox in python is equal to Jacobian Matrix from FRA333_HW3_6556_6558.")
    else:
        print("Jacobian Matrix from Robotics Toolbox in python is not equal to Jacobian Matrix from FRA333_HW3_6556_6558.")
```

Run Proof_endEffectorJacobianHW3:
```
Proof_endEffectorJacobianHW3()
```

Some example of result:
```
Jacobian Matrix from Robotics Toolbox in python :
 [[ 2.43746008e-01  3.07953982e-01 -5.84768069e-02]
 [ 6.45692571e-01 -6.31823413e-02  1.19975769e-02]
 [ 6.07153217e-18  6.81505696e-01  4.79759675e-01]
 [-9.71445147e-17 -2.00981670e-01 -2.00981670e-01]
 [ 8.67361738e-18 -9.79595002e-01 -9.79595002e-01]
 [ 1.00000000e+00  6.12323400e-17  6.12323400e-17]]

 Jacobian Matrix from FRA333_HW3_6556_6558 :
 [[ 2.43746008e-01  3.07953983e-01 -5.84768071e-02]
 [ 6.45692571e-01 -6.31823432e-02  1.19975772e-02]
 [-0.00000000e+00  6.81505699e-01  4.79759678e-01]
 [ 0.00000000e+00 -2.00981677e-01 -2.00981677e-01]
 [ 0.00000000e+00 -9.79595006e-01 -9.79595006e-01]
 [ 1.00000000e+00  6.12323426e-17  6.12323426e-17]]

q1 = 2.939232717093267, q2 = 2.065415519456799, q3 = 1.0063972889247221
Jacobian Matrix from Robotics Toolbox in python is equal to Jacobian Matrix from FRA333_HW3_6556_6558.
```

## Question 2

## Question 3
