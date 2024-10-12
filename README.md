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

```python
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

```python
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
```python
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

Run this function:
```python
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

Let the Taskspace Variable be:

$$
p_{0,e}^0 = [p_x, p_y, p_z]
$$

In controlling the RRR robotic arm, there are several positions in the configuration space that can lead to a Singularity, making it impossible to find a solution to the equation. The robot is considered to be in a state of Singularity when:

$$
\lVert \text{det}(J^*(q)) \rVert < \epsilon
$$

where $\epsilon = 0.001$, and $J^*(\cdot)$ is the reduced Jacobian matrix.

### Solution
To find the determinant of a matrix, it must be a square matrix. The Jacobian obtained earlier is a 6x3 matrix, so we need to reduce it to a 3x3 matrix in order to calculate the determinant by cut J_w.

```Python
def checkSingularityHW3(q:list[float])->bool:

    # get J from 'endEffectorJacobianHW3' Function
    J = endEffectorJacobianHW3(q)

    # Get reduced J -> J_v
    J_v = J[0:3]

    # find det of reduced Jv
    det_Jv = (
        J_v[0, 0] * (J_v[1, 1]*J_v[2, 2] - J_v[1, 2]*J_v[2, 1])
        - J_v[0, 1] * (J_v[1, 0]*J_v[2, 2] - J_v[1, 2]*J_v[2, 0])
        + J_v[0, 2] * (J_v[1, 0]*J_v[2, 1] - J_v[1, 1]*J_v[2, 0])
    )

    # define tolerance as 10^-3
    tolerance = 1e-3
    
    # return singularity
    print("Det of Jacobian from my code : ", abs(det_Jv))
    return abs(det_Jv) < tolerance
```
### Validation
Create a function that generates random joint angles and compares the det of reduced jacobian matrix in my own implementation and computed using the Robotics Toolbox in Python:

```Python
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
    print("Det of Jacobian from RTB     : ", np.linalg.norm((np.linalg.det(robot.jacob0(q)[:3]))))   
    Jacobian_MyCode = m.checkSingularityHW3(q)
    if (np.linalg.norm((np.linalg.det(robot.jacob0(q)[:3]))) < 0.001) ==  Jacobian_MyCode:
        print("Det of Jacobian from RTB is equal to Det of Jacobian from my code")
        print(f"Singularity of this taskspace is {Jacobian_MyCode}!!")
    else:
        print("Det of Jacobian from RTB is not equal to Det of Jacobian from my code")
```

Run this function:
```Python
Proof_checkSingularityHW3()
```

Result of the taskspace that is singularity:
```
Det of Jacobian from RTB     :  0.0006664243850487095
Det of Jacobian from my code :  0.0006664243850487117
Det of Jacobian from RTB is equal to Det of Jacobian from my code
Singularity of this taskspace is True!!
```

Result of the taskspace that is not singularity:
```
Det of Jacobian from RTB     :  0.010538854253145597
Det of Jacobian from my code :  0.010538854664914912
Det of Jacobian from RTB is equal to Det of Jacobian from my code
Singularity of this taskspace is False!!
```

## Question 3
This setup involves using a Force Sensor model FT300, which is capable of measuring forces and torques in three dimensions. The sensor is installed at the end-effector of a manipulator with an RRR configuration, enabling the measurement of force and torque values as follows:

$$
\begin{bmatrix}
\textit{moment}(n^e) \\
\textit{force}(f^e)
\end{bmatrix}
$$

Write the function to determine the effort exerted by each joint when an external wrench acts on the end-effector in the reference frame.
### Solution
The joint effort of each joint can be calculated using following equations:

$$
\tau = J^T W
$$

Where:

- $\tau$ is the effort that affects the Manipulator, represented as $\mathbb{R}^{3 \times 1}$.
- $J^T$ is the Transposed Jacobian of the Manipulator, represented as $\mathbb{R}^{3 \times 6}$.
- $W$ is the Wrench that acts on the end-effector in the base frame, represented as $\mathbb{R}^{6 \times 1}$.

```Python
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
```

### Validation
Create a function that generates random joint angles and wrenches to compare the joint efforts in my own implementation and computed using the Robotics Toolbox in Python:

```Python
def Proof_computeEffortHW3():

    # Random each joint's angle
    q1 = random.uniform(0,pi)
    q2 = random.uniform(0,pi)
    q3 = random.uniform(0,pi)

    print(f"Config Space :\nq1 = {q1}, q2 = {q2}, q3 = {q3}")

    q = [q1,q2,q3]

    # Random wrench 
    w1 = random.uniform(0,pi)
    w2 = random.uniform(0,pi)
    w3 = random.uniform(0,pi)
    w4 = random.uniform(0,pi)
    w5 = random.uniform(0,pi)
    w6 = random.uniform(0,pi)

    print(f"At the center of end-effector : \nMx = {w1}, My = {w2}, Mz = {w3}\nFx = {w4}, Fy = {w5}, Fz = {w6}")

    w = [w1,w2,w3,w4,w5,w6]

    # use pay to get wrench at end-effector
    tau_roboticstool = robot.pay(w,J=robot.jacobe(q),frame = 1)

    # tau from 'FRA333_HW3_6556_6558'
    tau = m.computeEffortHW3(q,w)

    # compare
    if np.allclose(tau, -tau_roboticstool, atol=1e-6):
        print("Joint effort from RTB is equal to Joint effort from my code")
    else:
        print("Joint effort from RTB is not equal to Joint effort from my code") 
```

Run this function:
```python
Proof_computeEffortHW3()
```

Some example of result:
```
Config Space :
q1 = 0.24155096895482703, q2 = 1.793619901010365, q3 = 1.7500993807700733
At the center of end-effector :
Mx = 0.3760800656202432, My = 2.0405879780805374, Mz = 2.4474301198244195
Fx = 1.2517152445198672, Fy = 0.30387990297362, Fz = 2.2657397888852153
Joint effort from RTB is equal to Joint effort from my code
```
