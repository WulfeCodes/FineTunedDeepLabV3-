import numpy as np
import casadi as ca
from casadi import *
import matplotlib.pyplot as plt
#TODO Q and R tuning analysis with constraints
#TODO implement acados/qpoases/odys/forces pro
#casadi+acodos
def create_stateSpace(I_z,mass,L_f, L_r,B,C,D,
                      X,U, F_xf):
    
    x,y, V_x, V_y, yaw, r = X[0], X[1], X[2], X[3], X[4], X[5]
    steering_angle, control_accel = U[0], U[1]
    fwd_slipAngle=steering_angle - ca.arctan(V_y + L_f * r/V_x)
    rear_slipAngle = - ca.arctan(V_y - L_r * r/V_x)
    #PacejkaMagicFormula
    #B = stiffness
    #C = shape factor
    #D = peak lateral force 
    F_yr = D * ca.sin(C * ca.arctan(B * rear_slipAngle))
    F_yf = D * ca.sin(C * ca.arctan(B * fwd_slipAngle))
    F_xr = control_accel * mass


    r_dot = 1/I_z * (L_f * (F_yf * ca.cos(steering_angle)) - L_r * F_yr)

    x_dot_g  = V_x * ca.cos(yaw) - V_y * ca.sin(yaw)
    y_dot_g = V_x * ca.sin(yaw) + V_y * ca.cos(yaw)
    
    nonInertial_xAccel = 1/mass *(F_xf * ca.cos(steering_angle) - F_yf * ca.sin(steering_angle) + F_xr) + r* V_y
    nonInertial_yAccel= 1/mass * (F_yf * ca.cos(steering_angle) + F_xf * ca.sin(steering_angle) + F_yr) - r * V_x

    f = ca.vertcat(x_dot_g, y_dot_g, nonInertial_xAccel, nonInertial_yAccel, r, r_dot)
    
    return f

def create_symbolicVectors():
    # Number of states and inputs
    nx = 6
    nu = 2

    # Define symbolic state and input vectors
    x = ca.SX.sym('x')     # x position
    y = ca.SX.sym('y')     # y position
    Vx = ca.SX.sym('Vx')   # longitudinal velocity
    Vy = ca.SX.sym('Vy')   # lateral velocity
    yaw = ca.SX.sym('yaw') # heading
    r = ca.SX.sym('r')     # yaw rate

    steer = ca.SX.sym('steer')  # steering angle
    a = ca.SX.sym('a')          # acceleration (throttle/brake)

    # Combine into full state and control vectors
    X = ca.vertcat(x, y, Vx, Vy, yaw, r)
    U = ca.vertcat(steer, a)
    return X, U

def step(X, U, f_dyn, dt):
    return X + dt * f_dyn(X, U)

def create_costFunction(X,U,X_ref, U_ref, Q,R):
    state_error = X - X_ref
    #u_ref should typically be defined as U_prev
    ip_error = U - U_ref
    #no terminal cost here 
    cost = ca.mtimes([state_error.T,Q, state_error]) + ca.mtimes([ip_error.T,R,ip_error])
    #ca.Function is parameterized by list of input deps and output
    #creates the computation graph linking them
    #done here and with the state space function
    return ca.Function("cost_func",[X,U],[cost])

    #how should Q and R be tuned?

def create_trackCoordinates():
    pass
    #frenet definition

def main():
    mass = 12
    length = 12
    width = 12
    I_z=mass/12 *(length**2  +width**2)

    #distance from front and rear axle to COM
    #simplified assumption by only longitudinal axis
    L_f = 6
    L_r = 6

    #tire model parameters
    B = 10
    C= 1.3
    D = 58.86
    #B = stiffness
    #C = shape factor
    #D = peak lateral force

    #initial conditions:
    V_x = 1.0  # m/s
    V_y = 0.0
    yaw = 0.0
    r = 0.0
    dt = .01

    #TODO these need to be changed per the problem statemt
    Q = ca.diag([10, 10, 1, 1, 1, 1])  # State cost
    R = ca.diag([1, 1])                # Input cost
    X_ref = ca.DM([5, 5, 0, 0, 0, 0])  # Desired state
    U_ref = ca.DM([0, 0])             # Desired control
    # Control inputs
    steering_angle = 0.0  # rad
    control_accel = 0.5  # m/s²

    # Forces
    F_xf = 0.0

    X,U=create_symbolicVectors()
    f=create_stateSpace(I_z,mass,L_f, L_r,B,C,D,
                      X,U, F_xf)
    f_dyn = ca.Function("f_dyn", [X, U], [f])
    x_next=step(X,U,f_dyn,dt)
    cost_function=create_costFunction(X,U,X_ref, U_ref, Q,R)

    #time horizon is defined to be 10 here, 
    # 40 would be around upper bound of irl case
    opti=Opti()
    time_horizon = 10
    u_opti=opti.variable(2,time_horizon)
    x_opti = opti.variable(6,time_horizon+1)
    #constraint definitions
    #steering angle and acceleration bounds
        
    opti.subject_to(opti.bounded(-3,u_opti[0,:],2))
    opti.subject_to(opti.bounded(-0.5,u_opti[1,:],.5))

    #how should state constraints be defined?
    print("successfully created optimizer variable")

    traj_coords=create_trackCoordinates()

    #simulation loop:
if __name__ == '__main__':
    main()
