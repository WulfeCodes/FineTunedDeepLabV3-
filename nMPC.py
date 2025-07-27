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
    fwd_slipAngle=steering_angle - ca.arctan((V_y + L_f * r)/V_x)
    rear_slipAngle = - ca.arctan((V_y - L_r * r)/V_x)
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

    X_DOT = ca.vertcat(x_dot_g, y_dot_g, nonInertial_xAccel, nonInertial_yAccel, r, r_dot)
    
    return X_DOT

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
    Q = ca.diag([10, 10, 1, 1, 1, 1])  # State cost
    R = ca.diag([1, 1])                # Input cost


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
    #semi major and minor for x,y in ellipse
    a  = 20
    b = 10
    s = np.linspace(0,2 * np.pi, 500)
    x_ref = a * np.cos(s)
    y_ref = b * np.sin(s)

    #make some heading angle
    #dy/ds, #dx/ds
    dx = - a * np.sin(s)
    dy = b * np.cos(s)
    angle_ref=np.arctan2(dy,dx)
    #normalize tangent vector for longitudinal error
    T_hat = (1/np.sqrt((dx)**2 + (dy)**2)) * np.array([dx,dy])
    N_hat = (1/np.sqrt((dx)**2 + (dy)**2)) * np.array([-dy,dx])
    #TODO make this continuous 
    angle_ref = np.arctan2(T_hat[1],T_hat[0])
    #plt.figure(figsize=(6, 6))
    #plt.plot(x_ref, y_ref, label="Reference Trajectory (Ellipse)")
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.title("Elliptical Raceline")
    #plt.axis('equal')
    #plt.grid(True)
    #plt.legend()
    #plt.show()
    #plot
    return x_ref,y_ref,s,a,b

    #frenet definition
    #reference point is always a reference point ahead delS or delt
def createStartVector(arcStart,a,b):
    xStart = a * np.cos(arcStart)
    yStart = b * np.sin(arcStart)
    vX = -a * np.sin(arcStart)
    vY = b * np.cos(arcStart)
    v = np.sqrt(np.square(vX)+np.square(vY))

    aX = - a * np.cos(arcStart)
    aY = - b * np.cos(arcStart)

    yaw = np.arctan2(vY,vX)
    Tangent_hat = 1/np.sqrt(vY**2 + vX**2) *  np.array([vX,vY])
    dTangent_hat = 1/np.sqrt(vY**2 + vX**2) *  np.array([aX,aY])
    k=Tangent_hat[0] * dTangent_hat[1] - dTangent_hat[0] * Tangent_hat[1]
    r = k * v
    return ca.vertcat(xStart,yStart,vX,vY,yaw,r)

def computeDesiredVector(arc, a,b,Xnext):
    xDes = a * np.cos(arc)
    yDes = b * np.sin(arc)
    vX = -a * np.sin(arc)
    vY = b * np.cos(arc)
    aX = - a * np.cos(arc)
    aY = - b * np.cos(arc)

    v = np.sqrt(np.square(vX)+np.square(vY))
    tanVector = np.array([vX,vY])
    dtanVector = np.array([aX,aY])
    normalVector = np.array([-vY,vX])

    k=Tangent_hat[0] * dTangent_hat[1] - dTangent_hat[0] * Tangent_hat[1]
    r = k * v
    side_slip = np.arctan2(vY,vX)
    Tangent_hat = 1/np.sqrt(vY**2 + vX**2) *  tanVector
    dTangent_hat = 1/np.sqrt(vY**2 + vX**2) * dtanVector 

    
    xError = Xnext[0]-xDes
    yError = Xnext[1]-yDes
    vXerror = Xnext[2]-vX
    vYerror = Xnext[3]-vY
    yaw_err = Xnext[4] - side_slip

    prjctdLongerr=tanVector @ np.array([xError,yError]).T
    prjctdLaterr = normalVector @ np.array([xError,yError]).T


    aX = - a * np.cos(arc)
    aY = - b * np.cos(arc)

    yaw = np.arctan2(vY,vX)
    Tangent_hat = 1/np.sqrt(vY**2 + vX**2) *  np.array([vX,vY])
    dTangent_hat = 1/np.sqrt(vY**2 + vX**2) *  np.array([aX,aY])
    k=Tangent_hat[0] * dTangent_hat[1] - dTangent_hat[0] * Tangent_hat[1]
    r = k * v
    
    rerror = Xnext[5] - r
    return ca.vertcat(prjctdLongerr,prjctdLaterr,vXerror,vYerror,yaw_err,rerror)
    #compute velocity by dt, or by velocity at current 

def create_optimizer_function(f_dyn,dt,time_horizon):
    Q = ca.diag([10, 10, 1, 1, 1, 1])  # State cost
    R = ca.diag([1, 1])                # Input cost

    opti = Opti()


    x_opti = opti.variable(6,time_horizon+1)
    u_opti = opti.variable(2,time_horizon)
    
    opti.subject_to(opti.bounded(-3,u_opti[0,:],2))
    opti.subject_to(opti.bounded(-0.5,u_opti[1,:],.5))
    
    #SHOULD WE CONSTRAIN END STATE AS WELL?
    x0 = opti.parameter(6)
    opti.subject_to(x_opti[:, 0] == x0)

    x_des = opti.parameter(6,time_horizon+1)
    u_des = opti.parameter(2,time_horizon)

    cost  = 0

    for i in range(time_horizon):
        cost+=x_des[:,i].T @ Q @ x_des[:,i] + u_des[:,i].T @ R @ u_des[:,i]
        xNext = step(x_opti[:,i], u_opti[:,i], f_dyn, dt)
        opti.subject_to(x_opti[:,i+1]==xNext)

    cost += x_des[:, -1].T @ Q @ x_des[:, -1]


    solver_function = opti.to_function(
            "mpc_solver", 
            [x_des,u_des,x_opti[:,0]],u_opti)

    return solver_function
    #no set actual values yet, only in error params

def sim(f_dyn,dt):

    time_horizon = 10

    x_coor_ref, y_coord_ref,arc_ref,a,b =create_trackCoordinates()
    xCurr=createStartVector(arc_ref[0],a,b)

    nextX=createStartVector(arc_ref[10],a,b)

    num_des_x = np.zeros((6,time_horizon+1))
    num_des_u = np.zeros((2,time_horizon))



    optimizer=create_optimizer_function(f_dyn,dt,time_horizon)
    
    for i in range(40):

        xCurrSim = xCurr
        for j in range(time_horizon):
            
            if i == 0:
                num_des_u[:,j]=np.array([0,0]).T
                num_des_x[:,j]=computeDesiredVector(arc_ref[((i+1)*10)*(j+1)],a,b,xCurr)
                xCurrSim = step(xCurrSim,num_des_u[:,j],f_dyn,dt)

            else:
                num_des_u[:,j]=U_actual[:,0]
                num_des_x[:,j]=computeDesiredVector(arc_ref[((i+1)*10)*(j+1)],a,b,xCurr)
                xCurrSim=step(xCurrSim,U_actual[:,0],f_dyn,dt)
        
        U_actual=optimizer(num_des_x,num_des_u,xCurr)
        xCurr = step(xCurr,U_actual[:,0],f_dyn,dt)


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

    X_ref = ca.DM([5, 5, 0, 0, 0, 0])  # Desired state
    U_ref = ca.DM([0, 0])             # Desired control
    # Control inputs
    steering_angle = 0.0  # rad
    control_accel = 0.5  # m/s²

    # Forces
    F_xf = 0.0

    X,U=create_symbolicVectors()
    X_DOT=create_stateSpace(I_z,mass,L_f, L_r,B,C,D,
                      X,U, F_xf)
    f_dyn = ca.Function("f_dyn", [X, U], [X_DOT])
    x_next=step(X,U,f_dyn,dt)
    simulation(f_dyn,dt)

if __name__ == '__main__':
    main()
