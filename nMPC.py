import numpy as np
import casadi as ca
from casadi import *
import matplotlib.pyplot as plt
import torch
#TODO Q and R tuning analysis with constraints
#TODO fix iterator
#TODO implement acados/qpoases/odys/forces pro
#casadi+acodos

def plot_loss(loss_values, title="Training Loss", xlabel="Iteration", ylabel="Loss", 
              figsize=(10, 6), color='blue', grid=True, save_path=None):
    """
    Plot loss values over iterations/epochs
    
    Parameters:
    -----------
    loss_values : array-like
        Array of loss values to plot
    title : str
        Title for the plot
    xlabel : str  
        Label for x-axis
    ylabel : str
        Label for y-axis
    figsize : tuple
        Figure size (width, height)
    color : str
        Color of the line
    grid : bool
        Whether to show grid
    save_path : str, optional
        Path to save the plot (e.g., 'loss_plot.png')
    """
    
    # Ensure loss_values is 1D
    loss_values = np.squeeze(np.array(loss_values))
    if loss_values.ndim > 1:
        loss_values = loss_values.flatten()
    
    plt.figure(figsize=figsize)
    
    # Create x-axis (iteration numbers)
    iterations = range(len(loss_values))
    
    # Plot the loss
    plt.plot(iterations, loss_values, color=color, linewidth=2, marker='o', markersize=3)
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if grid:
        plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    min_loss = np.min(loss_values)
    min_idx = np.argmin(loss_values)
    final_loss = loss_values[-1]
    
    stats_text = f'Min Loss: {min_loss:.4f} (iter {min_idx})\nFinal Loss: {final_loss:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()



def create_stateSpace(I_z,mass,L_f, L_r,B,C,D,
                      X,U, F_xf):
    
    x,y, V_x, V_y, yaw, r = X[0], X[1], X[2], X[3], X[4], X[5]

    V_x_safe = ca.fmax(ca.fabs(V_x),1e-6)

    steering_angle, control_accel = U[0], U[1]
    fwd_slipAngle=steering_angle - ca.arctan((V_y + L_f * r)/V_x_safe)
    rear_slipAngle = - ca.arctan((V_y - L_r * r)/V_x_safe)
    #PacejkaMagicFormula
    #B = stiffness
    #C = shape factor
    #D = peak lateral force 
    F_yr = D * ca.sin(C * ca.arctan(B * rear_slipAngle))
    F_yf = D * ca.sin(C * ca.arctan(B * fwd_slipAngle))
    F_xr = control_accel * mass


    r_dot = 1/I_z * (L_f * (F_yf * ca.cos(steering_angle)) - L_r * F_yr)

    x_dot_g  = V_x_safe * ca.cos(yaw) - V_y * ca.sin(yaw)
    y_dot_g = V_x_safe * ca.sin(yaw) + V_y * ca.cos(yaw)
    
    nonInertial_xAccel = 1/mass *(F_xf * ca.cos(steering_angle) - F_yf * ca.sin(steering_angle) + F_xr) + r* V_y
    nonInertial_yAccel= 1/mass * (F_yf * ca.cos(steering_angle) + F_xf * ca.sin(steering_angle) + F_yr) - r * V_x_safe

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
    # X_plusone=X + dt * f_dyn(X, U)
    # return X_plusone
    k1 = f_dyn(X, U)
    k2 = f_dyn(X + 0.5 * dt * k1, U)
    k3 = f_dyn(X + 0.5 * dt * k2, U)
    k4 = f_dyn(X + dt * k3, U)
    return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)



def create_costFUnction(X,U,X_ref, U_ref, Q,R):
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
    s = np.linspace(0,2 * np.pi, 5000)
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

def wrap_to_pi(angle):
    return ca.atan2(ca.sin(angle), ca.cos(angle))

def computeDesiredVector(arc, a,b,Xnext):
    xDes = a * np.cos(arc)
    yDes = b * np.sin(arc)
    vX = -a * np.sin(arc)
    vY = b * np.cos(arc)
    aX = - a * np.cos(arc)
    aY = - b * np.cos(arc)

    v = np.sqrt(np.square(vX)+np.square(vY))

    tanVector = np.zeros((2,1))
    dtanVector = np.zeros((2,1))
    coordErrVector = np.zeros((2,1))
    dtanVector = np.zeros((2,1))
    normalRowVector = np.zeros((1,2))
    tanRowVector = np.zeros((1,2))
    dtanRowVector = np.zeros((1,2))

    #TODO This needs to be optimized bad, too many creations
    tanVector[0,0],tanVector[1,0]= vX, vY
    tanRowVector[0,0],tanRowVector[0,1] = vX, vY
    dtanRowVector[0,0],dtanRowVector[0,1] = aX, aY
    dtanVector[0,0],dtanVector[1,0]= aX, aY
    normalRowVector[0,0],normalRowVector[0,1]= -vY,vX
    
    side_slip = np.arctan2(vY,vX)
    Tangent_hat = 1/np.sqrt(vY**2 + vX**2) *  tanVector
    dTangent_hat = 1/np.sqrt(vY**2 + vX**2) * dtanVector 
    dTangent_hatRow = 1/np.sqrt(vY**2 + vX**2) * dtanRowVector
    Tangent_hatRow = 1/np.sqrt(vY**2 + vX**2) * tanRowVector
    normalRowVector = 1/np.sqrt(vY**2 + vX**2) * normalRowVector

    coordErrVector[0,0],coordErrVector[1,0] = Xnext[0]-xDes, Xnext[1]-yDes

    vXerror = Xnext[2]-vX
    vYerror = Xnext[3]-vY
    yaw_err = wrap_to_pi(Xnext[4] - side_slip)

# Convert numpy arrays to tensors

    prjctdLongerr = Tangent_hatRow @ coordErrVector
    prjctdLaterr = normalRowVector @ coordErrVector
    aX = - a * np.cos(arc)
    aY = - b * np.cos(arc)

    yaw = np.arctan2(vY,vX)
    k=Tangent_hat[0,0] * dTangent_hat[1,0] - dTangent_hat[0,0] * Tangent_hat[1,0]
    r = k * v
    print("shapes computed")
    
    return ca.vertcat(xDes,yDes,vX,vY,side_slip,r),prjctdLongerr,prjctdLaterr
    #compute velocity by dt, or by velocity at current 

def create_optimizer_function(f_dyn,dt,time_horizon):
    Q = ca.diag([10, 10, 1, 1, 1, 1])  # State cost
    R = ca.diag([1, 1])                # Input cost

    opti = Opti()

    x_opti = opti.variable(6,time_horizon+1)
    u_opti = opti.variable(2,time_horizon)
    
    opti.subject_to(opti.bounded(-3,u_opti[0,:],2))
    opti.subject_to(opti.bounded(-0.5,u_opti[1,:],.5))
    
    x0 = opti.parameter(6)
    opti.subject_to(x_opti[:, 0] == x0)

    x_des = opti.parameter(6,time_horizon+1)
    u_des = opti.parameter(2,time_horizon)

    cost  = 0


    for i in range(time_horizon):
        state_err = x_opti[:,i] - x_des[:,i]
        #state_err[:2,]
        input_err = u_opti[:,i] - u_des[:,i]
        

        cost += state_err.T @ Q @ state_err + input_err.T @ R @ input_err        
        xNext = step(x_opti[:,i], u_opti[:,i], f_dyn, dt)
        opti.subject_to(x_opti[:,i+1]==xNext)

        final_err = x_opti[:, -1] - x_des[:, -1]
        cost += final_err.T @ Q @ final_err

    opti.minimize(cost)
    opti.solver('ipopt')

    solver_function = opti.to_function('mpc_solver',[x_opti,x_des,u_opti,u_des,x0],[u_opti])
    print("saved comp function")
    return solver_function
    #no set actual values yet, only in error params

def sim(f_dyn,dt):
    lossXArr = []
    lossYArr = []

    time_horizon = 10

    x_coor_ref, y_coord_ref,arc_ref,a,b =create_trackCoordinates()
    xCurr=createStartVector(arc_ref[0],a,b)

    nextX=createStartVector(arc_ref[10],a,b)

    num_des_x = np.zeros((6,time_horizon+1))
    num_des_u = np.zeros((2,time_horizon))

    num_opti_x = np.zeros((6,time_horizon+1))
    U_actual = np.zeros((2,time_horizon))

    optimizer=create_optimizer_function(f_dyn,dt,time_horizon)
    err = 0
    for i in range(400):

        xCurrSim = xCurr
        for j in range(time_horizon):
            
            if i == 0:
                if j == 0:
                    U_prev=np.array([0,0]).T

                num_des_u[:,j]=np.array([0,0]).T
                U_actual[:,j]=np.array([0,0]).T
                currErrVector,Xerr,Yerr=computeDesiredVector(arc_ref[((i+1)*10)+((j+1)*10)],a,b,xCurr)
                num_opti_x[:,j] = xCurr.full().flatten()
                num_des_x[:,j]=currErrVector.full().flatten()
                lossXArr.append(Xerr)
                lossYArr.append(Yerr)

                xCurrSim = step(xCurrSim,num_des_u[:,j],f_dyn,dt)
            else:
                if j ==0:
                    num_des_u[:,j]=U_prev
                    U_prev = U_actual[:,0].full().flatten()

                else:
                    num_des_u[:,j]=U_actual[:,j-1].full().flatten()

                currErrVector,Xerr,Yerr=computeDesiredVector(arc_ref[((i+1)*10)+((j+1)*10)],a,b,xCurr)
                num_opti_x[:,j] = xCurr.full().flatten()
                num_des_x[:,j]=currErrVector.full().flatten()
                lossXArr.append(Xerr)
                lossYArr.append(Yerr)
                xCurrSim=step(xCurrSim,U_actual[:,j],f_dyn,dt)
                
        print(f"ran {i +1} times")
        U_actual=optimizer(num_opti_x,num_des_x,U_actual,num_des_u,xCurr)
        xCurr = step(xCurr,U_actual[:,0],f_dyn,dt)
    return lossXArr,lossYArr

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
    B = 5
    C= 1.2
    D = 30
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
    lossXArr=[]
    lossYArr=[]

    # Control inputs

    # Forces
    F_xf = 0.0

    X,U=create_symbolicVectors()
    X_DOT=create_stateSpace(I_z,mass,L_f, L_r,B,C,D,
                      X,U, F_xf)
    f_dyn = ca.Function("f_dyn", [X, U], [X_DOT])
    lossXArr,lossYArr=sim(f_dyn,dt)
    plot_loss(lossXArr,"Longitudinal Positional Diff")
    plot_loss(lossYArr,'Lateral Positional Diff')

if __name__ == '__main__':
    main()
