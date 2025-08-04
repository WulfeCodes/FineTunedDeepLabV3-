import numpy as np
import casadi as ca
from casadi import *
import matplotlib.pyplot as plt
import torch
#TODO Q and R tuning analysis with constraints
#TODO normalization across system parameters
    #control bounds, car weight/length, pacejka
#TODO implement acados/qpoases/odys/forces pro
#TODO convert startvector velocities to body frame

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


def plotter(x_values, y_values, title):
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must be the same length.")
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_desired(actual_x, actual_y, desired_x, desired_y, title):
    # Safety check
    if not (len(actual_x) == len(actual_y) and len(desired_x) == len(desired_y)):
        raise ValueError("Coordinate lists must be of equal length.")

    plt.figure(figsize=(8, 6))
    
    # Plot paths
    plt.plot(actual_x, actual_y, 'bo-', label='Actual Path')    # blue line + circles
    plt.plot(desired_x, desired_y, 'r*-', label='Desired Path') # red line + stars
    
    # Plot start points
    plt.plot(actual_x[0], actual_y[0], 'go', markersize=10, label='Start (Actual)')
    plt.plot(desired_x[0], desired_y[0], 'g*', markersize=12, label='Start (Desired)')
    
    # Plot end points
    plt.plot(actual_x[-1], actual_y[-1], 'ks', markersize=10, label='End (Actual)')
    plt.plot(desired_x[-1], desired_y[-1], 'kX', markersize=12, label='End (Desired)')
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Maintains aspect ratio
    plt.tight_layout()
    plt.show()

def create_stateSpace(I_z,mass,L_f, L_r,max_force,C,slip_threshold,
                      X,U, F_xr):
    
    
    x,y, V_x, V_y, yaw, r = X[0], X[1], X[2], X[3], X[4], X[5]

    V_x_safe = ca.fmax(ca.fabs(V_x),.1)

    steering_angle, control_accel = U[0], U[1]
    fwd_slipAngle=steering_angle - ca.arctan((V_y + L_f * r)/V_x_safe)
    rear_slipAngle = -ca.arctan((V_y - L_r * r)/V_x_safe)
    #TanH saturation model
    saturation_slip = max_force/C

    F_yr = max_force * ca.tanh(rear_slipAngle/saturation_slip)
    F_yf = max_force * ca.tanh(fwd_slipAngle/saturation_slip)
    F_xf = control_accel * mass

    #=L_f * (F_yf * cos(steering_angle) - F_xf * sin(steering_angle)) 0 L_r * F_yr

    r_dot = 1/I_z * (L_f * (F_yf * cos(steering_angle) - F_xf * sin(steering_angle)) - L_r * F_yr)

    x_dot_g  = V_x_safe * ca.cos(yaw) - V_y * ca.sin(yaw)
    y_dot_g = V_x_safe * ca.sin(yaw) + V_y * ca.cos(yaw)
    
    nonInertial_xAccel = 1/mass *(F_xf * ca.cos(steering_angle) + F_yf * ca.sin(steering_angle) + F_xr) - r* V_y
    nonInertial_yAccel= 1/mass * (F_yf * ca.cos(steering_angle) - F_xf * ca.sin(steering_angle) + F_yr) + r * V_x_safe

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

def create_trackCoordinates():
    #semi major and minor for x,y in ellipse
    a  = 20
    b = 10
    s = np.linspace(0,2 * np.pi, 2000)
    x_ref = a * np.cos(s)
    y_ref = b * np.sin(s)
    for i,angle in enumerate(s):
        print(f"angle {i},{angle}")
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
    yaw = np.arctan2(vY,vX)

    v = np.sqrt(np.square(vX)+np.square(vY))
    vX_ = vX * np.cos(yaw) + np.sin(yaw) * vY
    vY_ = vX * -np.sin(yaw) + np.cos(yaw) * vY

    aX = - a * np.cos(arcStart)
    aY = - b * np.cos(arcStart)

    Tangent_hat = 1/np.sqrt(vY**2 + vX**2) *  np.array([vX,vY])
    dTangent_hat = 1/np.sqrt(vY**2 + vX**2) *  np.array([aX,aY])
    k=Tangent_hat[0] * dTangent_hat[1] - dTangent_hat[0] * Tangent_hat[1]
    r = k * v
    return ca.vertcat(xStart,yStart,vX_,vY_,yaw,r)

def wrap_to_pi(angle):
    return ca.atan2(ca.sin(angle), ca.cos(angle))

def computeDesiredVector(arc, a,b,Xnext):
    xDes = a * np.cos(arc)
    yDes = b * np.sin(arc)
    vX = -a * np.sin(arc)
    vY = b * np.cos(arc)
    aX = - a * np.cos(arc)
    aY = - b * np.cos(arc)
    yaw = np.arctan2(vY,vX)

    vX_ = vX * np.cos(yaw) + np.sin(yaw) * vY
    vY_ = vX * -np.sin(yaw) + np.cos(yaw) * vY
    vY = vY_
    vX = vX_

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
    
    yaw = np.arctan2(vY,vX)
    Tangent_hat = 1/np.sqrt(vY**2 + vX**2) *  tanVector
    dTangent_hat = 1/np.sqrt(vY**2 + vX**2) * dtanVector 
    dTangent_hatRow = 1/np.sqrt(vY**2 + vX**2) * dtanRowVector
    Tangent_hatRow = 1/np.sqrt(vY**2 + vX**2) * tanRowVector
    normalRowVector = 1/np.sqrt(vY**2 + vX**2) * normalRowVector

    coordErrVector[0,0],coordErrVector[1,0] = Xnext[0]-xDes, Xnext[1]-yDes

    vXerror = Xnext[2]-vX
    vYerror = Xnext[3]-vY
    yaw_err = wrap_to_pi(Xnext[4] - yaw)

# Convert numpy arrays to tensors

    prjctdLongerr = Tangent_hatRow @ coordErrVector
    prjctdLaterr = normalRowVector @ coordErrVector
    aX = - a * np.cos(arc)
    aY = - b * np.cos(arc)

    k=Tangent_hat[0,0] * dTangent_hat[1,0] - dTangent_hat[0,0] * Tangent_hat[1,0]
    r = k * v
    
    return ca.vertcat(xDes,yDes,vX,vY,yaw,r),prjctdLongerr,prjctdLaterr
    #compute velocity by dt, or by velocity at current 

def create_optimizer_function(f_dyn,dt,time_horizon):
    Q = ca.diag([100, 100, 1, 1, 100, 10])  # State cost
    R = ca.diag([10, 1])                # Input cost

    opti = Opti()
    opts = {
        'ipopt.print_level': 0,           # 0 = no output, 1 = minimal, 5 = full debug
        'print_time': False,              # Don't print timing info
        'ipopt.sb': 'yes',                # Suppress IPOPT banner
        'ipopt.max_iter': 500,            # Limit iterations (default is often 3000)
    }
    opti.solver('ipopt',opts)


    x_opti = opti.variable(6,time_horizon+1)
    u_opti = opti.variable(2,time_horizon)
    
    x0 = opti.parameter(6)

    max_steer_rad = 0.4  # approx 23 degrees
    max_accel_ms2 = 3.0

    opti.subject_to(opti.bounded(-max_steer_rad, u_opti[0,:], max_steer_rad))
    opti.subject_to(opti.bounded(-max_accel_ms2, u_opti[1,:], max_accel_ms2))
    opti.subject_to(x_opti[:, 0] == x0)

    opti.subject_to(opti.bounded(0.1, x_opti[2, :], 15.0))  # Vx
    opti.subject_to(opti.bounded(-5.0, x_opti[3, :], 5.0))  # Vy 

    x_des = opti.parameter(6,time_horizon+1)
    u_des = opti.parameter(2,time_horizon)
    tan = opti.parameter(1,2)
    normal = opti.parameter(1,2)
    err_prj = opti.parameter(2)

    cost  = 0

    for i in range(time_horizon):
        state_err = x_opti[:,i] - x_des[:,i]

        #frenet frame logic, current vX and vY are in the body frame
        #rotation
        tan[0,0]=ca.cos(x_des[-2,i]) * x_des[2,i] - ca.sin(x_des[-2,i]) * x_des[3,i]
        tan[0,1]=ca.sin(x_des[-2,i]) * x_des[2,i] +  ca.cos(x_des[2,i]) * x_des[3,i]
        tan/=(tan[0,1]**2+tan[0,0]**2)**(1/2)
        normal[0,0]=-tan[0,1]
        normal[0,1]=tan[0,0]
        err_prj[0]=tan @ state_err[:2]
        err_prj[1]= normal @ state_err[:2]
        state_err[0]=err_prj[0]
        state_err[1]=err_prj[1]

        #state_err[:2,]
        input_err = u_opti[:,i] - u_des[:,i]
        

        cost += state_err.T @ Q @ state_err + input_err.T @ R @ input_err        
        xNext = step(x_opti[:,i], u_opti[:,i], f_dyn, dt)
        opti.subject_to(x_opti[:,i+1]==xNext)

    final_err = x_opti[:, -1] - x_des[:, -1]
    cost += final_err.T @ Q @ final_err

    opti.minimize(cost)

    solver_function = opti.to_function('mpc_solver',[x_des,u_des,x0],[u_opti])
    print("saved comp function")
    return solver_function
    #no set actual values yet, only in error params

def scale(x,scales,u=None,flag=False):
    x[0,:] /= scales['x']
    x[1,:] /= scales['y']
    x[2,:] /= scales['x']
    x[3,:] /= scales['y']
    x[4,:] /= scales['yaw']
    x[5,:] /= scales['r']

    if flag:
        u[0,:] /= scales['delta']
        u[1,:] /=scales['a']
        return x,u
    return x
def sim(f_dyn,dt,scales):
    lossXArr = []
    lossYArr = []

    actual_x_coords = []
    actual_y_coords = []
    desired_x_coords = []
    desired_y_coords = []
    plot_steering_angle = []
    plot_control_accel = []

    velocityX_chosen = []
    velocityY_chosen = []
    velocityX_des = []
    velocityY_des = []
    yaw_chosen = []
    r_chosen = []
    yaw_desired = []
    r_desired = []

    time_horizon = 30

    x_coor_ref, y_coord_ref,arc_ref,a,b =create_trackCoordinates()
    xCurr=createStartVector(arc_ref[0],a,b)

    nextX=createStartVector(arc_ref[10],a,b)

    num_des_x = np.zeros((6,time_horizon+1))
    num_des_u = np.zeros((2,time_horizon))

    num_opti_x = np.zeros((6,time_horizon+1))
    U_actual = np.zeros((2,time_horizon))

    optimizer=create_optimizer_function(f_dyn,dt,time_horizon)

    err = 0
    for i in range(60):

        xCurrSim = xCurr
        actual_x_coords.append(xCurr[0].full().item())
        actual_y_coords.append(xCurr[1].full().item())
        for j in range(time_horizon):
            arc_idx=(((i+1)*5)+((j+1)*5))%len(arc_ref)

            if i == 0:

                num_des_u[:,j]=np.array([0,0]).T
                U_actual[:,j]=np.array([0,0]).T
                currErrVector,Xerr,Yerr=computeDesiredVector(arc_ref[arc_idx],a,b,xCurr)
                num_opti_x[:,j] = xCurr.full().flatten()
                num_des_x[:,j]=currErrVector.full().flatten()
                lossXArr.append(Xerr)
                lossYArr.append(Yerr)

                xCurrSim = step(xCurrSim,num_des_u[:,j],f_dyn,dt)
                if j == 0:
                    U_prev=np.array([0,0]).T
                    desired_x_coords.append(currErrVector[0].full().item())
                    desired_y_coords.append(currErrVector[1].full().item())
                    velocityX_des.append(currErrVector[2].full().flatten())
                    velocityY_des.append(currErrVector[3].full().flatten())
                    r_desired.append(currErrVector[-1].full().flatten())
                    yaw_desired.append(currErrVector[-2].full().flatten())

            else:

                currErrVector,Xerr,Yerr=computeDesiredVector(arc_ref[arc_idx],a,b,xCurr)
                num_opti_x[:,j] = xCurr.full().flatten()
                num_des_x[:,j]=currErrVector.full().flatten()
                lossXArr.append(Xerr)
                lossYArr.append(Yerr)
                xCurrSim=step(xCurrSim,U_actual[:,j],f_dyn,dt)

                if j ==0:
                    num_des_u[:,j]=U_prev
                    U_prev = U_actual[:,0].full().flatten()
                    desired_x_coords.append(currErrVector[0].full().item())
                    desired_y_coords.append(currErrVector[1].full().item())
                    velocityX_des.append(currErrVector[2].full().flatten())
                    velocityY_des.append(currErrVector[3].full().flatten())
                    r_desired.append(currErrVector[-1].full().flatten())
                    yaw_desired.append(currErrVector[-2].full().flatten())

                else:
                    num_des_u[:,j]=U_actual[:,j-1].full().flatten()
                
        print(f"ran {i +1} times")
        des_x,des_u=scale(num_des_x,scales,num_des_u,flag=True)
        x_scaled = scale(x=xCurr,scales=scales)
        U_actual=optimizer(des_x,des_u,xCurr)

        U_actual[0,:] *= scales['delta']
        U_actual[1,:] *= scales['a']

        xCurr = step(xCurr,U_actual[:,0],f_dyn,dt)

        plot_control_accel.append(U_actual[0,1].full().flatten())
        plot_steering_angle.append(U_actual[0,0].full().flatten())

        yaw_chosen.append(xCurr[-2].full().flatten())
        r_chosen.append(xCurr[-1].full().flatten())

        velocityX_chosen.append(xCurr[2].full().flatten())
        velocityY_chosen.append(xCurr[3].full().flatten())
        
    plot_actual_vs_desired(velocityX_chosen,velocityY_chosen,velocityX_des,velocityY_des,"Velocities")
    plot_actual_vs_desired(yaw_chosen,r_chosen,yaw_desired,r_desired,"angle plotS")
    plotter(plot_steering_angle,plot_control_accel,"Steering and Control ")
    #TODO plot prev states
    #plot code
    print("LENGTH CHECK",len(desired_x_coords),len(desired_y_coords))
    print(f"RANGE CHECK: des x: {max(desired_x_coords)-min(desired_x_coords)},des y: {max(desired_y_coords)-min(desired_y_coords)}")

    print(f"RANGE CHECK: actual x: {max(actual_x_coords)-min(actual_x_coords)},actual y: {max(actual_y_coords)-min(actual_y_coords)}")

    plt.figure(figsize=(12, 8))
    plt.plot(actual_x_coords, actual_y_coords, 'b-', label='Actual Path', linewidth=2)
    plt.plot(desired_x_coords, desired_y_coords, 'r--', label='Desired Path', linewidth=2)
    
    plt.scatter(actual_x_coords[0], actual_y_coords[0], color='blue', s=100, marker='o', label='Start')
    plt.scatter(actual_x_coords[-1], actual_y_coords[-1], color='blue', s=100, marker='s', label='End')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Vehicle Trajectory: Actual vs Desired')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    # Remove axis('equal') temporarily to see all data
    # plt.axis('equal')  # Comment this out
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return lossXArr,lossYArr

def main():
    mass = 150  # kg
    length = 4   # m
    width = 2.0  # m, Changed from 12
    I_z = mass / 12 * (length**2 + width**2)

    #distance from front and rear axle to COM
    #simplified assumption by only longitudinal axis
    L_f = 1.2
    L_r = 1.3



    #tire model parameters
    D = 400         # N (reasonable tire force)
    B = 10           # tire stiffness
    C = 1.3          # shape factor
    #B = stiffness
    #C = shape factor
    #D = peak lateral force

    dt = .01
    #TODO these need to be changed per the problem statemt
    lossXArr=[]
    lossYArr=[]

    # Control inputs

    # Forces
    F_xr = 0.0
    max_force = 400
    C = 3000
    slip_threshold = .1

    #normalization experiments for X :4, U 4:
    # assumes velocity to be ab 30 m/s, .3 radians
    a  = 20
    b = 10

    scales = {'x': a,
              'y': b,
              'v': 30.0,
              'yaw': 2 * ca.pi,
              'r': ca.pi,
              'delta': 0.3,
              'a': 10.0,

    }



    X,U=create_symbolicVectors()
    X_DOT=create_stateSpace(I_z,mass,L_f, L_r,max_force,C,slip_threshold,
                      X,U, F_xr)
    f_dyn = ca.Function("f_dyn", [X, U], [X_DOT])
    lossXArr,lossYArr=sim(f_dyn,dt,scales)
    plot_loss(lossXArr,"Longitudinal Positional Diff")
    plot_loss(lossYArr,'Lateral Positional Diff')

if __name__ == '__main__':
    main()