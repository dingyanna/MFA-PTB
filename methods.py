###############################
# Methods to compute steady states
###############################
import numpy as np
import time 
from scipy.optimize import root
import random 
from scipy.interpolate import CubicSpline
lmd = 1e-8

def mape(true, pred):
    mask = np.abs(true) > 1e-8
    return np.mean(np.abs(pred[mask] - true[mask]) / true[mask])

def residual(steady, dxdt, net):
    return np.mean(np.abs(dxdt(steady, 0, net.unload(net.gt))))

def deriv(pred):
    return np.mean(np.abs(pred))

def full(net, args, logger, save_state=False, SOLVER='odeint'):
    '''
    Numerical integration using RK4. 

    The difference between solve_ode and rk4 is that the former is implemented by Scipy.
    '''
    lb = 10
    ub = 20
    lambd=1e-18
    if args.dynamics == 'epi':
        lb = 0.2
        ub = 0.4
    x0 = np.random.uniform(lb, ub, net.N)
 
    t1 = time.time()  
    t = np.linspace(0,10,2)
    if SOLVER != 'rk4':
        steady = net.solve_ode(net.dxdt, x0, t, net.gt) 
    else:
        steady, dxdt = rk4(net.dxdt, x0, args=(net.unload(net.gt), ), logger=logger, max_iter=args.full_max_iter*1000, lambd=1e-18, h = args.h)
    
    logger.info(f"\n<x> = {np.mean(steady)}\n")
    t2 = time.time()
    res = residual(steady, net.dxdt, net)
    logger.info(f"Run time to compute true steady state {t2-t1}")
    logger.info(f"Residual <dxdt> {res}")
    logger.info(f"Avg steady states {np.mean(steady)}")
    dxdt = np.mean(np.abs(net.dxdt(steady, 0, net.unload(net.gt))))
    return t2 - t1, res, steady, dxdt

 
def L_avg(net):
    return net.out_degree @ net.degree / net.degree.sum()  

def L_2in(net):
    return net.degree @ net.A @ net.degree / (net.degree.T @ net.degree)  

def mfa(net, args, logger, true, L, save_state=False, SOLVER='odeint'):
    t = np.linspace(0,10,2)  
    t1 = time.time() 
    logger.info(f'\nStart computing xeff {time.ctime()}')   
    beta = L(net)  

    net.xeff = net.get_xeff(net.gt, 20, beta) 
    logger.info(f'Finish computing xeff {time.ctime()}\n') 
    logger.info(f'xeff {net.xeff}\n') 
    if SOLVER != 'rk4':
        xhat_uni = net.solve_ode(net.dxdt_mfa, np.full(len(net.deg_unique), net.xeff), t, net.gt, net.deg_unique) 
        xhat  = np.zeros(net.N)
        for i in range(len(xhat_uni)):
            xhat[net.degree == net.deg_unique[i]] = xhat_uni[i] 
    else: 
        sample_deg = random.sample( list(net.deg_unique), min(int(len(net.deg_unique)),1000))
        sample_deg = np.sort(sample_deg) 
        xhat_unique = rk4(net.dxdt_mfa, np.full(len(sample_deg), net.xeff), args=(net.unload(net.gt), sample_deg), logger=logger, max_iter=args.full_max_iter, lambd=lmd, h = args.h)
         
        cs = CubicSpline(sample_deg, xhat_unique) # change the number of sampled 
            
        xhat = cs(net.degree)   
       
    t2 = time.time() 
    logger.info(f"mfa time {t2 - t1}")
    dxdt = np.mean(np.abs(net.dxdt(xhat, 0, net.unload(net.gt)))) 
     
    return t2 - t1, residual(xhat, net.dxdt, net), mape(true, xhat), dxdt, xhat
  
def mfap(net, args, logger, true, L, save_state=False, SOLVER='odeint'):
    rt, res, state_error, mfa_deriv, xhat = mfa(net, args, logger, true, L, save_state, SOLVER)
    t1 = time.time()
    t = np.linspace(0,1,2)
    if SOLVER != 'rk4':
        y = net.solve_ode(net.dxdt, xhat, t, net.gt, method='odeint') 
    else:
        y = rk4(net.dxdt, xhat, args=(net.unload(net.gt), ), logger=logger, max_iter=args.full_max_iter, lambd=lambd, h = args.h)
    t2 = time.time()
    logger.info(f"Runtime of mfa+ avg {rt+t2-t1}\n")
    dxdt = np.mean(np.abs(net.dxdt(y, 0, net.unload(net.gt))))
    return t2 - t1 + rt, residual(y, net.dxdt, net), mape(true, y), dxdt
 
def prt(net, args, logger, true, L, save_state=False, SOLVER='odeint'):
    """
    return
    ------
    phase1_runtime: float
    phase1_state_error: float
    runtime: float
    residual: float
    state_error: float
    y3: np.array
    """
    rt, res, phase1_state_error, phase1_deriv, xhat = mfa(net, args, logger, true, L, save_state, SOLVER)
    t1 = time.time()       
    y3 = perturbation(net.perturb_err_matrix, xhat, net.unload(net.gt), logger, max_iter=20, lambd=1e-4, h = args.h, net=net)
    t2 = time.time()  
    logger.info(f"Runtime of perturbation method {rt+t2 - t1} ")
    dxdt = np.mean(np.abs(net.dxdt(y3, 0, net.unload(net.gt)))) 
    return rt, phase1_state_error, phase1_deriv, t2 - t1, residual(y3, net.dxdt, net), mape(true, y3), dxdt
 
def rk4(fun, y0, logger=None, h=0.001, args=None, max_iter=100, err_y=None, lambd=1e-18, true_x=None):
    y = y0 
     
    t = 0
    step = 0 
    dxdt = np.mean(np.abs(fun(y,t,*args))) 
    logger.info(f"[RK4][Step {step}] | abs(dxdt) {dxdt} | {time.ctime()}")
    
    last_dxdt = dxdt
    while dxdt > lambd:    
        if step == max_iter:
            break
        k1 = fun(y, t, *args)
        k2 = fun(y+0.5*h*k1, t+0.5*h, *args)
        k3 = fun(y+0.5*h*k2, t+0.5*h, *args)
        k4 = fun(y+h*k3, t+h, *args)
        y = y + (k1 + 2*k2 + 2*k3 + k4) * h / 6
        t = t + h 
        step += 1 
        last_dxdt = dxdt
        dxdt = np.mean(np.abs(fun(y,t,*args))) 
        if step % 5000 == 0:
            logger.info(f"[RK4][Step {step}] | abs(dxdt) {dxdt} | {time.ctime()}")     
        if np.abs(dxdt - last_dxdt) < 1e-12:
            logger.info(f"Break: dxdt is not changing")
            break
    logger.info(f"NFE = {step}\n")
 
    return y, dxdt

 
def perturbation(err_matrix, y0, param, logger, h=0.01, max_iter=100, lambd=1e-7, net=None ):
    '''
    Let e_i = x_i - z_i 

    d e_i / dt ~ [f'(z_i) + sum_j A_ij g_1 (z_i, z_j)] e_i + sum_j A_ij g_2 (z_i, z_j) e_j + f(z_i) + sum_j A_ij g(z_i, z_j) 
    where g1 (g2) is the partial derivative of g w.r.t. the first (second) argument
    
    Let a_i = [f'(z_i) + sum_j A_ij g_1 (z_i, z_j)]
    Let b_ij = g_2 (z_i, z_j)
    Let B_ij = A_ij * b_ij if i != j; B_ij = A_ij * b_ij + a_i if i == j
    Let c_i = f(z_i) + sum_j A_ij g(z_i, z_j) 

    d e_i / dt = B @ e + c (Linear ODE system)
    To compute the equilibrium of this system, there are several ways to do this:
        1. Direct method: compute the analytical solution to this system and take t to infinity
            - Suppose B has real eigenvectors, the general solution of e_i is 
                e = sum_j c_j e^{lambda_j t} v_j
                where c_j are constants, lambda_j, v_j are the jth eigenvalue, eigenvector respectively.
                To determine c_j, we would need n pairs of points (t, e)
        2. Numerical method: RK4 from a small IC
        3. Solving a linear system B @ e = - c using a numerical method starting from a small e. 
    ''' 
    y = y0
    np.random.seed(0)
    err = np.random.normal(0,0.001,len(y)) 
    step = 0 
    prev_err = np.inf
    abs_deriv = np.mean(np.abs(net.dxdt(y, 0, net.gt)))
    while np.mean(np.abs(err)) > 1e-8: 
        if step == max_iter:
            logger.info(f"perturbation ==> reach max iter") 
        logger.info(f"Perturbation Step {step}")  
        B, c = err_matrix(param, y)
        B_inv = np.linalg.inv(B)
        err = B_inv @ (-c)  
        y = y + err 
        abs_deriv = np.mean(np.abs(net.dxdt(y, 0, net.gt)))
        logger.info(f"<|dxd|> {abs_deriv}")
        if abs_deriv > prev_err:
            logger.info(f"Break: Abs deriv increasing")
            break 
        prev_err = abs_deriv
        step += 1
    return y  