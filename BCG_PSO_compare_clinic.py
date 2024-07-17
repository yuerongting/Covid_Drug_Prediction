# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:42:36 2023

@author: ych22001
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import math

#import seaborn as sns
#sns.set_context('talk')

def step(t,start):
    """
    Returns a step function that is 0 for t < start and 1 for t >= start.
    The input t is a time vector.
    """
    return (t >= start).astype(float)

def impulsive(t,start):
    """
    Returns a step function that is 0 for t < start and 1 for t >= start.
    The input t is a time vector.
    """
    return (t == start).astype(float)

# Model parameters
u1 = 1
u2 = 0.41 
p1 = 1.25
p2 = 0.285
p3 = 1.1
p4 = 0.12
p5 = 0.003
alpha = 0.52
# beta = 0.02
beta = 0.011
# r = 0.033
r = 0.032
initial_state = [0.1,0.1,0,0.8]    # initial states
terminal_percent = 0.01

dt = 0.01

# Nonlinear system with input
def derivC(X,t,u):
    x1,x2,x3,x4 = X
    
    u_input = u
    # u_input = 0
    
    B = x1
    E = x2
    Ti = x3
    Tu = x4
    
    
    dB_dt = -u1 * B - p1 * E * B - p2 * B * Tu + u_input
    dE_dt = -u2 * E + alpha * Ti + p4 * E * B - p5 * E * Ti
    dTi_dt = -p3 * E * Ti + p2 * B * Tu
    dTu_dt = -p2 * B * Tu + r * (1 - beta * Tu) * Tu
    return [dB_dt, dE_dt, dTi_dt, dTu_dt]


def con(p, rho, mu):
    G = p[1]  ### Gap
    N = p[2]
    # return [abs(N-math.ceil(N)), abs(G-math.ceil(G))]
    # return [abs(N-round(N)), abs(G-math.ceil(G))]
    # return [N-math.ceil(N), G-math.ceil(G)]
    return [abs(N-round(N)), G - round(G)]


def terminal_state(p):
    
    # p = [6.4     ,   5.     ,    5.98981557] 
    # p = [ 5.99458042,  5.    ,    10.        ]
    
    Ts = 1e-2
    x1,x2,x3,x4 = initial_state
    I = 1
    D = p[0]  ### Dose
    # G = math.ceil(p[1])  ### Gap
    # N = math.ceil(p[2])  ### Total period
    
    G = p[1]  ### Gap
    N = round(p[2])  ### Total period
    
    # t = np.arange(0, N*(G), Ts) ### terminal cost at D*N
    t = np.arange(0, 100, Ts)
    
    u = 0
    for n in range(0,N):
        u = u + D *  impulsive(t,n*(G))   # len = 10^4
    # results
    log3 = []
    for i in range(0,len(t)-1):
        x = [x1,x2,x3,x4]
        log3.append([t[i],x1,x2,x3,x4,u[i]]) # logd3[4]: Tu
        tspan = [t[i],t[i+1]]
        
        x1,x2,x3,x4 = odeint(derivC,x,tspan,args=(u[i],))[-1]
        
        x1 = x1 + u[i]
        
    logd3 = np.asarray(log3).T
    
    return logd3[4][-1] # Tu

#%%
def func2_max_Tu():  
    Ts = 1e-2
    weight = 5
    x1,x2,x3,x4 = initial_state
    I = 1
    
    # D = p[0]  ### Dose
    # G = p[1]  ### Gap
    # N = round(p[2])  ### Total period

    t = np.arange(0, 100, Ts)
    u = np.zeros(len(t))

    log3 = []
    
    for i in range(0,len(t)-1):
        x = [x1,x2,x3,x4]
        log3.append([t[i],x1,x2,x3,x4,u[i]]) # logd3[4]: Tu
        tspan = [t[i],t[i+1]]
        
        x1,x2,x3,x4 = odeint(derivC,x,tspan,args=(u[i],))[-1]
        x1 = x1 + u[i]

    logd3 = np.asarray(log3).T

    return logd3[4][-1]

Tu_max = func2_max_Tu()  ### Max Tu for normalize

def func2(p, rho, mu):  
    # Augmented Lagrangian for rho and mu (not used)
    #%% test
    
    # p = [ 4.8252752,  5.       , 10.       ]
    # p = [ 5.46,  5     ,    10        ]
    
    
    # p = [lb[0], ub[1], lb[2]]
    # rho = 1.0
    # scaling_factor = 2.0
    # mu0 = 0.0
    # mu = 0
    # p = x0
    # rho = 0 
    # mu = 0

    #%%
    
    Ts = 1e-2
    # weight = 5* 1e2 ### 3* 1e2
    # weight = 1000
    weight = 200
    # weight = 500
    
    x1,x2,x3,x4 = initial_state
    I = 1
    
    D = p[0]  ### Dose
    
    # G = round(p[1])  ### Gap
    G = p[1]  ### Gap
    # N = math.ceil(p[2])  ### Total period
    
    N = round(p[2])  ### Total period
    
    
    # t = np.arange(0, N*(G), Ts)
    t = np.arange(0, 100, Ts)
    u = 0
    for n in range(0,N):
        u = u + D *  impulsive(t,n*(G))   # len = 10^4

    log3 = []
    
    for i in range(0,len(t)-1):
        x = [x1,x2,x3,x4]
        log3.append([t[i],x1,x2,x3,x4,u[i]]) # logd3[4]: Tu
        tspan = [t[i],t[i+1]]
        
        x1,x2,x3,x4 = odeint(derivC,x,tspan,args=(u[i],))[-1]
        x1 = x1 + u[i]

    logd3 = np.asarray(log3).T
    
    # Cost function
    J = 0
    J = D*I*N
    # J = logd3[1][len(logd3[0])-1]*Ts * weight
    for n in range(0,  len(logd3[0])  ):
        # if  logd3[4][ n ] > 0.10 * initial_state[3]:
        
        norm_stage_cost = logd3[4][ n ] / Tu_max
        
        if  norm_stage_cost > 1e-6:
            # J = J + logd3[4][ n ] * Ts * weight
            # J = J + logd3[4][ n ] * Ts * weight
            
            J = J + norm_stage_cost * Ts * weight



    # Terminal constraints
    terminal_weight = 1 * 1e5
    
    # terminal_weight = 5 * 1e6
    
    
    # if terminal_state([D,G,N]) < initial_state[3] * terminal_percent:
    #     terminal_con_func = 0
    # else:
    #     terminal_con_func = terminal_state([D,G,N]) / Tu_max * terminal_weight
    
    terminal_con_func = terminal_state([D,G,N]) / Tu_max * terminal_weight
    
    
    # J = J + rho/2.0 * terminal_con_func**2 - mu * terminal_con_func
    # J = J + rho/2.0 * terminal_con_func**2 - mu * terminal_con_func + 1e2 * con(p, rho, mu)[0] + 1e2 * con(p, rho, mu)[1]
    # J = J + terminal_con_func + 1e2 * con(p, rho, mu)[0] + 1e2 * con(p, rho, mu)[1]
    
    constraint_weight = 1*1e4
    
    J = J + terminal_con_func + constraint_weight * con(p, rho, mu)[0]  ## normalized cost

    
    #%% Plot
    plot=0
    if(plot==1):
        
        plt.rc('font', size=font_size)
        
        import seaborn as sns
        clrs = sns.color_palette("husl", 5)
        
        fig, ax1 = plt.subplots()
        ax1.set_ylabel( 'BCG (1 x $10^6$ c.f.u )', color='C0') # B
        ax1.set_xlabel('Time (days)')
        ax1.tick_params(axis='y', color='C0', labelcolor='C0')
        
        plot_index = len(logd3[0])
        B_plot = ax1.plot(t[:plot_index], logd3[1][:plot_index], label = 'RMC-$B$', color='C0')
        ax1.fill_between( t[:plot_index], logd3[1][:plot_index] ,alpha=0.3, facecolor=clrs[3])
        
        
        # plt.ylim(0,7)
        
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', color='C1', labelcolor='C1')
        ax2.set_ylabel('Cell Population (1 x $10^6$)', color='C1')
        Tu_plot = ax2.plot(t[:plot_index], logd3[4][:plot_index], 'C1', label = 'RMC-$T_u$')
        E_plot = ax2.plot(t[:plot_index], logd3[2][:plot_index], 'C2', label = 'RMC-$E$')
        Ti_plot = ax2.plot(t[:plot_index], logd3[3][:plot_index], 'C3', label = 'RMC-$T_i$')
        
        input_log = logd3[5][:plot_index]
        input_log[input_log == 0] = 'nan'
        # u_log = ax1.scatter(t[:plot_index], input_log,  label = 'RMC-$u$', s= 300, marker = "P", color = 'C4')
        
        u_nan = np.empty((len(t[:plot_index])))
        u_nan[:] = np.nan
        u_plot = ax2.plot(t[:plot_index], u_nan, 'C4', label = 'RMC-$u$', marker = "P", linestyle = 'None')
        
        ax2.fill_between( t[:plot_index], logd3[4][:plot_index] , alpha=0.3, facecolor='C1')
        
        import scipy.io
        scipy.io.savemat('u_plot.mat', dict(x=t[:plot_index], y=input_log))        
        plt.ylim(0,8)
        ax2.spines['right'].set_color('C1')
        ax2.spines['left'].set_color('C0')
        
        # Solution for having two legends
        leg = B_plot + E_plot + Ti_plot + Tu_plot 
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc=0, frameon=True,edgecolor = 'Black', fontsize = font_size)
        
        for line in ax1.legend(leg, labs, loc=0, frameon=True,edgecolor = 'Black', fontsize = font_size).get_lines():
            line.set_linewidth(5.0)
            line.set_markersize(20)
        
        # ax1.get_frame().set_edgecolor('b')
        # plt.xlim(0,21)
        ax1.set_ylim(0,8)
        plt.ylim(0,1)
        plt.xlim(0,60)
        plt.rc('xtick', labelsize=font_size)
        plt.rc('ytick', labelsize=font_size)
        plt.grid()
        plt.show()
        
        
    return J



#%% PSO https://pythonhosted.org/pyswarm/
from pyswarm import pso


#%% Searching range expanded

# import random
lb = [0,   1,     1] ### Large constraint
ub = [20,    20,     20]


p = [80,20,1]
p = [30,2,1]


x0 = [lb[0], ub[1], lb[2]]
best_sol_PSO = []
# best_cost_PSO = 10
rho = 1.0
scaling_factor = 2.0
mu0 = 0.0
mu = mu0

record_PSO_search_range = np.array(x0)
cost_record_PSO_search_range = [func2(x0, rho, mu)]


seed_value = 1
# seed_value = 111
np.random.seed(seed_value)

while abs( terminal_state(x0) ) > initial_state[3] * terminal_percent or func2(x0, rho, mu) > 80:
    
    # Define the Lagrange multiplier update rule
    mu = mu0 - rho * terminal_state(x0)
    

        
    # Solve the particle swarm optimization problem
    xopt, fopt = pso(func2, lb, ub, args=(rho, mu),  swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=1)
    print(xopt)
    print(func2(xopt,rho,mu))
    print(fopt)
    # Update the decision variables and Lagrange multiplier
    x0 = xopt
    mu0 = mu
    rho *= scaling_factor


    record_PSO_search_range = np.vstack((record_PSO_search_range, x0))
    
    cost_record_PSO_search_range.append(func2(x0, rho, mu))

#%% Plot Searching Range Expand (PSO)
D_PSO = record_PSO_search_range[:, 0]  # Dose
G_PSO = record_PSO_search_range[:, 1]  # Gap
N_PSO = record_PSO_search_range[:, 2]  # Total period

x_PSO_plot = np.linspace(0,len(record_PSO_search_range), len(record_PSO_search_range)-1)

fig, ax1 = plt.subplots()

# Plot Dose, Gap, and # of Treatment on the first y-axis (ax1)
line_width = 5
font_size = 20

ax1.plot(x_PSO_plot, D_PSO[1:], label="Dose (1 x $10^6$ c.f.u )", linewidth=line_width)
ax1.plot(x_PSO_plot, G_PSO[1:], label="Gap (days)", linewidth=line_width)
ax1.plot(x_PSO_plot, N_PSO[1:], label="# of Treatment", linewidth=line_width)

# Set labels and legends for the first y-axis
ax1.set_xlabel("Iterations", fontsize=font_size)
ax1.set_ylabel("Amplitude", fontsize=font_size)
ax1.legend(loc='best', fontsize=font_size)
ax1.tick_params(axis='y', labelsize=font_size)

# Create a second y-axis (ax2) for the "Cost" data
ax2 = ax1.twinx()
ax2.plot(np.linspace(0,len(record_PSO_search_range), len(record_PSO_search_range)), cost_record_PSO_search_range, label="Cost", linewidth=line_width, color='red')
# ax2.plot(x_PSO_plot, cost_record_PSO, label="Cost", linewidth=line_width, color='red')
# Set labels and legends for the second y-axis
ax2.set_ylabel("Cost", fontsize=font_size, color='red')
ax2.tick_params(axis='y', labelcolor='red', labelsize=font_size)

# Combine legends for both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.tick_params(axis='x', labelsize=font_size)
ax1.legend(lines, labels, loc='best', fontsize=font_size)

plt.title('Particle Swarm Optimization')
# plt.xlim(0, 12)
plt.grid()

plt.show()

# PSO_exp_solution = [30., 20. , 1.]
# PSO_exp_solution = [30., 20. , 2.]
# PSO_exp_solution = [30., 1. , 1.]
# PSO_exp_solution = [50., 1. , 1.]

# PSO_exp_solution = [5.2,7,7]

# # PSO_exp_solution = [1.09036884e+01, 4.77626033e+00, 3.76755783e+00]
# PSO_exp_solution = [10.9, 5, 4]


# PSO_exp_solution = [50, 20, 1]

PSO_exp_solution = [ 3.25018106 , 1.      ,   20.        ]
# cost 83.69263446173372

terminal_state(PSO_exp_solution)

func2(PSO_exp_solution, rho, mu)





#%% (1) PSO (normal range)

# seed_value = 1
# p = [ 4.89954693 , 5.      ,   10.        ]


seed_value = 111
# Stopping search: maximum iterations reached --> 1
# Solution [ 5.46473987  5.         10.        ]
# cost 87.7159871825318
# Terminal 2.41619335066772e-05

# Solution [ 5.46473987  5.         10.        ]
# cost 87.89471640491232
# Terminal 2.6817057580941576e-05


np.random.seed(seed_value)

lb = [2.2,   5,     3]
ub = [6.4,    10,     10]

# initial value
rho = 1.0
scaling_factor = 2.0
mu0 = 0.0

mu = mu0

x0 = [lb[0], ub[1], lb[2]]

best_sol_PSO = []
# best_cost_PSO = 10
record_PSO = np.array(x0)
cost_record_PSO = [func2(x0, rho, mu)]


# xopt = [ 5.99458042,  6.1    ,    8.9        ]


xopt = [ 5.46473987,  5.        , 10.        ]
while abs(terminal_state(x0)) > initial_state[3] * terminal_percent :
# while abs(terminal_state(x0)) > 1e-2:
    # Define the Lagrange multiplier update rule
    mu = mu0 - rho * terminal_state(x0)

    # Solve the particle swarm optimization problem
    xopt, fopt = pso(func2, lb, ub, args=(rho, mu),  swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=1)
    print("Solution", xopt)
    print("cost", func2(xopt,rho,mu) )
    print("Terminal", terminal_state(xopt))
    # print(fopt)
    # Update the decision variables and Lagrange multiplier
    x0 = xopt
    mu0 = mu
    rho *= scaling_factor


    record_PSO = np.vstack((record_PSO, x0))
    
    cost_record_PSO.append(func2(x0, rho, mu))
    



#%% Plot PSO
D_PSO = record_PSO[:, 0]  # Dose
G_PSO = record_PSO[:, 1]  # Gap
N_PSO = record_PSO[:, 2]  # Total period

x_PSO_plot = np.linspace(0,len(record_PSO)-1, len(record_PSO))

fig, ax1 = plt.subplots()

# Plot Dose, Gap, and # of Treatment on the first y-axis (ax1)
line_width = 5
font_size = 30

ax1.plot(x_PSO_plot, D_PSO[0:], label="Dose (1 x $10^6$ c.f.u )", linewidth=line_width)
ax1.plot(x_PSO_plot, G_PSO[0:], label="Gap (days)", linewidth=line_width)
ax1.plot(x_PSO_plot, N_PSO[0:], label="# of Treatment", linewidth=line_width)

# Set labels and legends for the first y-axis
ax1.set_xlabel("Iterations", fontsize=font_size)
ax1.set_ylabel("Amplitude", fontsize=font_size)
ax1.legend(loc='best', fontsize=font_size)
ax1.tick_params(axis='y', labelsize=font_size)

# Create a second y-axis (ax2) for the "Cost" data
ax2 = ax1.twinx()
# ax2.plot(np.linspace(0,len(record_PSO), len(record_PSO)), cost_record_PSO, label="Cost", linewidth=line_width, color='red')
ax2.plot(x_PSO_plot, cost_record_PSO, label="Cost", linewidth=line_width, color='red')
# Set labels and legends for the second y-axis
ax2.set_ylabel("Cost", fontsize=font_size, color='red')
ax2.tick_params(axis='y', labelcolor='red', labelsize=font_size)

# Combine legends for both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.tick_params(axis='x', labelsize=font_size)
ax1.legend(lines, labels, loc='best', fontsize=font_size)

plt.title('PSO')
# plt.xlim(0, 12)
plt.grid()

plt.show()


### Best PSO solution (normal range)
best_PSO = record_PSO[np.where(cost_record_PSO == np.min(cost_record_PSO))[0][0],:]


#%% Best solution
# p = [ 5.46473987,  5.        , 10.        ]
terminal_state(p)
x0 = p

xopt, fopt = pso(func2, lb, ub, debug = True, f_ieqcons=con,swarmsize=100, phip = 0.5, phig = 0.5)





# #%% (2) Simulated annealing
# #%%
# ### Parameter settings
# initial_temperature = 1.0
# temperature = initial_temperature
# cooling_rate = 0.95
# max_iterations = 100


# best_sol_SA = []
# best_cost_SA = 10
# record_SA = np.array(x0)
# cost_record_SA = []

# # cost_function = func2


# rho = 1.0
# scaling_factor = 2.0
# mu0 = 0.0


# # Initial solution
# x0 = [lb[0], ub[1], lb[2]]  # Initial solution (replace with your initial solution)


# # seed_value = 10
# seed_value = 111
# np.random.seed(seed_value)
# # Simulated Annealing optimization loop
# while abs(terminal_state(x0)) > initial_state[3] * terminal_percent  or best_cost_SA > 100:
# # while abs(terminal_state(x0))  > 1e-3 and max_iterations > 0 or best_cost_SA > 100:
    
#     mu = mu0 - rho * terminal_state(x0)
    
#     # Generate a neighboring solution
#     neighbor_solution = x0 + np.random.uniform(low=-10, high=10, size=len(x0))

#     # Clip the neighbor solution within bounds (lb and ub should be defined)
#     neighbor_solution = np.clip(neighbor_solution, lb, ub)

#     # Calculate the cost of the neighbor solution
#     neighbor_cost = func2(neighbor_solution, rho, mu)

#     # Decide whether to accept the neighbor solution
#     # if neighbor_cost < func2(x0, rho, mu) or np.random.rand() < np.exp((func2(x0, rho, mu) - neighbor_cost) / temperature):
#     #     x0 = neighbor_solution
    
#     cost_change = neighbor_cost - func2(x0, rho, mu)
    
#     if cost_change < 0 or np.random.rand() < np.exp(-cost_change / temperature):
#         x0 = neighbor_solution



#     # Reduce the temperature
#     temperature *= cooling_rate
#     max_iterations -= 1
    
#     if func2(x0, rho, mu) < best_cost_SA:
#         best_cost_SA = func2(x0, rho, mu) 
#         best_sol_SA = x0
    
    
#     print("x0 value: ", x0)
#     print("Cost value: ", func2(x0, rho, mu) )
    
#     mu0 = mu
#     rho *= scaling_factor
    
#     # record_SA = record_SA.append(x0)
    
#     # record_SA = np.concatenate((record_SA, x0), axis = 1)
#     record_SA = np.vstack((record_SA, x0))
    
#     cost_record_SA.append(func2(x0, rho, mu))
    
    
# # X0 = [ 4.07633343 ,10.       ,   4.95174861]
# # X0 = [ 4.07633343 ,5.       ,   4.95174861]


# #%% Plot (2) Simulated Annealing
# line_width = 5
# font_size = 40
# plt.rc('font', size=font_size)

# D_SA = record_SA[:, 0]  # Dose
# G_SA = record_SA[:, 1]  # Gap
# N_SA = record_SA[:, 2]  # Total period

# x_SA_plot = np.linspace(0,len(record_SA)-2, len(record_SA)-1)

# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.plot(x_SA_plot, D_SA[1:], label="Dose (1 x $10^6$ c.f.u )", linewidth=line_width)
# ax1.plot(x_SA_plot, G_SA[1:], label="Gap (days)", linewidth=line_width)
# ax1.plot(x_SA_plot, N_SA[1:], label="# of Treatment", linewidth=line_width)

# # Set labels and legends for the first y-axis
# ax1.set_xlabel("Iterations", fontsize=font_size)
# ax1.set_ylabel("Amplitude", fontsize=font_size)
# ax1.legend(loc='best', fontsize=font_size)
# ax1.tick_params(axis='y', labelsize=font_size)

# # Create a second y-axis (ax2) for the "Cost" data
# ax2 = ax1.twinx()
# ax2.plot(x_SA_plot, cost_record_SA, label="Cost", linewidth=line_width, color='red')

# # Set labels and legends for the second y-axis
# ax2.set_ylabel("Cost", fontsize=font_size, color='red')
# ax2.tick_params(axis='y', labelcolor='red', labelsize=font_size)

# # Combine legends for both y-axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# lines = lines1 + lines2
# labels = labels1 + labels2
# ax1.tick_params(axis='x', labelsize=font_size)
# ax1.legend(lines, labels, loc='upper center', fontsize=font_size)
# figure_width = 10  # Width of the entire figure in inches
# figure_height = 6  # Height of the entire figure in inches
# left_margin = 0.2  # Adjust to set the left margin ratio
# right_margin = 0.15  # Adjust to set the right margin ratio
# top_margin = 0.1  # Adjust to set the top margin ratio
# bottom_margin = 0.1  # Adjust to set the bottom margin ratio

# # Calculate the plot area size based on the margins
# plot_width = 1 - left_margin - right_margin
# plot_height = 1 - top_margin - bottom_margin

# # Set the plot area size
# ax1.set_position([left_margin, bottom_margin, plot_width, plot_height])
# ax2.set_position([left_margin, bottom_margin, plot_width, plot_height])
# plt.title('Simulated Annealing')
# # plt.xlim(0, 12)
# plt.grid()

# plt.show()


# #%% (3) Ant Colony Optimization

# import numpy as np
# import random
# def calculate_heuristics(D, G, N):
#     # the higher the heuristic value is desired
#     return 1/D, G , 1/N
# def determine_parameter_to_update():
#     # Define exploration probability (e.g., 10%)
#     exploration_probability = 0.1

#     # Generate a random number to decide between exploration and exploitation
#     exploration_decision = np.random.uniform(0, 1)

#     if exploration_decision < exploration_probability:
#         # Explore: Randomly select one of the parameters (D, G, or N)
#         parameter_index = random.randint(0, 2)
#         # Adjust the parameter value by a random amount within the step size
#         parameter_values[ant][parameter_index] += random.uniform(-exploration_step_size, exploration_step_size)
#         # Clip the parameter value to stay within bounds (lb and ub)
#         parameter_values[ant][parameter_index] = np.clip(parameter_values[ant][parameter_index], lb[parameter_index], ub[parameter_index])
        
#     else:
#         # Exploit: Select the parameter with the highest pheromone level
#         parameter_index = select_parameter_with_highest_pheromone()

#     return parameter_index

# def select_parameter_with_highest_pheromone():
#     return np.random.randint(0, 2)  # Randomly select among D, G, and N

# # rho = 1.0
# # scaling_factor = 2.0
# # mu0 = 0.0
# # Define problem bounds and parameters
# # lb = [2.2, 5, 3]
# # ub = [6.4, 10, 10]
# num_ants = 100
# num_iterations = 10
# pheromone_decay = 0.2
# alpha = 1.0  # Weight of pheromone information
# beta = 0.8  # Weight of heuristic information

# initial_solution = [lb[0], ub[1], lb[2]]

# parameter_ranges = np.array([ub[i] - lb[i] for i in range(3)])
# parameter_mins = np.array(lb)
# parameter_values = np.tile(initial_solution, (num_ants, 1))  # Set initial values to [1, 2, 3]

# selected_params = initial_solution
# # Define the pheromone matrix and initialize it
# pheromone_matrix = np.ones((num_ants, 3))  # 3 parameters: D, G, N

# best_solution = np.array([lb[0], ub[1], lb[2]])
# best_cost = float('inf')

# record_ACO = np.array(x0)
# cost_record_ACO = []

# exploration_step_size = 0.2  # Adjust as needed

# # for iteration in range(num_iterations):
# max_iterations = 10
# # while abs( terminal_state(selected_params) )  > 1e-3 and max_iterations > 0 or best_cost>100:

# seed_value = 10
# np.random.seed(seed_value)

# while abs( terminal_state(selected_params) ) > initial_state[3] * terminal_percent :
# # while abs( terminal_state(selected_params) )  > 1e-3 and max_iterations > 0:
#     for ant in range(num_ants):
#         # Get the current parameter values for this ant
#         selected_params = parameter_values[ant]
        
#         # selected_params = [2.2 ,5. , 3. ]
#         # selected_params = [2.2    ,    8.47974751, 5.66899089]

#         cost = func2(selected_params, rho, mu)

#         # Calculate probabilities for selecting parameter values
#         pheromone_weights = pheromone_matrix[ant] ** alpha
#         heuristic_weights = calculate_heuristics(selected_params[0], selected_params[1], selected_params[2])
#         total_weights = pheromone_weights * heuristic_weights

#         # Normalize the total_weights to create a valid probability distribution
#         total_weights /= total_weights.sum()

#         # Use the probability distribution to probabilistically update parameter values
#         parameter_values[ant] = np.random.rand(3)  # Reset parameter values
#         for i in range(3):

#             cumulative_prob = np.cumsum(total_weights)
#             rand_val = np.random.rand()
            
#             # Calculate a scaling factor based on the flexibility (e.g., scaling_factor)
#             step_size = parameter_ranges[i] * scaling_factor
            
#             # Update the parameter value by adding a random step within the chosen range
#             parameter_values[ant][i] += np.random.uniform(-step_size, step_size)
            
#             # Clip the parameter value to stay within bounds (lb and ub)
#             parameter_values[ant][i] = np.clip(parameter_values[ant][i], lb[i], ub[i])
        
#         parameter_index = determine_parameter_to_update()
        
#         # Pheromone evaporation: Reduce pheromone levels globally
#         pheromone_matrix *= (1.0 - pheromone_decay)

#         # Pheromone deposition: Increase pheromone levels on the chosen path
#         pheromone_deposit = 1.0 / (cost + 1e-6)  # Consider a small constant to avoid division by zero
        
#         # pheromone_matrix[ant] += pheromone_deposit
#         pheromone_matrix[ant, parameter_index] += pheromone_deposit
        
#         # print("Ant solution: ", selected_params)
#         # print("Cost:", func2(selected_params, rho, mu))
        
#     # Update best solution if needed
#     if func2(selected_params, rho, mu) < best_cost:
#         best_solution = selected_params
#         best_cost = func2(selected_params, rho, mu)
        
#     # if func2(x0, rho, mu) < best_cost_SA:
#     #     best_cost_SA = func2(x0, rho, mu) 
#     #     best_sol_SA = x0        
#     # print("iteration: ", iteration)
        

#     record_ACO = np.vstack((record_ACO, best_solution))
    
#     cost_record_ACO.append(best_cost)
    
#     # Print the best solution and cost at each iteration
#     print( "Best Solution (D, G, N):", best_solution, "Minimum Cost J:", best_cost)


# #%% Plot (Ant)
# line_width = 5
# font_size = 40
# plt.rc('font', size=font_size)

# D_ACO = record_ACO[:, 0]  # Dose
# G_ACO = record_ACO[:, 1]  # Gap
# N_ACO = record_ACO[:, 2]  # Total period

# x_ACO_plot = np.linspace(0,len(record_ACO)-2, len(record_ACO)-1)


# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.plot(x_ACO_plot, D_ACO[1:], label="Dose (1 x $10^6$ c.f.u )", linewidth=line_width)
# ax1.plot(x_ACO_plot, G_ACO[1:], label="Gap (days)", linewidth=line_width)
# ax1.plot(x_ACO_plot, N_ACO[1:], label="# of Treatment", linewidth=line_width)

# # Set labels and legends for the first y-axis
# ax1.set_xlabel("Iterations", fontsize=font_size)
# ax1.set_ylabel("Amplitude", fontsize=font_size)
# ax1.legend(loc='best', fontsize=font_size)
# ax1.tick_params(axis='y', labelsize=font_size)

# # Create a second y-axis (ax2) for the "Cost" data
# ax2 = ax1.twinx()
# ax2.plot(x_ACO_plot, cost_record_ACO, label="Cost", linewidth=line_width, color='red')

# # Set labels and legends for the second y-axis
# ax2.set_ylabel("Cost", fontsize=font_size, color='red')
# ax2.tick_params(axis='y', labelcolor='red', labelsize=font_size)

# # Combine legends for both y-axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# lines = lines1 + lines2
# labels = labels1 + labels2
# ax1.tick_params(axis='x', labelsize=font_size)
# ax1.legend(lines, labels, loc='upper right', fontsize=font_size)

# figure_width = 10  # Width of the entire figure in inches
# figure_height = 6  # Height of the entire figure in inches
# left_margin = 0.2  # Adjust to set the left margin ratio
# right_margin = 0.15  # Adjust to set the right margin ratio
# top_margin = 0.1  # Adjust to set the top margin ratio
# bottom_margin = 0.1  # Adjust to set the bottom margin ratio

# # Calculate the plot area size based on the margins
# plot_width = 1 - left_margin - right_margin
# plot_height = 1 - top_margin - bottom_margin

# # Set the plot area size
# ax1.set_position([left_margin, bottom_margin, plot_width, plot_height])
# ax2.set_position([left_margin, bottom_margin, plot_width, plot_height])

# plt.title('Ant Colony Optimization')
# # plt.xlim(0, 12)
# plt.grid()
# plt.show()




#%% Uncertainty Parameter

Ts = 1e-2

import random
t = np.arange(0, 100, Ts)
# num_sim_ini_condition = 20

initial_state = np.array(initial_state)

Nsim = 100/0.01;
len_sim = round(Nsim -1) ;


# best_solution = [ 5.46,  5   ,    10       ]
best_solution = [ 5.6,  7     ,    6        ]
# best_solution = [ 5.6,  7     ,    6        ]

u_input_seq = []
u = 0
D = best_solution[0]
G = best_solution[1]
N = best_solution[2]

for n in range(0,N):
    u = u + D *  impulsive(t,n*(G))   # len = 10^4
u_input_seq = u
# u_input_seq = u_input_seq[::10]






import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the Monte Carlo simulation
num_iterations = 200  # Number of Monte Carlo iterations
# =============================================================================
# num_iterations = 1  # Number of Monte Carlo iterations
# =============================================================================
num_parameters = 10  # Number of model parameters to sample

# Model parameters
u1 = 1
u2 = 0.41 
p1 = 1.25
p2 = 0.285
p3 = 1.1
p4 = 0.12
p5 = 0.003
alpha = 0.52
beta = 0.02
r = 0.033
initial_state = [0.1, 0.1, 0, 0.8]  # Initial states

# Initialize arrays to store sampled parameters and simulation results
sampled_parameters = np.zeros((num_iterations, num_parameters))
simulation_results = np.zeros((num_iterations, 4))  # Assuming there are 4 output variables

uncer_percent = 0.1
# =============================================================================
# uncer_percent = 0.0
# =============================================================================

# Perform Monte Carlo simulation to sample parameters
for i in range(num_iterations):
    sampled_parameters[i] = [
        np.random.normal(u1, u1*uncer_percent),  # Sample u1 from a normal distribution with std deviation 0.1
        np.random.normal(u2, u2*uncer_percent),  # Sample u2 from a normal distribution with std deviation 0.05
        np.random.normal(p1, p1*uncer_percent),  # Sample p1 from a normal distribution with std deviation 0.1
        np.random.normal(p2, p2*uncer_percent),  # Sample p2 from a normal distribution with std deviation 0.05
        np.random.normal(p3, p3*uncer_percent),  # Sample p3 from a normal distribution with std deviation 0.1
        np.random.normal(p4, p4*uncer_percent),  # Sample p4 from a normal distribution with std deviation 0.05
        np.random.normal(p5, p5*uncer_percent),  # Sample p5 from a normal distribution with std deviation 0.01
        np.random.normal(alpha, alpha*uncer_percent),  # Sample alpha from a normal distribution with std deviation 0.02
        np.random.normal(beta, beta*uncer_percent),  # Sample beta from a normal distribution with std deviation 0.005
        np.random.normal(r, r*uncer_percent)  # Sample r from a normal distribution with std deviation 0.01
    ]

# Define the derivative function based on the sampled parameters
def derivC(X, t, u, p):
    x1, x2, x3, x4 = X
    # u_input = 0
    u_input = u
    
    B, E, Ti, Tu = x1, x2, x3, x4
    dB_dt = -p[0] * B - p[2] * E * B - p[3] * B * Tu + u_input
    dE_dt = -p[1] * E + p[7] * Ti + p[5] * E * B - p[6] * E * Ti
    dTi_dt = -p[4] * E * Ti + p[3] * B * Tu
    dTu_dt = -p[3] * B * Tu + p[9] * (1 - p[8] * Tu) * Tu
    return [dB_dt, dE_dt, dTi_dt, dTu_dt]

B_rec = np.zeros((num_iterations, len_sim) );

E_rec = np.zeros((num_iterations, len_sim) );

Ti_rec = np.zeros((num_iterations, len_sim) );

Tu_rec = np.zeros((num_iterations, len_sim) );

u_rec = np.zeros((num_iterations, len_sim) );

t_terminal = np.zeros((num_iterations) );

# initial_rec = np.zeros((num_sim_ini_condition, 4) );

# u = u_record[0,:]
u = u_input_seq

# Run the model with sampled parameters to obtain results
for n in range(num_iterations):
# for n in range(num_iterations):
    sampled_params = sampled_parameters[n]
 
    # u1,u2,p1,p2,p3,p4 ,p5 ,alpha,beta,r = sampled_params
    initial_state = [0.1, 0.1, 0, 0.8]  # Initial states
    x1,x2,x3,x4 = initial_state
    
    # results
    log3 = []
    for i in range(0,len(t)-1):
        x = [x1,x2,x3,x4]
        log3.append([t[i],x1,x2,x3,x4,u[i]]) # logd3[4]: Tu
        tspan = [t[i],t[i+1]]
        
        x1,x2,x3,x4 = odeint(derivC,x,tspan,args=(u[i],sampled_params))[-1]
        x1 = x1 + u[i]
        
    B_rec[n,:] = (np.asarray(log3).T)[1]
    E_rec[n,:] = (np.asarray(log3).T)[2]
    Ti_rec[n,:] = (np.asarray(log3).T)[3]
    Tu_rec[n,:] = (np.asarray(log3).T)[4]
    u_rec[n,:] = (np.asarray(log3).T)[5]
    
    t_terminal[n] = np.where( (np.asarray(log3).T)[4] > initial_state[3] * terminal_percent)[0][-1] * Ts
    
    print("Num_iterations: ", n)
    
    # simulation_results[n] = np.array([x1,x2,x3,x4])

confidence_intervals = []

for i in range(0,len(t)-1):  # Assuming there are 4 output variables
    lower_percentile = np.percentile(B_rec[:, i], 2.5)
    upper_percentile = np.percentile(B_rec[:, i], 97.5)
    confidence_intervals.append((lower_percentile, upper_percentile))

B_record_max = np.percentile(B_rec, q = 97.5, axis=0)
B_record_min = np.percentile(B_rec, q = 2.5, axis=0)

E_record_max = np.percentile(E_rec, q = 97.5, axis=0)
E_record_min = np.percentile(E_rec, q = 2.5, axis=0)

Ti_record_max = np.percentile(Ti_rec, q = 97.5, axis=0)
Ti_record_min = np.percentile(Ti_rec, q = 2.5, axis=0)

Tu_record_max = np.percentile(Tu_rec, q = 97.5, axis=0)
Tu_record_min = np.percentile(Tu_rec, q = 2.5, axis=0)


#%% Plot

# %matplotlib qt

font_size = 40
y_color = '#dc4d01'


# =============================================================================
# line_width = 1
# =============================================================================
line_width = 3
plt.rc('font', size=font_size)

import seaborn as sns
clrs = sns.color_palette("husl", 5)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_ylabel( 'BCG (1 x $10^6$ c.f.u )', color='C0') # B
ax1.set_xlabel('Time (days)')
ax1.tick_params(axis='y', color='C0', labelcolor='C0')

plot_index = len(Ti_rec[0])

# B_plot_max = ax1.plot(t[:plot_index], B_record_max[:plot_index], label = 'RMC-$B$', color='C0', linewidth= line_width)
B_plot_max = ax1.plot(t[:plot_index], B_record_max[:plot_index], label = 'Clinic-$B$', color='C0', linewidth= line_width)

B_plot_min = ax1.plot(t[:plot_index], B_record_min[:plot_index],  color='C0', linewidth= line_width)

ax1.fill_between( t[:plot_index], B_record_min[:plot_index], B_record_max[:plot_index] ,alpha=0.3, facecolor='C0')

ax2 = ax1.twinx()
ax2.tick_params(axis='y', color='C1', labelcolor=y_color)
ax2.set_ylabel('Cell Population (1 x $10^6$)', color=y_color)

E_plot_max = ax2.plot(t[:plot_index], E_record_max[:plot_index], 'C2', label = 'Clinic-$E$', linewidth= line_width)
E_plot_min = ax2.plot(t[:plot_index], E_record_min[:plot_index], 'C2', linewidth= line_width)

ax2.fill_between( t[:plot_index], E_record_max[:plot_index], E_record_min[:plot_index] ,alpha=0.3, facecolor='C2')

Ti_plot_max = ax2.plot(t[:plot_index], Ti_record_max[:plot_index], 'C3', label = 'Clinic-$T_i$', linewidth= line_width)
Ti_plot_min = ax2.plot(t[:plot_index], Ti_record_min[:plot_index], 'C3', linewidth= line_width)

ax2.fill_between( t[:plot_index], Ti_record_max[:plot_index], Ti_record_min[:plot_index] ,alpha=0.3, facecolor='C3')

Tu_plot_max = ax2.plot(t[:plot_index], Tu_record_max[:plot_index], 'C1', label = 'Clinic-$T_u$', linewidth= line_width)
Tu_plot_min = ax2.plot(t[:plot_index], Tu_record_min[:plot_index], 'C1', linewidth= line_width)

ax2.fill_between( t[:plot_index], Tu_record_max[:plot_index], Tu_record_min[:plot_index] ,alpha=0.3, facecolor='C1')


# Create legends for ax1 and ax2
legend1 = ax1.legend(loc='upper right', markerscale=5)
legend2 = ax2.legend(loc='lower right', markerscale=5)

ax1.get_legend().set_visible(False)
ax2.get_legend().set_visible(False)

# Combine handles and labels from both legends into a shared legend
# combined_handles = legend1.legendHandles + legend2.legendHandles
combined_labels = [label.get_text() for label in legend1.get_texts()] + [label.get_text() for label in legend2.get_texts()]

# Create a shared legend outside of the axes
# Get handles and labels from both legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine handles and labels from both legends into a shared legend
combined_handles = handles1 + handles2


legend_coords = (0.9, 0.8)
# legend_coords = (0.0, 0.0)
combined_legend = fig.legend(combined_handles, combined_labels,  markerscale=10, bbox_to_anchor=legend_coords)

for line in combined_legend.get_lines():
    line.set_linewidth(5.0)
    line.set_markersize(5)
    
ax1.set_ylim(0,6)
plt.ylim(0,1.2)
plt.xlim(0,70)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
# plt.legend()
plt.grid()
plt.show()





#%% Input compare
# =============================================================================
# # u_record[j,:]
# 
# 
# input_log = u_input_seq[:plot_index]
# input_log[input_log == 0] = 'nan'
# # u_log = ax1.scatter(t[:plot_index], input_log,  label = 'RMC-$u$', s= 300, marker = "P", color = 'C4')
# 
# u_nan = np.empty((len(t[:plot_index])))
# u_nan[:] = np.nan
# u_plot = ax2.plot(t[:plot_index], u_nan, 'C4', label = 'RMC-$u$', marker = "P", linestyle = 'None')
# 
# # ax2.fill_between( t[:plot_index], u_record , u_record, alpha=0.3, facecolor='C1')
# 
# import scipy.io
# scipy.io.savemat('H:\\Users\\yrt05\\Desktop\\KoopmanMPC\\u_plot.mat', dict(x=t[:plot_index], y=input_log))        
# =============================================================================




#%% Settling time error bar (Uncertain parameter)
import matplotlib.pyplot as plt

t_terminal_max = np.percentile(t_terminal, q = 97.5, axis=0)
t_terminal_low = np.percentile(t_terminal, q = 2.5, axis=0)

t_terminal_data = t_terminal[ np.bitwise_and(t_terminal> t_terminal_low , t_terminal < t_terminal_max)]

t_mean = np.mean(t_terminal_data)
t_std = np.std(t_terminal_data)

materials = ['']
x_pos = np.arange(len(materials))
CTEs = [t_mean]
error = [2*t_std]

fig, ax = plt.subplots()

# Define the ratio of the figure size to the plot area
figure_width = 10  # Width of the entire figure in inches
figure_height = 6  # Height of the entire figure in inches
left_margin = 0.4  # Adjust to set the left margin ratio
right_margin = 0.15  # Adjust to set the right margin ratio
top_margin = 0.1  # Adjust to set the top margin ratio
bottom_margin = 0.1  # Adjust to set the bottom margin ratio

# Calculate the plot area size based on the margins
plot_width = 1 - left_margin - right_margin
plot_height = 1 - top_margin - bottom_margin

# Set the plot area size
ax.set_position([left_margin, bottom_margin, plot_width, plot_height])

ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='red', color = 'None', capsize=20, width = 0.5, linewidth = 100, error_kw={'elinewidth': 3})
ax.set_ylabel('Time (Days)', fontsize = font_size)
ax.set_xticks(x_pos)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.set_xticklabels(materials, fontsize = font_size)
# ax.set_title('Settling Time', fontsize = font_size)
ax.yaxis.grid(True)

ax.plot(x_pos, CTEs, marker='o', color='red', markersize=18)  # Add this line

plt.ylim(18, 32)
plt.show()







#%%
#%% Uncertain initial condition


# best_solution = [ 5.46,  5   ,    10       ]
best_solution = [ 5.6,  7     ,    6        ]



u_input_seq = []
u = 0
D = best_solution[0]
G = best_solution[1]
N = best_solution[2]

for n in range(0,N):
    u = u + D *  impulsive(t,n*(G))   # len = 10^4
u_input_seq = u
# u_input_seq = u_input_seq[::10]


import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the Monte Carlo simulation
num_iterations = 200  # Number of Monte Carlo iterations
num_parameters = 10  # Number of model parameters to sample

# Model parameters
u1 = 1
u2 = 0.41 
p1 = 1.25
p2 = 0.285
p3 = 1.1
p4 = 0.12
p5 = 0.003
alpha = 0.52
beta = 0.02
r = 0.033
initial_state = [0.1, 0.1, 0, 0.8]  # Initial states

# Initialize arrays to store sampled parameters and simulation results
sampled_parameters = np.zeros((num_iterations, num_parameters))
simulation_results = np.zeros((num_iterations, 4))  # Assuming there are 4 output variables

uncer_percent = 0.0

# Perform Monte Carlo simulation to sample parameters
for i in range(num_iterations):
    sampled_parameters[i] = [
        np.random.normal(u1, u1*uncer_percent),  # Sample u1 from a normal distribution with std deviation 0.1
        np.random.normal(u2, u2*uncer_percent),  # Sample u2 from a normal distribution with std deviation 0.05
        np.random.normal(p1, p1*uncer_percent),  # Sample p1 from a normal distribution with std deviation 0.1
        np.random.normal(p2, p2*uncer_percent),  # Sample p2 from a normal distribution with std deviation 0.05
        np.random.normal(p3, p3*uncer_percent),  # Sample p3 from a normal distribution with std deviation 0.1
        np.random.normal(p4, p4*uncer_percent),  # Sample p4 from a normal distribution with std deviation 0.05
        np.random.normal(p5, p5*uncer_percent),  # Sample p5 from a normal distribution with std deviation 0.01
        np.random.normal(alpha, alpha*uncer_percent),  # Sample alpha from a normal distribution with std deviation 0.02
        np.random.normal(beta, beta*uncer_percent),  # Sample beta from a normal distribution with std deviation 0.005
        np.random.normal(r, r*uncer_percent)  # Sample r from a normal distribution with std deviation 0.01
    ]

# Define the derivative function based on the sampled parameters
def derivC(X, t, u, p):
    x1, x2, x3, x4 = X
    # u_input = 0
    u_input = u
    
    B, E, Ti, Tu = x1, x2, x3, x4
    dB_dt = -p[0] * B - p[2] * E * B - p[3] * B * Tu + u_input
    dE_dt = -p[1] * E + p[7] * Ti + p[5] * E * B - p[6] * E * Ti
    dTi_dt = -p[4] * E * Ti + p[3] * B * Tu
    dTu_dt = -p[3] * B * Tu + p[9] * (1 - p[8] * Tu) * Tu
    return [dB_dt, dE_dt, dTi_dt, dTu_dt]

B_rec_ini = np.zeros((num_iterations, len_sim) );
E_rec_ini = np.zeros((num_iterations, len_sim) );
Ti_rec_ini = np.zeros((num_iterations, len_sim) );
Tu_rec_ini = np.zeros((num_iterations, len_sim) );
u_rec_ini = np.zeros((num_iterations, len_sim) );
t_terminal_ini = np.zeros((num_iterations) );

u = u_input_seq

initial_state_high = [i / 0.9 for i in initial_state]
initial_state_low = [i / 1.1 for i in initial_state]

init_condition_x1 = np.random.uniform(initial_state_low[0], initial_state_high[0], num_iterations)
init_condition_x2 = np.random.uniform(initial_state_low[1], initial_state_high[1], num_iterations)
init_condition_x3 = np.random.uniform(initial_state_low[2], initial_state_high[2], num_iterations)
init_condition_x4 = np.random.uniform(initial_state_low[3], initial_state_high[3], num_iterations)



# Run the model with sampled parameters to obtain results
for n in range(num_iterations):
# for n in range(num_iterations):
    sampled_params = sampled_parameters[n]
 
    # u1,u2,p1,p2,p3,p4 ,p5 ,alpha,beta,r = sampled_params
    initial_state = [init_condition_x1[n], init_condition_x2[n], init_condition_x3[n], init_condition_x4[n]]
    x1,x2,x3,x4 = initial_state

    # results
    log3 = []
    for i in range(0,len(t)-1):
        x = [x1,x2,x3,x4]
        log3.append([t[i],x1,x2,x3,x4,u[i]]) # logd3[4]: Tu
        tspan = [t[i],t[i+1]]
        
        x1,x2,x3,x4 = odeint(derivC,x,tspan,args=(u[i],sampled_params))[-1]
        x1 = x1 + u[i]
        
    B_rec_ini[n,:] = (np.asarray(log3).T)[1]
    E_rec_ini[n,:] = (np.asarray(log3).T)[2]
    Ti_rec_ini[n,:] = (np.asarray(log3).T)[3]
    Tu_rec_ini[n,:] = (np.asarray(log3).T)[4]
    u_rec_ini[n,:] = (np.asarray(log3).T)[5]
    
    t_terminal_ini[n] = np.where( (np.asarray(log3).T)[4] > initial_state[3] * terminal_percent)[0][-1] * Ts
    
    print("Num_iterations: ", n)
    
    # simulation_results[n] = np.array([x1,x2,x3,x4])

confidence_intervals = []

for i in range(0,len(t)-1):  # Assuming there are 4 output variables
    lower_percentile = np.percentile(B_rec_ini[:, i], 2.5)
    upper_percentile = np.percentile(B_rec_ini[:, i], 97.5)
    confidence_intervals.append((lower_percentile, upper_percentile))

B_record_max_ini = np.percentile(B_rec_ini, q = 97.5, axis=0)
B_record_min_ini = np.percentile(B_rec_ini, q = 2.5, axis=0)

E_record_max_ini = np.percentile(E_rec_ini, q = 97.5, axis=0)
E_record_min_ini = np.percentile(E_rec_ini, q = 2.5, axis=0)

Ti_record_max_ini = np.percentile(Ti_rec_ini, q = 97.5, axis=0)
Ti_record_min_ini = np.percentile(Ti_rec_ini, q = 2.5, axis=0)

Tu_record_max_ini = np.percentile(Tu_rec_ini, q = 97.5, axis=0)
Tu_record_min_ini = np.percentile(Tu_rec_ini, q = 2.5, axis=0)


#%% Plot

# %matplotlib qt


line_width = 3
# y_color = [0.1216, 0.4667, 0.7059]
# y_color = '#d9544d'
y_color = '#dc4d01'

plt.rc('font', size=font_size)

import seaborn as sns
clrs = sns.color_palette("husl", 5)

fig, ax1 = plt.subplots(figsize=(10, 6))
# fig, ax1 = plt.subplots()
ax1.set_ylabel('BCG (1 x $10^6$ c.f.u )', color='C0') # B
ax1.set_xlabel('Time (days)')
ax1.tick_params(axis='y', color='C0', labelcolor='C0')

plot_index = len(Ti_rec[0])

B_plot_max = ax1.plot(t[:plot_index], B_record_max_ini[:plot_index], label = 'Clinic-$B$', color='C0', linewidth= line_width)
B_plot_min = ax1.plot(t[:plot_index], B_record_min_ini[:plot_index], color='C0', linewidth= line_width)

ax1.fill_between( t[:plot_index], B_record_min_ini[:plot_index], B_record_max_ini[:plot_index] ,alpha=0.3, facecolor='C0')

ax2 = ax1.twinx()
ax2.tick_params(axis='y', color='C1', labelcolor=y_color)
ax2.set_ylabel('Cell Population (1 x $10^6$)', color=y_color)

E_plot_max = ax2.plot(t[:plot_index], E_record_max_ini[:plot_index], 'C2', label = 'Clinic-$E$', linewidth= line_width)
E_plot_min = ax2.plot(t[:plot_index], E_record_min_ini[:plot_index], 'C2', linewidth= line_width)

ax2.fill_between( t[:plot_index], E_record_max_ini[:plot_index], E_record_min_ini[:plot_index] ,alpha=0.3, facecolor='C2')

Ti_plot_max = ax2.plot(t[:plot_index], Ti_record_max_ini[:plot_index], 'C3', label = 'Clinic-$T_i$', linewidth= line_width)
Ti_plot_min = ax2.plot(t[:plot_index], Ti_record_min_ini[:plot_index], 'C3', linewidth= line_width)

ax2.fill_between( t[:plot_index], Ti_record_max_ini[:plot_index], Ti_record_min_ini[:plot_index] ,alpha=0.3, facecolor='C3')

Tu_plot_max = ax2.plot(t[:plot_index], Tu_record_max_ini[:plot_index], 'C1', label = 'Clinic-$T_u$', linewidth= line_width)
Tu_plot_min = ax2.plot(t[:plot_index], Tu_record_min_ini[:plot_index], 'C1', linewidth= line_width)

ax2.fill_between( t[:plot_index], Tu_record_max_ini[:plot_index], Tu_record_min_ini[:plot_index] ,alpha=0.3, facecolor='C1')


# Create legends for ax1 and ax2
legend1 = ax1.legend(loc='upper right', markerscale=5)
legend2 = ax2.legend(loc='lower right', markerscale=5)

ax1.get_legend().set_visible(False)
ax2.get_legend().set_visible(False)

# Combine handles and labels from both legends into a shared legend
# combined_handles = legend1.legendHandles + legend2.legendHandles

# Get handles and labels from both legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine handles and labels from both legends into a shared legend
combined_handles = handles1 + handles2


combined_labels = [label.get_text() for label in legend1.get_texts()] + [label.get_text() for label in legend2.get_texts()]

# Create a shared legend outside of the axes

legend_coords = (0.9, 0.8)
combined_legend = fig.legend(combined_handles, combined_labels,  markerscale=10, bbox_to_anchor=legend_coords)

for line in combined_legend.get_lines():
    line.set_linewidth(5.0)
    line.set_markersize(5)
    
ax1.set_ylim(0,6)
plt.ylim(0,1.2)
plt.xlim(0,70)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
# plt.legend()
plt.grid()
plt.show()






#%% Settling time error bar (Uncertain initial condition)
import matplotlib.pyplot as plt

font_size = 40

t_terminal_max_ini = np.percentile(t_terminal_ini, q = 97.5, axis=0)
t_terminal_low_ini = np.percentile(t_terminal_ini, q = 2.5, axis=0)

t_terminal_data_ini = t_terminal_ini[ np.bitwise_and(t_terminal_ini> t_terminal_low_ini , t_terminal_ini < t_terminal_max_ini)]

t_mean_ini = np.mean(t_terminal_data_ini)
t_std_ini = np.std(t_terminal_data_ini)

materials_ini = ['']
x_pos = np.arange(len(materials_ini))
CTEs = [t_mean_ini]
error = [2*t_std_ini]
# error = [t_std_ini]
fig, ax = plt.subplots()


# Define the ratio of the figure size to the plot area
figure_width = 10  # Width of the entire figure in inches
figure_height = 6  # Height of the entire figure in inches
left_margin = 0.4  # Adjust to set the left margin ratio
right_margin = 0.15  # Adjust to set the right margin ratio
top_margin = 0.1  # Adjust to set the top margin ratio
bottom_margin = 0.1  # Adjust to set the bottom margin ratio

# Calculate the plot area size based on the margins
plot_width = 1 - left_margin - right_margin
plot_height = 1 - top_margin - bottom_margin

# Set the plot area size
ax.set_position([left_margin, bottom_margin, plot_width, plot_height])

ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='red', color = 'None', capsize=20, width = 0.5, linewidth = 100, error_kw={'elinewidth': 3})
ax.set_ylabel('Time (Days)', fontsize = font_size)
ax.set_xticks(x_pos)
ax.tick_params(axis='both', which='major', labelsize=font_size)
# ax.set_xticklabels(materials, fontsize = font_size)
# ax.set_title('Settling Time', fontsize = font_size)
ax.yaxis.grid(True)

ax.plot(x_pos, CTEs, marker='o', color='red', markersize=18)  # Add this line
plt.ylim(20.3, 20.5)
plt.ylim(28.25, 28.4)
plt.show()


