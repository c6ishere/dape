# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:23:53 2019

@author: c6
"""
import xmlrpc.client
import numpy as np

def CallPlecs(x,const_Pn,const_sim_time,const_cut_waveform,const_Ulim,const_Ilim,hold_scope):
    #configure the absolute model path
    model_path = "D:\design_automation\plecs"
    #configure the XML-RPC port -> needs to coincide with the PLECS configuration
    port = "1080"
    
    #start PLECS
    server = xmlrpc.client.Server("http://localhost:" + port + "/RPC2")

    ################################################################################
    # Example: Parameter sweep of a cllc converter
    ################################################################################

    # define model name, parameter paths
    model_name = "cllc"
    scope_ref = model_name+'/Scope'
    voltage_path = model_name+'/V_dc'
    resistor_path = model_name+'/R_load'
    vref_path = model_name+'/Crt_DAB/Constant'
    fs_path = model_name+'/Crt_DAB/Pulse'
    tri_path = model_name+'/Crt_DAB/Delay/Triangular'
    lm_path = model_name+'/Linear'
    lr1_path = model_name+'/L1'
    lr2_path = model_name+'/L2'
    cr1_path = model_name+'/C3'
    cr2_path = model_name+'/C4'
    v_ini_path = model_name+'/C6'

    # open the model using the XMLRPC server, needs absolute path
    server.plecs.load(model_path+'\cllc.plecs')
    #clear existing traces in the scope
    server.plecs.scope(scope_ref,'ClearTraces')

    # define parameter values for parameter sweep
    vdc_vals = x[:,0]
    r_vals = x[:,0]*x[:,0]/const_Pn
    vref_vals = x[:,0]
    vini_vals = x[:,0]*0.8
    fs_vals = np.rint(x[:,1]/1000)*1000
    lm_vals = x[:,2]
    lr1_vals = x[:,3]
    lr2_vals = x[:,4]
    cr1_vals = np.rint(x[:,5]*1e6)*1e-6
    cr2_vals = np.rint(x[:,6]*1e6)*1e-6
    f1 = np.empty_like(vdc_vals)
    f2 = np.empty_like(vdc_vals)
    g1 = np.empty_like(vdc_vals)
    g2 = np.empty_like(vdc_vals)

    # loop for all values
    for i in range(len(vdc_vals)):
        #set value for V_dc R_load and control parameters, using plecs.set
        server.plecs.set(voltage_path, 'V','{}'.format(vdc_vals[i])) 
        server.plecs.set(resistor_path, 'R','{}'.format(r_vals[i]))
        server.plecs.set(vref_path, 'Value','{}'.format(vref_vals[i]))
        server.plecs.set(v_ini_path,'v_init','{}'.format(vini_vals[i]))
        server.plecs.set(fs_path,'f','{}'.format(fs_vals[i]))
        server.plecs.set(tri_path,'f','{}'.format(fs_vals[i]*2))
        server.plecs.set(lm_path,'Lm','{}'.format(lm_vals[i]))
        server.plecs.set(lr1_path,'L','{}'.format(lr1_vals[i]))
        server.plecs.set(lr2_path,'L','{}'.format(lr2_vals[i]))
        server.plecs.set(cr1_path,'C','{}'.format(cr1_vals[i]))
        server.plecs.set(cr2_path,'C','{}'.format(cr2_vals[i]))
        #start simulation, get the return array of float64
        opts = {'SolverOpts' :  { 'StopTime' : const_sim_time } } 
        result = server.plecs.simulate(model_name,opts)
        time_vec = np.array(result['Time'])
        ucr1_vec = np.array(result['Values'][0])
        ilr1_vec = np.array(result['Values'][1])
        vout_vec = np.array(result['Values'][2])
        um_vec = np.array(result['Values'][3])
        # cut a slice for further analysis (10 periods)
        ind_last_ten_period = (time_vec >= const_sim_time-const_cut_waveform/fs_vals[i])
        time_vec = time_vec[ind_last_ten_period]
        ucr1_vec = ucr1_vec[ind_last_ten_period]
        ilr1_vec = ilr1_vec[ind_last_ten_period]
        vout_vec = vout_vec[ind_last_ten_period]
        um_vec = um_vec[ind_last_ten_period]
        #add trace to the scope
        if hold_scope == True:
            server.plecs.scope(scope_ref, 'HoldTrace', 'V_dc = {0} V'.format(vdc_vals[i]))
        
        #debug:
        print(i)
        
        # define an objective function to be evaluated using var1
        f1[i] = max(ucr1_vec)-const_Ulim
        f2[i] = max(ilr1_vec)-const_Ilim

        # !!! only if a constraint value is positive it is violated !!!
        # the constraint function gi(x) needs to be converted to gi(x)≤0 constraint
        # steady state error < 1% Vdc
        g1[i] = np.abs(np.mean(vout_vec)-vdc_vals[i])-0.05*vdc_vals[i]
        # softswitching ir1_vec[0]<0
        g2[i] = ilr1_vec[0]
    
    return f1,f2,g1,g2


#%%
# additonal verification in plecs
CallPlecs(res.X, const_Pn = 50000, const_sim_time = 1,
          const_cut_waveform = 10, const_Ulim = 500,
          const_Ilim = 10, hold_scope = True)  
#%%
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem
from pymoo.factory import get_algorithm
import numpy as np

# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(n_dim=4, n_partitions=6).do()

# create the algorithm object
method = get_algorithm("unsga3",
                      pop_size=120,
                      ref_dirs=ref_dirs)

# execute the optimization
res = minimize(get_problem("ackley", n_var=30),
               method,
               termination=('n_gen', 150),
               save_history=True)

print("UNSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))