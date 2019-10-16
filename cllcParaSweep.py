import xmlrpc.client
import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem

def CallPlecs(x,const_Pn,const_sim_time,const_cut_waveform,const_Ulim,const_Ilim,hold_scope,reverse):
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
    zoh_path = model_name+'/Crt_DAB/PI/zoh'
    zoh1_path = model_name+'/Crt_DAB/zoh1'
    di_path = model_name+'/Crt_DAB/PI/di'

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
        server.plecs.set(zoh_path,'Ts','{}'.format(1/fs_vals[i]))
        server.plecs.set(zoh1_path,'Ts','{}'.format(1/fs_vals[i]))
        server.plecs.set(di_path,'SampleTime','{}'.format(1/fs_vals[i]))
        server.plecs.set(lm_path,'Lm','{}'.format(lm_vals[i]))
        # considering power flow in both direction: forward and reverse
        if reverse == False:
            server.plecs.set(lr1_path,'L','{}'.format(lr1_vals[i]))
            server.plecs.set(lr2_path,'L','{}'.format(lr2_vals[i]))
            server.plecs.set(cr1_path,'C','{}'.format(cr1_vals[i]))
            server.plecs.set(cr2_path,'C','{}'.format(cr2_vals[i]))
        else:
            server.plecs.set(lr1_path,'L','{}'.format(lr2_vals[i]))
            server.plecs.set(lr2_path,'L','{}'.format(lr1_vals[i]))
            server.plecs.set(cr1_path,'C','{}'.format(cr2_vals[i]))
            server.plecs.set(cr2_path,'C','{}'.format(cr1_vals[i]))
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
        #print(i)
        
        # define an objective function to be evaluated using var1
        f1[i] = max(ucr1_vec)
        f2[i] = max(ilr1_vec)

        # !!! only if a constraint value is positive it is violated !!!
        # the constraint function gi(x) needs to be converted to gi(x)≤0 constraint
        # steady state error < 1% Vdc
        g1[i] = np.abs(np.mean(vout_vec)-vdc_vals[i])-0.05*vdc_vals[i]
        # softswitching ir1_vec[0]<0
        g2[i] = ilr1_vec[0]
    
    return f1,f2,g1,g2


# always derive from the main problem for the evaluation
class MyProblem(Problem):

    def __init__(self, const_VL=500, const_VU=700,const_FL=10e3, const_FU=20e3,
                 const_LmL=1e-3, const_LmU=10e-3, const_LL=10e-6, const_LU=200e-6, 
                 const_CL=1e-6, const_CU=100e-6, const_cut_waveform=10, const_sim_time = 0.2,
                 const_Ulim = 500, const_Pn = 50000, const_Ilim = 100):

        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = np.array([const_VL, const_FL, const_LmL, const_LL, const_LL, const_CL, const_CL])
        xu = np.array([const_VU, const_FU, const_LmU, const_LU, const_LU, const_CU, const_CU])

        super().__init__(n_var=7, n_obj=4, n_constr=4, xl=xl, xu=xu, evaluation_of="auto")

        # store custom variables needed for evaluation
        self.const_cut_waveform = const_cut_waveform
        self.const_sim_time = const_sim_time
        self.const_Pn = const_Pn
        self.const_Ulim = const_Ulim
        self.const_Ilim = const_Ilim

    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self, x, out, *args, **kwargs):
        
        f1,f2,g1,g2 = CallPlecs(x,self.const_Pn, self.const_sim_time,
                                self.const_cut_waveform, self.const_Ulim,
                                self.const_Ilim,hold_scope = False, reverse = False)
        # f3,f4,g3,g4 = CallPlecs(x,self.const_Pn, self.const_sim_time,
                                # self.const_cut_waveform, self.const_Ulim,
                                # self.const_Ilim,hold_scope = False, reverse = True)
        # out["F"] = anp.column_stack([f1, f2, f3, f4])
        # out["G"] = anp.column_stack([g1, g2, g3, g4])
        out["F"] = anp.column_stack([f1,f2])
        out["G"] = anp.column_stack([g1,g2])

problem = MyProblem()

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_termination

algorithm = NSGA2(
    pop_size=10,
    sampling=np.loadtxt("D:\design_automation\plecs\ini_pop.txt"),
    crossover=get_crossover("real_sbx", prob=1.0, eta=20),
    mutation=get_mutation("real_pm",prob=1.0, eta=10),
    elimate_duplicates=True
)

termination = get_termination("n_gen",2)

from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.util import plotting

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history = True)

#%%
# visualization of designing results
import math
print("\nDesign Space")
for i in range(len(res.X)):
    print("""No.{} best solution found: (with unit)
          \r V_dc = {}V,\n fs = {}kHz,\n lm = {}mH,\n lr1 = {}uH,\n lr2 = {}uH,\n cr1 = {}uF,\n cr2 = {}uF,
          \r Obj1 Function value = {},\n Obj2 Function value = {},
          \r fr1 = {}Hz,\n fr2 = {}Hz,\n Zr1 = {}Ohm,\n Zr2 = {}Ohm,\n"""
          .format(i+1,
                  res.X[i,0],
                  np.rint(res.X[i,1]/1000),
                  res.X[i,2]*1000,
                  res.X[i,3]*1e6,
                  res.X[i,4]*1e6,
                  np.rint(res.X[i,5]*1e6),
                  np.rint(res.X[i,6]*1e6),
                  res.F[i,0],
                  res.F[i,1],
                  1/2/math.pi/math.sqrt(res.X[i,3]*np.rint(res.X[i,5]*1e6)*1e-6),
                  1/2/math.pi/math.sqrt(res.X[i,4]*np.rint(res.X[i,6]*1e6)*1e-6),
                  math.sqrt(res.X[i,3]/(np.rint(res.X[i,5]*1e6)*1e-6)),
                  math.sqrt(res.X[i,4]/(np.rint(res.X[i,6]*1e6)*1e-6))))

#%%
# visualization
print("\nObjective Space")
plt.scatter(res.F[:,0],res.F[:,1])
plt.title("Pareto Front")
plt.xlabel("Obj1: Vmax on resonant Cr")
plt.ylabel("Obj2: Imax on resonant Lr")
plt.show()

from pymoo.performance_indicator.hv import Hypervolume

# create the performance indicator object with reference point (4,4)
metric = Hypervolume(ref_point=np.array([30, 100]))

# collect the population in each generation
pop_each_gen = [a.pop for a in res.history]

# receive the population in each generation
obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]

# calculate for each generation the HV metric
hv = [metric.calc(f) for f in obj_and_feasible_each_gen]

# visualze the convergence curve
plt.plot(np.arange(len(hv)), hv, '-o')
plt.title("Convergence")
plt.xlabel("Generation")
plt.ylabel("Hypervolume")
plt.show()

#%%
# additonal verification in plecs
CallPlecs(res.X, const_Pn = 50000, const_sim_time = 0.5,
          const_cut_waveform = 10, const_Ulim = 500,
          const_Ilim = 100, hold_scope = True,reverse = True) 
# and then the waveform csv can be output by plecs  