# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:09:54 2019

@author: c6
"""
import numpy as np
import math

def CalculateDabLoss(r_ds_on_pri = 13e-3, # r_switch i>0 CAS120M12BM2
                     r_ds_on_sec = 1.5/120, # r_switch i<0 CAS120M12BM2
                     r_trans_pri = 45e-3,
                     r_trans_sec = 45e-3,
                     r_inductor = 22.5e-3,
                     r_capacitor = 22.5e-3,
                     r_pcb_ac = 263e-6,
                     r_pcb_dc = 219e-6,
                     e_sw = 3e-3/600/120,
                     um_vec,
                     vout_vec,
                     il_vec,
                     t_vec):
    il_rms = np.sqrt(np.mean(il_vec**2))
    ptr_cond, p_core = CalculateTransformerLoss(il_rms,um_vec)
    pl_cond =  r_inductor*il_rms*il_rms
    p_pcb_cond = (r_pcb_ac+r_pcb_dc)*il_rms*il_rms
    pc_cond = r_capacitor*il_rms*il_rms
    pswitch_cond = CalculateSwitchConductionLoss(il_vec,vout_vec,t_vec)
    
    
    
def CalculateTransformerLoss(il_rms):
    # -----------core parameter
    k = 1.409
    alpha = 1.33
    beta = 1.712
    n1 = 8
    ae = 1160e-6
    ki = KiCal(k,alpha,beta,1e-3)
    core_np = 5
    bs = 0.4
    le = 504e-3
    ve = 593600e-9
    w_core = 30e-3
    w_hole = 70e-3
    l_core = 5*47e-3-7e-3
    h_hole = 126e-3
    # -------------winding parameter
    d0 = 0.1e-3 # Single strand diameter
    dc_single_wire = 4.4e-3 # Single Litz wire diameter
    n_single_wire = 1050 # Number of strands in a single wire
    wire_np = 10
    dc = dc_single_wire*np.sqrt(wire_np*4/math.pi)
    n0 = wire_np*n_single_wire
    n = n1
    m_max = np.floor(w_hole/2/dc_single_wire)
    m_min = np.ceil(n*dc/h_hole)
    m = m_min
    l = 2*w_core + 2*l_core + 8*dc*m + 4*10e-3
    # -------------waveform parameters
    pv = CoreLoss(um_vec, t_vec, k, alpha, beta, n1, ae*core_np, ki)
    p_core = pv*ve*core_np
    copper_loss1 = CopperLoss(il_rms, fs, d0, dc, n0, 2*n, 2*m, l)
    rou = 1/59/6e6
    mu0 = 4*math.pi*1e-7
    d02 = 5e-3
    dc2 = 5e-3
    n02 = 1
    n2 = 1
    m2 = 1
    l2 = 4
    delta = math.sqrt(rou/math.pi/mu0/fs)
    zeta = d02/delta
    p = posai(zeta, 50)
    kd = zeta/2/math.sqrt(2)*p
    rdc = l2*4*n2*rou/n02/math.pi/d02/d02
    rac = kd*rdc
    copper_loss2 = il_rms*il_rms*rac
    ptr_cond = copper_loss1+copper_loss2
    return ptr_cond, p_core
    
    
    
def KiCal(k, alpha, beta, step):
    x = np.linspace(0,step,2*math.pi)
    y = np.abs(np.power(np.cos(x),alpha)*np.power(2,beta-alpha))
    integral = np.trapz(y,x)
    return k/np.power(2*math.pi,alpha-1)/integral
    
    
def CoreLoss(um, t, k, alpha, beta, n1, ae, ki):
    parse_step = 10
    t = t[np.linspace(0,t[-1],parse_step)]
    um = um[np.linspace(0,um[-1],parse_step)]
    dt = t[1]-t[0]
    u_integ = np.zeros(np.shape(um))
    u_integ[0] = um[0]
    for i in range(1,np.shape(um)):
        u_integ[i] = u_integ[i-1] + um[i]
    b = dt*u_integ/n1/ae
    min_b = np.min(b)
    max_b = np.max(b)
    b_extend1 = np.column_stack(b[-1],b[0:-2])
    b_extend2 = np.column_stack(b[1:-1],b[0])
    t_extend1 = np.column_stack(t[0]-dt,t[0:-1])
    t_extend2 = np.column_stack(t[1:-1],t[-1]+dt)
    db = (b_extend2-b_extend1)/(t_extend2-t_extend1)
    integral = np.trapz(ki*np.abs(np.power(db,alpha)*np.power((max_b-min_b),(beta-alpha))),t)
    return integral/(t[-1]-t[0])

def wireResis(f, d0, dc, n0, n, m, l):
    rou = 1/59.6e6
    mu0 = 4*math.pi*1e-7
    b = np.power((d0/2),2)*n0/np.power((dc/2),2)
    delta = np.sqrt(rou/math.pi/mu0/f)
    
def CalculateSwitchConductionLoss(il_vec,vout_vec,t_vec):
    wpri_cond = 