import matplotlib.pyplot as plt
import qutip as qt
import numpy as np


#=========================================================================================================================#
#====================================================== FUNCTIONS ========================================================#

# Generic Waveplate function
# --------------------------
# This function models a generic waveplate with a certain retardance (pi for HWP and pi/2 for QWP) considering attenuation 
# the error of said retardance. Returns the Jones operator for that particular waveplate.

def Generic_Waveplate(device_type, angle, reflectance, retardance_accuracy): 
    
    angle_rad = np.radians(angle)
    phase_error = 2*np.pi / retardance_accuracy
    losses = np.sqrt(1 - (reflectance/100))
  
    # Retardance and error for a Half-Waveplate
    if(device_type == 'Half'):
        phase = np.exp(1j*(np.pi + phase_error))
    
    # Retardance and error for a Quarter-Waveplate
    elif(device_type == 'Quarter'):
        phase = np.exp(1j*(np.pi/2 + phase_error))

    # Generic Jones matrix for a waveplate
    M11 = np.cos(angle_rad)**2 + phase * np.sin(angle_rad)**2
    M12 = (1 - phase) * np.cos(angle_rad) * np.sin(angle_rad)
    M21 = (1 - phase) * np.cos(angle_rad) * np.sin(angle_rad)
    M22 = np.sin(angle_rad)**2 + phase * np.cos(angle_rad)**2
    
    waveplate_losseless = qt.Qobj([[M11, M12], [M21, M22]])
    waveplate = losses*waveplate_losseless
    
    return(waveplate)


# Linear polarizer function
# -------------------------
# This function models a linear polarizer oriented at a certain angle. The matrix M_base defines the Jones
# operator of the LP at 0º with some extinction ratio. Later on a rotating matrix is applied.

def Linear_Polarizer(angle, reflectance, extinction_ratio):
    
    angle_rad = np.radians(angle)
    losses = np.sqrt(1 - (reflectance/100))
    
    # Base matrix and rotation matrix 
    M_base = qt.Qobj([[1, 0], [0, (1/ np.sqrt(extinction_ratio))]])
    R = qt.Qobj([[np.cos(angle_rad), np.sin(angle_rad)], 
                 [-np.sin(angle_rad), np.cos(angle_rad)]])
    
    polarizer_losseless = R.dag() * M_base * R
    polarizer = losses * polarizer_losseless

    return(polarizer)
      

# Beam Splitter (HOM) function
# ----------------------------
# This function returns the quantum operator of a BS acording to [1]. Non ideal split ratios can be modeled 
# changing the angle theta (pi/4 for 50:50, for reference).

def Beam_Splitter_HOM(split_ratio_R, split_ratio_T, reflectance):
    
    theta = np.arccos(np.sqrt(split_ratio_T))
    losses = np.sqrt(1 - (reflectance/100))
    
    # Defines the losseless Beam Splitter Operator 
    a  = qt.tensor(qt.destroy(2), qt.qeye(2), qt.qeye(2))
    b1 = qt.tensor(qt.qeye(2), qt.destroy(2), qt.qeye(2))
    b2 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(2))
    beam_splitter_losseless = (-1j * theta * (a.dag() * b1 + a * b1.dag())).expm()
    beam_splitter = losses * beam_splitter_losseless
    
    return(beam_splitter)


# Polarizing Beam Splitter function
# ---------------------------------
# This function models the Polarizing Beam Splitters as two operators, one for the transmission mode and
# one for the reflection mode, similar to two linear polarizers orthogonal to eachother.

def Polarizing_Beam_Splitter(angle, extinction_ratio_T, extinction_ratio_R, reflectance):
    
    angle_rad = np.radians(angle)
    losses = np.sqrt(1 - (reflectance/100))
    
    # Extinction ratio amplitudes
    delta_t = 1 / np.sqrt(extinction_ratio_T)
    delta_r = 1 / np.sqrt(extinction_ratio_R)
    
    # Base matrixes and rotation matrix
    m_t_base = qt.Qobj([[1, 0], [0, delta_t]])
    m_r_base = qt.Qobj([[delta_r, 0], [0, 1]])
    R = qt.Qobj([[np.cos(angle_rad), np.sin(angle_rad)], 
                 [-np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Final definition of the two operators
    pbs_t = losses * (R.dag() * m_t_base * R)
    pbs_r = losses * (R.dag() * m_r_base * R)
    
    return(pbs_t, pbs_r)
   
    
    



        



