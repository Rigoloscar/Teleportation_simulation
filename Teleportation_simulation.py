import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import qutip as qt
import numpy as np
import random
from SPDC_source_module import Simulate_SPDC
from SPDC_source_module import Simulate_Source_Entanglement
from Optical_module import Linear_Polarizer
from Optical_module import Generic_Waveplate
from Optical_module import Beam_Splitter_HOM
from Optical_module import Polarizing_Beam_Splitter


#=========================================================================================================================#
#===================================================== DEFINITIONS =======================================================#

# Source Parameters
# -----------------                                 
mu = 0.2                            # Brightness of the source (photon pairs per pulse)
delta_t = 21                        # Temporal walk-off of the source (fs)

# Optical components parameters
# -----------------------------
HWP_reflectance = 0.5               # Reflectance of the Half-Waveplate (%)
HWP_retardance_accuracy = 100       # Retardance accuracy of the Half-Waveplate (lambda dividing factor)
QWP_reflectance = 0.5               # Reflectance of the Quarter-Waveplate (%)
QWP_retardance_accuracy = 100       # Retardance accuracy of the Quarter-Waveplate (lambda dividing factor)
POL_reflectance = 0.25              # Reflectance of the linear polarizer (%)
POL_extinction_ratio = 10000        # Extinction ratio of the linear polarizer
BS_reflectance = 0.25               # Reflectance of the Beam Splitter (%)
BS_split_ratio_R = 0.5              # Reflection split ratio of the Beam Splitter
BS_split_ratio_T = 0.5              # Transmission split ratio of the Beam Splitter
PBS_extinction_ratio_T = 10000      # Extinction ratio in transmission of the Polarized Beam Splitter
PBS_extinction_ratio_R = 10000      # Extinction ratio in reflection of the Polarized Beam Splitter
PBS_reflectance = 0.25              # Reflectance of the Polarized Beam Splitter (%)

#=========================================================================================================================#
#================================================= INTERFACE FUNCTIONS ===================================================#

# Block diagram display function
# ------------------------------
def Display_block_diagram():
    plt.figure(3)
    img = mpimg.imread('block.png')
    imgplot = plt.imshow(img)
    plt.title("Block diagram of the teleportation")
    plt.axis('off')
    plt.tight_layout(pad = 1)
    

# Hinton representation display function
# --------------------------------------
def Display_hinton(density, title, label_size):
    fig, ax = plt.subplots()
    qt.hinton(density, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis = 'both', which = 'major', labelsize = label_size)


#=========================================================================================================================#
#================================================== BENCH 1 FUNCTIONS ====================================================#

# Node 1 functions: generation of X1 and state preparation
# --------------------------------------------------------
# This is what node 1 is in the block diagram. Alice's photon X1 is generated via SPDC type II 
# and is prepared in an arbitrary quantum state to be teleported

def Node_1():
                                       
    global mu 
    global delta_t 
    global HWP_reflectance               
    global HWP_retardance_accuracy
    global POL_reflectance 
    global POL_extinction_ratio

    # Generation of X1 via type II SPDC
    src_density_ideal, src_density, src_fidelity, src_concurrence = Simulate_Source_Entanglement(mu, delta_t)
    density_1 = src_density.ptrace(0)
    
    # X1 is polarized in the vertical axis
    polarizer = Linear_Polarizer(0, POL_reflectance, POL_extinction_ratio)
    operator = polarizer
    density_2 = operator * density_1 * operator.dag()
    density_2 = density_2 / density_2.tr()
    
    # X1 is rotated 45 degrees
    waveplate = Generic_Waveplate('Half', 15, HWP_reflectance, HWP_retardance_accuracy)
    operator = waveplate
    density_3 = operator * density_2 * operator.dag()
    density_3 = density_3 / density_3.tr()
    
    # X1 gets a phase shift
    waveplate = Generic_Waveplate('Half', 0, HWP_reflectance, HWP_retardance_accuracy)
    operator = waveplate
    density_4 = operator * density_3 * operator.dag()
    density_4 = density_4 / density_4.tr()
    
    return(density_4)
    

# Node 2 functions: generation of X2 and XX2, and preparation of both
# -------------------------------------------------------------------
# This is what node 2 is in the block diagram. Here the epr pair is generated via SPDC type II in the psi+ state. 
# In order to have the same bell state as they did in Sapienza one of the photons is rotated to get the phi+ state.

def Node_2():

    global mu
    global delta_t
    global HWP_reflectance               
    global HWP_retardance_accuracy
    global POL_reflectance 

    # Generation of X2 and XX2 via type II SPDC
    src_density_ideal, src_density, src_fidelity, src_concurrence = Simulate_Source_Entanglement(mu, delta_t)
    density_1 = src_density

    # Use of Half-Waveplates to prepare the Bell State psi -
    waveplate = Generic_Waveplate('Half', 45, HWP_reflectance, HWP_retardance_accuracy)
    operator = qt.tensor(waveplate, qt.qeye(2))
    density_2 = operator * density_1 * operator.dag()
    density_2 = density_2 / density_2.tr()

    return(density_2)
    

# Bell State measurement function: HOM interference and psi- projection
# ---------------------------------------------------------------------
# Here the HOM interference is done using the defined beam splitter quantum operator in [1]. 
# The combined state of the three qbits enter the function and X1 and X2 interfere. Later on a psi- measurement
# is performed using a PBS that has two different operators, one for the transmission port and another for the 
# reflection.

def Bell_State_measurement(src_density):

    global BS_reflectance 
    global BS_split_ratio_R 
    global BS_split_ratio_T 
    global PBS_extinction_ratio_T 
    global PBS_extinction_ratio_R 
    global PBS_reflectance 

    # Definition of the used optical elements 
    beam_splitter = Beam_Splitter_HOM(BS_split_ratio_R, BS_split_ratio_T, BS_reflectance)
    pbs_t, pbs_r = Polarizing_Beam_Splitter(0, PBS_extinction_ratio_T, PBS_extinction_ratio_R, PBS_reflectance)

    # Performs the HOM interference at the beam splitter
    density_interfered = beam_splitter * src_density * beam_splitter.dag()
    density_interfered = density_interfered / density_interfered.tr()

    # Definition of the two possible events that define psi- projection 
    event_operator_1 = qt.tensor(pbs_t, pbs_r, qt.qeye(2))
    event_operator_2 = qt.tensor(pbs_r, pbs_t, qt.qeye(2))

    # One of these events will happen with a 50% chance each
    event = random.randint(1, 2)

    if(event == 1):
        density_1 = event_operator_1 * density_interfered * event_operator_1.dag()
        bob_density = density_1.ptrace(2)
        bob_density = bob_density / bob_density.tr()

    elif(event == 2):
        density_2 = event_operator_2 * density_interfered * event_operator_2.dag()
        bob_density = density_2.ptrace(2)
        bob_density = bob_density / bob_density.tr()

    return(bob_density, event)


#=========================================================================================================================#
#================================================== BENCH 2 FUNCTIONS ====================================================#

# Pauli correction function: applies the appropiate Pauli correction matrices 
# ---------------------------------------------------------------------------
# Here the Pauli corrections are applied. Depending on which of the two events that define the psi- measurement the matrix
# sigma_z (phase shift) is applied alongside sigma_x (bit flip).

def Pauli_correction(src_density, event):

    global HWP_reflectance               
    global HWP_retardance_accuracy

    # Defines the sigma_y matrix as a bit flip with a global phase
    waveplate = Generic_Waveplate('Half', 45, HWP_reflectance, HWP_retardance_accuracy)
    operator = waveplate
    density_1 = operator * src_density * operator.dag()
    density_1 = density_1 / density_1.tr()

    # Doesn't need an aditional sigma_z matrix to do the phase shift
    if(event == 1):
        density = density_1

    # Needs an aditional sigma_z matrix to do the phase shift
    elif(event == 2):
        waveplate = Generic_Waveplate('Half', 0, HWP_reflectance, HWP_retardance_accuracy)
        operator = waveplate
        density = operator * density_1 * operator.dag()
        density = density / density.tr()

    return(density)


#=========================================================================================================================#
#========================================================== MAIN =========================================================#

# Generation of the single photon and EPR pair
# --------------------------------------------
epr_density = Node_2()
alice_density = Node_1()

# Bell State measurement
# ----------------------
density_three_qbit = qt.tensor(alice_density, epr_density)
bob_density, event = Bell_State_measurement(density_three_qbit)

# Pauli corrections
# -----------------
corrected_density = Pauli_correction(bob_density, event)

# Plots
# -----
Display_hinton(alice_density, "Alice's qbit after node 1", 10)
Display_hinton(bob_density, "Bob's qbit after BSM", 10)
Display_hinton(epr_density, "EPR qbits after node 2", 10)
Display_hinton(corrected_density, "Bob's qbit after Pauli correction", 10)
fidelity = qt.fidelity(alice_density, corrected_density)
print(fidelity)
plt.show()



