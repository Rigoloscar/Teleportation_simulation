import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import qutip as qt
import numpy as np

#=========================================================================================================================#
#===================================================== DEFINITIONS =======================================================#

# Parameters
# ----------
wavelength = 0.75                                   # wavelength of the pump photon (um)
length = 1000                                       # length of the crystall (um)
theta_offset = 2                                    # tilt of the crystall with respect to the angle for maximum intensity of outcoming photons (degrees)
mu = 0.2                                            # Brightness (photon pairs per pulse)


#=========================================================================================================================#
#======================================================= FUNCTIONS =======================================================#

# Sellmeier equations
# -------------------
# These equations are only valid for beta_GaAs BBO crystals.

# Ordinary refractive index
def no_pure(wavelength):
    return(np.sqrt(2.7405 + (0.0184/((wavelength**2)-0.0179)) - 0.0155*(wavelength**2)))

# Pure extraordinary refractive index
def ne_pure(wavelength):
    return(np.sqrt(2.3730 + (0.0128/((wavelength**2)-0.0156)) - 0.0044*(wavelength**2)))

# Real extraordinary refractive index
def ne_theta(wavelength, theta):

    no = no_pure(wavelength)
    ne = ne_pure(wavelength)
    theta_rad = np.radians(theta)

    return(np.sqrt((ne**2 * no**2) / (ne**2 * np.cos(theta_rad)**2 + no**2 * np.sin(theta_rad)**2)))


# Phase matching condition
# ------------------------
# Phase matching condition for type II SPDC (e -> o + e)
def Phase_matching_function_type_II(wavelength):

    signal_wavelength = 2*wavelength
    signal_no = no_pure(signal_wavelength)
    
    # find the angle where pump_ne = 0.5 * (signal_no + idler_ne)
    objective = lambda theta: ne_theta(wavelength, theta) - 0.5 * (signal_no + ne_theta(signal_wavelength, theta))
    angle_solution = fsolve(objective, 30)
    return angle_solution


# Intensity simulations and plotting
# ----------------------------------
# 1D intensity sinc curve for type II SPDC
def Intensity_function_1D_type_II(wavelength, length):

    theta = Phase_matching_function_type_II(wavelength)
    angles_1D = np.linspace(theta-2, theta+2, 500)
    
    signal_wavelength = 2*wavelength
    kp_1D = (2*np.pi*ne_theta(wavelength, angles_1D))/wavelength
    ks_1D = (2*np.pi*no_pure(signal_wavelength))/signal_wavelength
    ki_1D = (2*np.pi*ne_theta(signal_wavelength, angles_1D))/signal_wavelength

    delta_k_1D = kp_1D - ks_1D - ki_1D
    intensity_1D = np.sinc((delta_k_1D*length)/(2*np.pi))**2

    return(angles_1D, intensity_1D, theta)


# Intensity 2D cone section for type II SPDC
def Intensity_function_2D_type_II(wavelength, length, theta_offset):

    theta = Phase_matching_function_type_II(wavelength) 
    signal_wavelength = 2*wavelength

    res = 1000
    limit = 0.15 # radians
    x = np.linspace(-limit, limit, res)
    y = np.linspace(-limit, limit, res)
    X, Y = np.meshgrid(x, y)

    # Computation of the spatial walk-off (rho)
    d_theta = 0.0001
    n_theta = ne_theta(wavelength, (theta + theta_offset))
    n_theta_plus = ne_theta(wavelength, (theta + theta_offset + d_theta))
    dn_dtheta = (n_theta_plus - n_theta) / np.radians(d_theta)
    rho = - (1 / n_theta) * dn_dtheta

    # Computation of the temporal walk-off
    no = no_pure(signal_wavelength)
    ne = n_theta
    c = 0.299792
    delta_t = (length * abs((1/ne) - (1/no))) / c

    # Ordinary and extraordinary set of angles for the simulation 
    angles_2D_o = np.sqrt(X**2 + Y**2)
    angles_2D_e = np.sqrt(X**2 + (Y - rho)**2)

    # Refractive indexes: pump -> e ; signal -> o ; idler -> e
    kp_2D = (2*np.pi*ne_theta(wavelength, (theta+theta_offset)))/wavelength 
    ks_2D = (2*np.pi*no_pure(signal_wavelength))/signal_wavelength 
    ki_2D = (2*np.pi*ne_theta(signal_wavelength, (theta+theta_offset)))/signal_wavelength 

    # Ordinary and extraordinary cones
    delta_k_2D_o = kp_2D - (ks_2D + ki_2D)*np.cos(angles_2D_o) 
    delta_k_2D_e = kp_2D - (ks_2D + ki_2D) * np.cos(angles_2D_e) 
    intensity_2D = np.sinc((delta_k_2D_o * length) / (2 * np.pi))**2 + np.sinc((delta_k_2D_e * length) / (2 * np.pi))**2

    return(intensity_2D, (theta + theta_offset), limit, rho, delta_t)


# Entanglement simulation
# -----------------------
# Simulates the quantum entanglement and computes a realistic density matrix. Both multi-photon emission and decoherence
# due to the temporal walk off have been considered.
 
def Entanglement_Computation(mu, delta_t):

    # Ideal density matrix computation
    H = qt.basis(2, 0)                                  
    V = qt.basis(2, 1)                                  
    psi = (qt.tensor(H, V) + qt.tensor(V, H)).unit()    
    density_ideal = qt.ket2dm(psi)         
    
    # Multi-photon emission
    p1 = (mu**1 * np.exp(-mu)) / 1                      
    p2 = (mu**2 * np.exp(-mu)) / 2                      
    density_1 = (p1 * density_ideal + p2 * qt.qeye_like(density_ideal).unit())
    density_1 = density_1 / density_1.tr()

    # Decoherence due to temporal walk-off
    t_coherence = 100   
    dephasing_rate = delta_t / t_coherence
    c_ops = [np.sqrt(dephasing_rate) * qt.tensor(qt.sigmaz(), qt.qeye(2)), np.sqrt(dephasing_rate) * qt.tensor(qt.qeye(2), qt.sigmaz())]
    density_2 = qt.mesolve(qt.qeye_like(density_ideal).unit()*0, density_1, [11], c_ops=c_ops).final_state
    density_real = density_2 / density_2.tr()

    return(density_ideal, density_real)


#=========================================================================================================================#
#================================================ VISUALIZATION FUNCTIONS ================================================#

# Intensity visualization
# --------------------------
# Represents the ideal sinc intensity curve with respect to the tilt angle
def Visualize_intensity_1D(angles_1D, intensity_1D, theta_ideal):
    f1 = plt.figure(1)
    plt.plot(angles_1D, intensity_1D, label=f"Crystall Length = {length} um")
    plt.axvline(theta_ideal, color="r", linestyle="--", label=f"Pump angle = {round(theta_ideal[0], 2)} deg")
    plt.title(f"SPDC Type II Simulation (Pump wavelength = {round(wavelength*1000)} nm)")
    plt.xlabel("Crystall angle")
    plt.ylabel("Relative intensity")
    plt.legend()
    plt.grid(True)


# Represents the section of the conic surfaces
def Visualize_intensity_2D(intensity_2D, theta_real, limit, rho, delta_t):
    f2 = plt.figure(2)
    plt.imshow(intensity_2D, extent=[-np.degrees(limit), np.degrees(limit), -np.degrees(limit), np.degrees(limit)], cmap="magma", origin="lower")
    plt.colorbar(label="Relative intensity")
    plt.title(f"Transversal section of SPDC Type II (Pump angle = {round(theta_real[0], 2)} deg)")
    plt.text(0, -7 , (f"Spatial walk-off (rho) = {round(np.degrees(rho[0]), 2)} deg"), fontsize = 10, color="w", ha='center')
    plt.text(0, -8, (f"Temporal walk-off = {round(delta_t[0], 2)} fs"), fontsize = 10, color="w", ha='center')
    plt.xlabel("Angle X (deg)")
    plt.ylabel("Angle Y (deg)")
    plt.grid(alpha=0.2)


#=========================================================================================================================#
#====================================================== MAIN FUNCTIONS ===================================================#

# Combined SPDC type II simulation
# --------------------------------
def Simulate_SPDC(wavelength, length, theta_offset):
    
    # Computes the intensities and the conic sections
    angles_1D, intensity_1D, theta_ideal = Intensity_function_1D_type_II(wavelength, length)
    intensity_2D, theta_real, limit, rho, delta_t = Intensity_function_2D_type_II(wavelength, length, theta_offset)
    
    # Plots the intensities and the conic sections
    Visualize_intensity_1D(angles_1D, intensity_1D, theta_ideal)
    Visualize_intensity_2D(intensity_2D, theta_real, limit, rho, delta_t)
    
    return(rho, delta_t, theta_ideal)
    

# Combined entangled photon source simulation
# -------------------------------------------
def Simulate_Source_Entanglement(mu, delta_t):

    density_ideal, density_real = Entanglement_Computation(mu, delta_t)
    fidelity = qt.fidelity(density_ideal, density_real)
    concurrence = qt.concurrence(density_real)
     
    return(density_ideal, density_real, fidelity, concurrence)

