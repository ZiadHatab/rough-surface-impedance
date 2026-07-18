"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Example comparing different probability distribution functions of the roughness on the surface impedance.
"""

import numpy as np
import matplotlib.pyplot as plt

# my code for computing surface impedance. Should be in same folder as this script
import surfz

if __name__ == '__main__':
    # frequency grid
    f   = np.logspace(-1, 2, 100)*1e9
    
    Rrms = 1e-6 
    material_properties = [{'sigma': 0}, {'sigma': 58e6}]
    Zs_norm = surfz.surface_impedance(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='norm')
    Zs_rayleigh = surfz.surface_impedance(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='rayleigh')
    Zs_uniform  = surfz.surface_impedance(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='uniform')
    
    # Plot surface impedance (real part)
    plt.figure()
    plt.plot(f*1e-9, Zs_norm.real, lw=2, label=f'Normal, Rrms={Rrms*1e6:.1f}um', linestyle='solid')
    plt.plot(f*1e-9, Zs_rayleigh.real, lw=2, label=f'Rayleigh, Rrms={Rrms*1e6:.1f}um', linestyle='dashed')
    plt.plot(f*1e-9, Zs_uniform.real, lw=2, label=f'Uniform, Rrms={Rrms*1e6:.1f}um', linestyle='dashdot')
    plt.title("Surface Impedance Real Part")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Surface Impedance (Ohm)")
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.legend(loc='upper right')

    # Plot surface impedance (imaginary part)
    plt.figure()
    plt.plot(f*1e-9, Zs_norm.imag, lw=2, label=f'Normal, Rrms={Rrms*1e6:.1f}um', linestyle='solid')
    plt.plot(f*1e-9, Zs_rayleigh.imag, lw=2, label=f'Rayleigh, Rrms={Rrms*1e6:.1f}um', linestyle='dashed')
    plt.plot(f*1e-9, Zs_uniform.imag, lw=2, label=f'Uniform, Rrms={Rrms*1e6:.1f}um', linestyle='dashdot')
    plt.title("Surface Impedance Imaginary Part")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Surface Impedance (Ohm)")
    plt.xlim([0, 100])
    plt.ylim([0, 3])
    plt.legend(loc='upper right')

    sigma_copper = 58e6
    Zs_smooth = surfz.smooth_surface_impedance(f, sigma_copper)  # reference smooth surface impedance based on copper
    
    # Normal
    sigma_eff_norm = sigma_copper*(Zs_smooth.real/Zs_norm.real)**2
    mur_eff_norm   = (Zs_norm.imag/Zs_smooth.real)**2
    
    # Rayleigh
    sigma_eff_rayleigh = sigma_copper*(Zs_smooth.real/Zs_rayleigh.real)**2
    mur_eff_rayleigh   = (Zs_rayleigh.imag/Zs_smooth.real)**2
    
    # Uniform
    sigma_eff_uniform = sigma_copper*(Zs_smooth.real/Zs_uniform.real)**2
    mur_eff_uniform   = (Zs_uniform.imag/Zs_smooth.real)**2

    # Plot effective conductivity
    plt.figure()
    plt.plot(f*1e-9, sigma_eff_norm/1e6, lw=2, label=f'Normal, Rrms={Rrms*1e6:.1f}um ')
    plt.plot(f*1e-9, sigma_eff_rayleigh/1e6, '--', lw=2, label=f'Rayleigh, Rrms={Rrms*1e6:.1f}um')
    plt.plot(f*1e-9, sigma_eff_uniform/1e6, '-.', lw=2, label=f'Uniform, Rrms={Rrms*1e6:.1f}um')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Conductivity (Ms/m)')
    plt.title("Effective Conductivity")
    plt.xlim([0, 100])
    plt.ylim([0, 60])
    plt.legend()

    # Plot relative effective permeability
    plt.figure()
    plt.plot(f*1e-9, mur_eff_norm, lw=2, label=f'Normal, Rrms={Rrms*1e6:.1f}um ')
    plt.plot(f*1e-9, mur_eff_rayleigh, '--', lw=2, label=f'Rayleigh, Rrms={Rrms*1e6:.1f}um')
    plt.plot(f*1e-9, mur_eff_uniform, '-.', lw=2, label=f'Uniform, Rrms={Rrms*1e6:.1f}um')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative Permeability (Unitless)')
    plt.title("Relative Effective Permeability")
    plt.xlim([0, 100])
    plt.ylim([0, 1200])
    plt.legend()

    
    plt.show()

# EOF