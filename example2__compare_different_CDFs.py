"""
@Author: Ziad (https://github.com/ZiadHatab)

Example comparing different probability distribution functions of the roughness on the surface impedance.
"""

import numpy as np
import matplotlib.pyplot as plt

# my code for computing surface impedance. Should be in same folder as this script
import surfz

if __name__ == '__main__':
    # constants
    c0 = 299792458   # speed of light in vacuum (m/s)
    mu0 = 4*np.pi*1e-7       # Permeability
    ep0 = 8.854187818814e-12 # Permittivity
    # useful functions
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    gamma2dbcm  = lambda x: mag2db(np.exp(x.real*1e-2))  # losses dB/cm
    time2distance = lambda x,er: x*c0/np.sqrt(er.real)
    Zsmooth = lambda f, sigma, mur: (1 + 1j)*np.sqrt(2*np.pi*f*mu0*mur/(2*sigma))
    er2sigma = lambda x, f: -ep0*x.imag*2*np.pi*f

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
    Zs_smooth = Zsmooth(f, sigma_copper, 1)  # reference smooth surface impedance based on copper
    
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