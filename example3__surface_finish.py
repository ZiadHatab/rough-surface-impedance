"""
@Author: Ziad (https://github.com/ZiadHatab)

Example showing how to compute surface impedance of coating layers of multiple thin conductors.
I'm using a PCB surface finish as an example based on the data from [1].

References:
[1] B. Schafsteller, M. Schwaemmlein, M. Rosin, G. Ramos, Z. Hatab, M. E. Gadringer, E. Schlaffer, 
"Investigating the Impact of Final Finishes on the Insertion Loss in As Received and After Aging," 
IMAPSource Proceedings, vol. 2023, no. Symposium. IMAPS - International Microelectronics Assembly 
and Packaging Society, Feb. 29, 2024. doi: https://doi.org/10.4071/001c.94519.
"""

import numpy as np
import matplotlib.pyplot as plt
import surfz  # my code for computing surface impedance. Should be in same folder as this script

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

    ## Example of multiple stacked thin conductors, such as surface finishes on PCBs
    # Refer to the paper [1] above for the effects of surface finishes on losses in transmission lines
    # In this example, the roughness is assumed to be very small, near flat. 
    # For the sake of including roughness effects, a value of 5nm is used.

    # ENIG example: Air-Gold-Nickel-Copper
    # Thicknesses (the first and last materials are base and assumed to extend to infinity):
        # Gold: 0.05um
        # Nickel: 4um

    # ENIPIG example: Air-Gold-Palladium-Nickel-Copper
    # Thicknesses:
        # Gold: 0.1um
        # Palladium: 0.1um
        # Nickel: 4um
    
    # For the material electrical properties, I took them from Ansys EDT. You can also just google them (e.g., wikipedia)
    # Air: sigma = 0; mur = 1
    # Gold: sigma = 41.1e6 S/m; mur = 0.99996
    # Palladium: sigma = 9.3e6 S/m; mur = 1.00082
    # Nickel: sigma = 14.5e6 S/m; mur = 600
    # Copper: sigma = 58e6 S/m; mur = 0.999991
    
    f = np.logspace(-1, 2.5, 256)*1e9  # frequency grid... up to roughly 300GHz

    # Material properties (they could be frequency dependent if desired)
    material_properties_air = [{'sigma': 0,'mur': 1}]                   # air
    material_properties_gold = [{'sigma': 41e6, 'mur': 0.99996}]        # gold
    material_properties_nickel = [{'sigma': 14.5e6, 'mur': 20}]        # nickel
    material_properties_palladium = [{'sigma': 9.3e6, 'mur': 1.00082}]  # palladium
    material_properties_copper = [{'sigma': 58e6, 'mur': 0.999991}]     # copper

    material_properties_enig = material_properties_air + material_properties_gold + material_properties_nickel + material_properties_copper
    material_properties_enipig = material_properties_air + material_properties_gold + material_properties_palladium + material_properties_nickel + material_properties_copper

    # Roughness
    Rrms = 5e-9 # same for all boundaries (if different, use a list)

    # Thickness
    thickness_enig = [0, 0.05e-6, 4e-6]             # ENIG
    thickness_enig_thick_gold = [0, 0.15e-6, 4e-6]  # ENIG with thicker gold layer
    thickness_enig_thick_gold2 = [0, 0.5e-6, 4e-6]  # ENIG with even thicker gold layer
    thickness_enipig = [0, 0.1e-6, 0.1e-6, 4e-6]    # ENIPIG

    # Boundary locations
    boundary_loc_enig = np.cumsum(thickness_enig)
    boundary_loc_enig_thick_gold = np.cumsum(thickness_enig_thick_gold)
    boundary_loc_enig_thick_gold2 = np.cumsum(thickness_enig_thick_gold2)
    boundary_loc_enipig = np.cumsum(thickness_enipig)

    # Calculate surface impedance for each case
    Zs_enig, B_enig, x_enig = surfz.surface_impedance(f, material_properties_enig, Rrms=Rrms, boundary_loc=boundary_loc_enig, distribution='rayleigh', recursion_span=[-1e-6, 5e-6], return_field=True)
    Zs_enig_thick_gold  = surfz.surface_impedance(f, material_properties_enig, Rrms=Rrms, boundary_loc=boundary_loc_enig_thick_gold, distribution='rayleigh', recursion_span=[-1e-6, 5e-6])
    Zs_enig_thick_gold2 = surfz.surface_impedance(f, material_properties_enig, Rrms=Rrms, boundary_loc=boundary_loc_enig_thick_gold2, distribution='rayleigh', recursion_span=[-1e-6, 5e-6])
    Zs_enipig, B_enipig, x_enipig = surfz.surface_impedance(f, material_properties_enipig, Rrms=Rrms, boundary_loc=boundary_loc_enipig, distribution='rayleigh', recursion_span=[-1e-6, 5e-6], return_field=True)
    
    sigma_copper = 58e6
    Zs_smooth = Zsmooth(f, sigma_copper, 1)  # reference smooth surface impedance based on copper
    
    # Plot surface impedance (real part)
    plt.figure()
    plt.plot(f*1e-9, Zs_enig.real, lw=2, label='ENIG, Gold=0.05um, Nickel=4um', linestyle='solid')
    plt.plot(f*1e-9, Zs_enig_thick_gold.real, lw=2, label='ENIG, Gold=0.15um, Nickel=4um', linestyle='dashed')
    plt.plot(f*1e-9, Zs_enig_thick_gold2.real, lw=2, label='ENIG, Gold=0.5um, Nickel=4um', linestyle='dashdot')
    plt.plot(f*1e-9, Zs_enipig.real, lw=2, label='ENIPIG, Gold=0.1um, Palladium=0.1um, Nickel=4um', linestyle=(0, (3, 1, 1, 1)))
    plt.plot(f*1e-9, Zs_smooth.real, lw=2, label='Smooth Copper', linestyle='--')
    plt.title("Surface Impedance Real Part")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Surface Impedance (Ohm)")
    plt.xlim([0, 300])
    plt.ylim([0, .8])
    plt.legend(loc='upper right')

    # Plot surface impedance (imaginary part)
    plt.figure()
    plt.plot(f*1e-9, Zs_enig.imag, lw=2, label='ENIG, Gold=0.05um, Nickel=4um', linestyle='solid')
    plt.plot(f*1e-9, Zs_enig_thick_gold.imag, lw=2, label='ENIG, Gold=0.15um, Nickel=4um', linestyle='dashed')
    plt.plot(f*1e-9, Zs_enig_thick_gold2.imag, lw=2, label='ENIG, Gold=0.5um, Nickel=4um', linestyle='dashdot')
    plt.plot(f*1e-9, Zs_enipig.imag, lw=2, label='ENIPIG, Gold=0.1um, Palladium=0.1um, Nickel=4um', linestyle=(0, (3, 1, 1, 1)))
    plt.plot(f*1e-9, Zs_smooth.imag, lw=2, label='Smooth Copper', linestyle='--')
    plt.title("Surface Impedance Imaginary Part")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Surface Impedance (Ohm)")
    plt.xlim([0, 300])
    plt.ylim([0, .3])
    plt.legend(loc='upper right')

    # Calculate effective conductivity and relative effective permeability
    sigma_eff_enig = sigma_copper*(Zs_smooth.real/Zs_enig.real)**2
    mur_eff_enig = (Zs_enig.imag/Zs_smooth.real)**2

    sigma_eff_enig_thick_gold = sigma_copper*(Zs_smooth.real/Zs_enig_thick_gold.real)**2
    mur_eff_enig_thick_gold = (Zs_enig_thick_gold.imag/Zs_smooth.real)**2

    sigma_eff_enig_thick_gold2 = sigma_copper*(Zs_smooth.real/Zs_enig_thick_gold2.real)**2
    mur_eff_enig_thick_gold2 = (Zs_enig_thick_gold2.imag/Zs_smooth.real)**2

    sigma_eff_enipig = sigma_copper*(Zs_smooth.real/Zs_enipig.real)**2
    mur_eff_enipig = (Zs_enipig.imag/Zs_smooth.real)**2

    # Plot effective conductivity
    plt.figure()
    plt.plot(f*1e-9, sigma_eff_enig/1e6, lw=2, label='ENIG, Gold=0.05um, Nickel=4um', linestyle='solid')
    plt.plot(f*1e-9, sigma_eff_enig_thick_gold/1e6, lw=2, label='ENIG, Gold=0.15um, Nickel=4um', linestyle='dashed')
    plt.plot(f*1e-9, sigma_eff_enig_thick_gold2/1e6, lw=2, label='ENIG, Gold=0.5um, Nickel=4um', linestyle='dashdot')
    plt.plot(f*1e-9, sigma_eff_enipig/1e6, lw=2, label='ENIPIG, Gold=0.1um, Palladium=0.1um, Nickel=4um', linestyle=(0, (3, 1, 1, 1)))
    plt.hlines(41.1, min(f)*1e-9, max(f)*1e-9, lw=2, linestyles='--', color='black', label='Gold conductivity')
    plt.title("Effective Conductivity")
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Conductivity (Ms/m)')
    plt.xlim([0, 300])
    plt.ylim([0, 70])
    plt.legend(loc='upper right')

    # Plot relative effective permeability
    plt.figure()
    plt.semilogy(f*1e-9, mur_eff_enig, lw=2, label='ENIG, Gold=0.05um, Nickel=4um', linestyle='solid')
    plt.semilogy(f*1e-9, mur_eff_enig_thick_gold, lw=2, label='ENIG, Gold=0.15um, Nickel=4um', linestyle='dashed')
    plt.semilogy(f*1e-9, mur_eff_enig_thick_gold2, lw=2, label='ENIG, Gold=0.5um, Nickel=4um', linestyle='dashdot')
    plt.semilogy(f*1e-9, mur_eff_enipig, lw=2, label='ENIPIG, Gold=0.1um, Palladium=0.1um, Nickel=4um', linestyle=(0, (3, 1, 1, 1)))
    plt.hlines(1, min(f)*1e-9, max(f)*1e-9, lw=2, linestyles='--', color='black', label='Gold relative permeability')
    plt.title("Effective Relative Permeability")
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative Permeability (Unitless)')
    plt.xlim([0, 300])
    plt.ylim([1e-1, 1e3])
    plt.legend()

    
    # plot field penetration depth for enig
    plt.figure()
    freq = [1e9, 10e9, 100e9]
    for ff in freq:
        idx = np.argmin(abs(f - ff))
        plt.semilogy(x_enig*1e6, abs(B_enig[idx]), lw=1.5, label=f'{ff*1e-9:.1f} GHz', linestyle='solid')
        #plt.semilogy(x_enipig*1e6, abs(B_enipig[idx]), lw=2, label=f'ENIPIG, Gold=0.1um, Palladium=0.1um, Nickel=4um {ff*1e-9:.2f}GHz', linestyle=(0, (3, 1, 1, 1)))
    higlight_area = [[-1, 0], [0, 0.05], [0.05, 4.05], [4.05, 5]]
    colors_area = np.array([(166, 231, 255), (212,175,55), (181, 182, 181), (184, 115, 51)])/255
    text_area = ['Air', 'Gold', 'Nickel', 'Copper']
    for inx, xspan in enumerate(higlight_area):
        plt.axvspan(*xspan, color=colors_area[inx], alpha=0.4, lw=0)
        plt.text(np.mean(xspan), 1.02, text_area[inx], ha='center', va='bottom')
    plt.xlabel('Distance (um)')
    plt.ylabel('Normalized Magnetic Field')
    plt.ylim([1e-21, 100])
    plt.xlim([-1, 5])
    plt.title('ENIG: Gold=0.05um, Nickel=4um')
    plt.legend(loc='lower left', ncol=3)

    # plot field penetration depth for enipig
    plt.figure()
    freq = [1e9, 10e9, 100e9]
    for ff in freq:
        idx = np.argmin(abs(f - ff))
        plt.semilogy(x_enipig*1e6, abs(B_enipig[idx]), lw=1.5, label=f'{ff*1e-9:.1f} GHz', linestyle='solid')
    higlight_area = [[-1,0], [0, 0.1], [0.1, 0.2], [0.2, 4.2], [4.2, 5]]
    colors_area = np.array([(166, 231, 255), (212,175,55), (207,205,201), (181, 182, 181), (184, 115, 51)])/255
    text_area = ['Air', 'Gold', 'Palladium', 'Nickel', 'Copper']
    for inx, xspan in enumerate(higlight_area):
        plt.axvspan(*xspan, color=colors_area[inx], alpha=0.4, lw=0)
        y = 10 if inx == 2 else 1.1
        xoffset = 0.2 if inx == 2 else -0.1 if inx == 1 else 0
        plt.text(np.mean(xspan)+xoffset, y, text_area[inx], ha='center', va='bottom')
    plt.xlabel('Distance (um)')
    plt.ylabel('Normalized Magnetic Field')
    plt.ylim([1e-21, 100])
    plt.xlim([-1, 5])
    plt.title('ENIPIG: Gold=0.1um, Palladium=0.1um, Nickel=4um')
    plt.legend(loc='lower left', ncol=3)

    
    plt.show()

# EOF