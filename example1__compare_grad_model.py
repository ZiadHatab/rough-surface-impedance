"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Example comparing the results of the gradient model [1] and the transmission line taper approach [2]

References:
[1] G. Gold and K. Helmreich, "A Physical Surface Roughness Model and Its Applications," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 65, no. 10, pp. 3720-3732, 
Oct. 2017, doi: https://doi.org/10.1109/TMTT.2017.2695192.

[2] B. Tegowski, T. Jaschke, A. Sieganschin and A. F. Jacob, 
"A Transmission Line Approach for Rough Conductor Surface Impedance Analysis," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 71, 
no. 2, pp. 471-479, Feb. 2023, doi: https://doi.org/10.1109/TMTT.2022.3206440
"""

import numpy as np
import matplotlib.pyplot as plt

# my code for computing surface impedance. Should be in same folder as this script
import surfz 
import gradmodel

if __name__ == '__main__':
    # frequency grid
    f   = np.logspace(-1, 2, 100)*1e9
    
    # Rough conductor (normal distribution)  
    Rrms = 1e-6  # roughness RMS value (standard deviation)
    Zs_rough_grad, B_rough, x = gradmodel.surface_impedance(f, sigma1=0, sigma2=58e6, N=128, distribution='norm', 
                                                            Rrms=Rrms, integration_span=[-5*Rrms, 10*Rrms])
    # Smooth conductor  
    Zs_smooth, B_smooth, x = gradmodel.surface_impedance(f, sigma1=0, sigma2=58e6, N=128, distribution='norm', 
                                                         Rrms=0, integration_span=[-5*Rrms, 10*Rrms])
    
    # TL model
    material_properties = [{'sigma': 0}, {'sigma': 58e6}]
    Zs_rough_TL, B_rough_TL, x_TL = surfz.surface_impedance(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='norm',
                                                            recursion_span=[-5*Rrms, 10*Rrms], return_field=True)
    
    # Plot normalized magnetic field magnitude
    plt.figure()
    for ff in [1e9, 10e9, 100e9]:
        idx = np.isclose(f, ff).nonzero()[0][0]
        plt.plot(x*1e6, abs(B_rough[idx]), lw=2, label=f'Gradient model, {Rrms*1e6:.1f}um, {ff*1e-9:.2f} GHz')
    for ff in [1e9, 10e9, 100e9]:
        idx = np.isclose(f, ff).nonzero()[0][0] 
        plt.plot(x_TL*1e6, abs(B_rough_TL[idx]), '-.', lw=2, label=f'TL model, {Rrms*1e6:.1f}um, {ff*1e-9:.2f} GHz') 
    for ff in [1e9, 10e9, 100e9]:
        idx = np.isclose(f, ff).nonzero()[0][0]
        plt.plot(x*1e6, abs(B_smooth[idx]), '--', lw=2, label=f'Smooth conductor, {ff*1e-9:.1f} GHz')
    plt.legend()
    plt.xlabel('Distance (µm)')
    plt.ylabel('Normalized Magnetic Field')
    plt.ylim([-0.1, 1.1])
    plt.xlim([-5, 10])

    # Plot surface impedance
    plt.figure()
    plt.plot(f*1e-9, Zs_rough_grad.real, lw=2, label=f'Re(Zs), {Rrms*1e6:.2f}um, Gradient model')
    plt.plot(f*1e-9, Zs_rough_grad.imag, lw=2, label=f'Im(Zs), {Rrms*1e6:.2f}um, Gradient model')
    plt.plot(f*1e-9, Zs_rough_TL.real, '--', lw=2, label=f'Re(Zs), {Rrms*1e6:.2f}um, TL model')
    plt.plot(f*1e-9, Zs_rough_TL.imag, '-.', lw=2, label=f'Im(Zs), {Rrms*1e6:.2f}um, TL model')
    plt.plot(f*1e-9, Zs_smooth.real, '--', lw=2, label='Re(Zs), smooth')
    plt.plot(f*1e-9, Zs_smooth.imag, '-.', lw=2, label='Im(Zs), smooth')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Surface Impedance (Ω)')
    plt.xlim([0, 100])
    plt.ylim([0, 3])

    
    plt.show()

# EOF