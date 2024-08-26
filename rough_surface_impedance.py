"""
@Author: Ziad (https://github.com/ZiadHatab)

A script to compute surface impedance of a rough interface of a conductor with another material (usually a dielectric).

Based on the paper about the Gradient model for roughness modeling:
[1] G. Gold and K. Helmreich, "A Physical Surface Roughness Model and Its Applications," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 65, no. 10, pp. 3720-3732, 
Oct. 2017, doi: 10.1109/TMTT.2017.2695192.

NOTE: The ODE solver can be unstable for certain roughness values and solution range.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

def conductivity(x, sigma1=.001, sigma2=58e6, Rrms=1e-6, distribution='norm'):
    """
    Profile of conductivity as function of distance from a given CDF and PDF for its derivative
        - x: (array) distance 
        - sigma1: conductivity of first medium (for dielectric sigma = tand/omega/ep0)
        - sigma2: conductivity of second medium (usually conductor, e.g., copper)
        - Rrms: standard deviation of roughness
        - distribution: probability distribution of the roughness (you can extend from scipy.stats)
    """
    
    if distribution == 'norm':
        CDF = scipy.stats.norm.cdf(x, loc=0, scale=Rrms)
        PDF = scipy.stats.norm.pdf(x, loc=0, scale=Rrms)
    elif distribution == 'rayleigh':    
        CDF = scipy.stats.rayleigh.cdf(x, loc=0, scale=Rrms)
        PDF = scipy.stats.rayleigh.pdf(x, loc=0, scale=Rrms)
    elif distribution == 'uniform':
        CDF = scipy.stats.uniform.cdf(x, loc=0, scale=Rrms)
        PDF = scipy.stats.uniform.pdf(x, loc=0, scale=Rrms)
    # add more here if you want... input parameters could differ depending on the distribution
        
    sigma = (sigma2 - sigma1)*CDF + sigma1
    sigma_diff = (sigma2 - sigma1)*PDF
    return sigma, sigma_diff

def diff_eq(x, B, *args):
    """ Differential equation describing B-fields in conductor. 
    This function is written to be passed to scipy.integrate.solve_ivp()
    """
    f = args[0]      # frequency in Hz
    sigma1 = args[1] # conductivity of first medium (S/m)
    sigma2 = args[2] # conductivity of second medium (S/m)
    Rrms   = args[3] # RMS roughness (standard deviation)
    distribution  = args[4]  # string indicating probability distribution of the roughness
    
    sigma, sigma_diff = conductivity(x, sigma1, sigma2, Rrms, distribution)
    
    mu0 = 4*np.pi*1e-7
    omega = 2*np.pi*f
    k = 1j*omega*mu0

    B_diff = B[1]
    B_diff_diff = (sigma_diff*B[1] + sigma**2*k*B[0])/sigma

    return [B_diff, B_diff_diff]

def gamma(f, sigma):
    """
    propagation constant of in a conductor with conductivity sigma 
    """
    omega = 2*np.pi*f
    mu0 = 4*np.pi*1e-7     # permeability
    ep0 = 8.854e-12        # permittivity
    ep = ep0 - 1j*sigma/omega
    gamma = omega*np.sqrt(-mu0*ep)
    return gamma*np.sign(gamma.real)  # make sure to take positive square-root

def solve_Zs_B(f, sigma1=.001, sigma2=58e6, Rrms=1e-6, integration_span=None, N=100, distribution='norm'):
    """ Solve for the surface impedance and magnetic fields (normalized) of an interface of rough conductor to a material. 

    Args:
        f (float/array): frequency in Hz
        sigma1 (float/array): conductivity of first medium, usually dielectric (for dielectric sigma = tand/omega/ep0). Can be frequency dependent.
        sigma2 (float/array): conductivity of second medium, usually conductor, e.g., copper. Can be frequency dependent.
        Rrms (float): RMS roughness (standard deviation).
        integration_span (list): range of integration. Recommendation [-5*Rrms, 5*Rrms] for normal distribution.
        N (int): number of samples to evaluate between the integration span. Defaults to 100.
        distribution (str): probability distribution of the roughness. Defaults to 'norm'. Other options: 'rayleigh', 'uniform'. See function conductivity()

    Returns:
        [Zs, B]: surface impedance (1D array matches frequency), B magnetic field (2D array: freq_length x N)
    """
    mu0 = 4*np.pi*1e-7     # permeability
    
    # set lower bound --> basically smooth (ODE solver gets confused if Rrms is near zero)
    use_smooth_model = False
    if Rrms < 0.0001e-6:
        use_smooth_model = True
        print('Rrms too small for ODE solver... using smooth model.')
    
    f = np.atleast_1d(f)
    sigma1 = np.atleast_1d(sigma1)*np.ones_like(f)  # force to have same length as frequency 
    sigma2 = np.atleast_1d(sigma2)*np.ones_like(f)
    
    integration_span = [-5*Rrms, 5*Rrms] if integration_span is None else integration_span
    integration_span = integration_span if integration_span[0] > integration_span[1] else integration_span[::-1]  # flip to match initial values
    integration_eval = np.linspace(integration_span[0], integration_span[1], N)
    
    BB  = []
    ZZs = []
    for ff, sig1, sig2 in zip(f, sigma1, sigma2):
        if not use_smooth_model:
            # solve ODE
            initial_values = [np.exp(-gamma(ff,sig2)*integration_span[0]), -gamma(ff,sig2)*np.exp(-gamma(ff,sig2)*integration_span[0])]
            
            sol = scipy.integrate.solve_ivp(fun=diff_eq, t_span=integration_span, 
                            y0=initial_values, method='DOP853',#'RK45', 
                            t_eval=integration_eval,
                            args=[ff, sig1, sig2, Rrms, distribution],
                            rtol=1e-6, atol=1e-10)  # change the tolarance for faster integration
            
            B = sol.y[0]/max(abs(sol.y[0]))  # normalized
            B_diff = sol.y[1]/max(abs(sol.y[0]))
            omega = 2*np.pi*ff
            Zs = -1j*omega*mu0*np.trapz(B, x=integration_eval)/np.trapz(B_diff, x=integration_eval)
            #Zs = -1j*omega*mu0*scipy.integrate.simpson(B, x=integration_eval)/scipy.integrate.simpson(B_diff, x=integration_eval)
        else:
            # use smooth model
            omega = 2*np.pi*ff
            Zs = (1+1j)*np.sqrt(omega*mu0/2/sig2)
            B = np.exp(-gamma(ff,sig2)*np.heaviside(integration_eval, 1)*integration_eval)
            B = B/max(abs(B))
        BB.append(B[::-1])
        ZZs.append(Zs)
        print(f'Frequency solved: {ff*1e-9:.5f} GHz')
    return np.array(ZZs), np.array(BB), integration_eval[::-1]


if __name__ == '__main__':    
    mu0 = 4*np.pi*1e-7     # permeability
    ep0 = 8.854e-12        # permittivity  
    f = np.logspace(-1, 2, 100)*1e9    
    
    # Rough conductor (normal distribution)  
    Rrms = 1e-6  # roughness RMS value (standard deviation)
    Zs_rough, B_rough, x = solve_Zs_B(f, N=256, distribution='norm', sigma1=0.001, sigma2=58e6,
                                      Rrms=Rrms, integration_span=[-5*Rrms, 10*Rrms])
    
    plt.figure()
    for ff in [1e9, 10e9, 100e9]:
        inx = np.where(np.round(f - ff, 10) == 0)[0][0]
        plt.plot(x*1e6, abs(B_rough[inx]), lw=2, label=f'Rough conductor, Rrms=1um, {ff*1e-9:.2f} GHz')
    
    # Smooth conductor  
    Rrms = 1e-6  # roughness RMS value (standard deviation)
    Zs_smooth, B_smooth, x = solve_Zs_B(f, N=256, distribution='norm', sigma1=0.001, sigma2=58e6,
                                        Rrms=0, integration_span=[-5*Rrms, 10*Rrms])
    
    for ff in [1e9, 10e9, 100e9]:
        inx = np.where(np.round(f - ff, 10) == 0)[0][0]
        plt.plot(x*1e6, abs(B_smooth[inx]), '--', lw=2, label=f'Smooth conductor, {ff*1e-9:.2f} GHz')
    plt.legend()
    plt.xlabel('Distance (um)')
    plt.ylabel('Normalized magnetic field')
    plt.ylim([-.1, 1.1])
    plt.xlim([-5, 10])
    
    plt.figure()
    plt.plot(f*1e-9, Zs_rough.real, lw=2, label='Real-part. Rough conductor, Rrms=1um')
    plt.plot(f*1e-9, Zs_rough.imag, lw=2, label='Imag-part. Rough conductor, Rrms=1um')
    plt.plot(f*1e-9, Zs_smooth.real, '--', lw=2, label='Real-part=Imag-part. Smooth conductor')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Surface Impedance (Ohm)')
    plt.ylim([0, 3])
    plt.xlim([0, 100])
    
    sigma_copper = 58e6
    sigma_eff = sigma_copper*(Zs_smooth.real/Zs_rough.real)**2
    mur_eff   = (Zs_rough.imag/Zs_smooth.real)**2
    
    plt.figure()
    plt.plot(f*1e-9, sigma_eff/1e6, lw=2, label='Effective conductivity, Rrms = 1um')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Conductivity (Ms/m)')
    plt.xlim([0, 100])
    plt.ylim([0, 60])
    plt.legend()
    
    plt.figure()
    plt.plot(f*1e-9, mur_eff, lw=2, label='Relative effective permeability, Rrms = 1um')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative permeability (unitless)')
    plt.xlim([0, 100])
    plt.ylim([0, 1000])
    plt.legend()
    
    
    plt.show()
    
    # EOF