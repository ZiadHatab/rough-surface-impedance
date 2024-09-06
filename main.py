"""
@Author: Ziad (https://github.com/ZiadHatab)

A script to compute the surface impedance of a rough interface between a conductor and another material (usually a dielectric).

Two approaches are implemented here:
    1. The original Gradient model based on the paper [1].
    2. The transmission line taper approach, based on the paper [2] (I generalized it to any material property).

[1] G. Gold and K. Helmreich, "A Physical Surface Roughness Model and Its Applications," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 65, no. 10, pp. 3720-3732, 
Oct. 2017, doi: 10.1109/TMTT.2017.2695192.

[2] B. Tegowski, T. Jaschke, A. Sieganschin and A. F. Jacob, 
"A Transmission Line Approach for Rough Conductor Surface Impedance Analysis," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 71, 
no. 2, pp. 471-479, Feb. 2023, doi: 10.1109/TMTT.2022.3206440

NOTE:
 - If you just want to compute the surface impedance, I recommend method [2]. It is more general and you can do more with it (at least the way I modified it).
 - The ODE solver for the gradient model [1] can be unstable, especially for low roughness values. Also it's slow!
 - Surface impedance calculation with gradient model [1] for small roughness values is unreliable... use method [2] instead.
 - I generalized the method of [2] to allow tapering of everything, i.e., permeability, permittivity, and conductivity.
 - In [1], the bulk conductivity can be frequency-dependent. In [2], all parameters can be frequency-dependent.
 - I only included some probability distribution functions. You can add more if you want something specific—see the corresponding function below.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

def get_CDF_PDF(x, Rrms, distribution='norm'):
    """
    Returns the CDF and PDF of the selected probability distribution.

    Args:
        x (float or array): Distance.
        Rrms (float): RMS roughness (standard deviation).
        distribution (str): Probability distribution of the roughness ('norm', 'rayleigh', 'uniform', etc.).

    Returns:
        tuple: CDF and PDF of the specified distribution.
    
    Raises:
        ValueError: If an unknown distribution is specified.
    """
    Rrms = 1e-15 if np.isclose(Rrms, 0, atol=1e-15) else Rrms  # Prevent division by zero
    
    if distribution == 'norm':
        # https://en.wikipedia.org/wiki/Normal_distribution
        scale = Rrms
        loc = 0
        CDF = scipy.stats.norm.cdf(x, loc=loc, scale=scale)
        PDF = scipy.stats.norm.pdf(x, loc=loc, scale=scale)
    elif distribution == 'rayleigh':
        # https://en.wikipedia.org/wiki/Rayleigh_distribution
        scale = np.sqrt(2/(4-np.pi))*Rrms
        loc = scale*np.sqrt(np.pi/2)
        CDF = scipy.stats.rayleigh.cdf(x, loc=loc, scale=scale)
        PDF = scipy.stats.rayleigh.pdf(x, loc=loc, scale=scale)
    elif distribution == 'uniform':
        # https://en.wikipedia.org/wiki/Continuous_uniform_distribution
        a = 0 - 3*Rrms
        b = 0 + 3*Rrms
        scale = 1/np.sqrt(12)*(b - a)
        loc = 0.5*(a + b)
        CDF = scipy.stats.uniform.cdf(x, loc=loc, scale=scale)
        PDF = scipy.stats.uniform.pdf(x, loc=loc, scale=scale)
    # Add more distributions here if needed
    else: 
        raise ValueError(f"Unknown distribution: {distribution}")

    return CDF, PDF

## This is the only function needed for the transmission line method [2] ( in addition to get_CDF_PDF(.) )
def Zs_TL(f, sigma1=0, sigma2=58e6, Rrms=1e-6, recursion_span=None, N=2048, distribution='norm',
          mur1=1-0j, mur2=1-0j, er1=None, er2=None):
    """
    Computes the surface impedance using the transmission line taper approach [2].

    Args:
        f (float or array): Frequency in Hz.
        sigma1 (float or array): Conductivity of the first medium (can be frequency dependent).
        sigma2 (float or array): Conductivity of the second medium (can be frequency dependent).
        Rrms (float): RMS roughness (standard deviation).
        recursion_span (list of two floats): Range for the recursion (default is [-5*Rrms, 10*Rrms]).
        N (int): Number of points for recursion evaluation (default is 2048).
        distribution (str): Probability distribution of roughness ('norm', 'rayleigh', 'uniform', etc.).
        mur1 (float or complex): Relative permeability of the first medium.
        mur2 (float or complex): Relative permeability of the second medium.
        er1 (float or array): Relative permittivity of the first medium (if given, overrides sigma1).
        er2 (float or array): Relative permittivity of the second medium (if given, overrides sigma2).

    Returns:
        numpy.ndarray: Surface impedance as a function of frequency.
    """
    # Constants
    mu0 = 4*np.pi*1e-7        # Permeability
    ep0 = 8.854187818814e-12  # Permittivity
    
    recursion_span = [-5*Rrms, 10*Rrms] if recursion_span is None else recursion_span
    
    # Ensure all inputs are arrays of the same length as frequency array
    f = np.atleast_1d(f)
    omega  = 2*np.pi*f
    sigma1 = np.atleast_1d(sigma1)*np.ones_like(f)
    sigma2 = np.atleast_1d(sigma2)*np.ones_like(f)
    mur1   = np.atleast_1d(mur1).astype(complex)*np.ones_like(f)
    mur2   = np.atleast_1d(mur2).astype(complex)*np.ones_like(f)
    
    # Calculate permittivity and select between sigma or er definition for the material
    if er1 is not None:
        er1 = np.atleast_1d(er1).astype(complex)*np.ones_like(f)
        ep1 = er1*ep0
    else:
        ep1 = ep0 - 1j*sigma1/omega

    if er2 is not None:
        er2 = np.atleast_1d(er2).astype(complex)*np.ones_like(f)
        ep2 = er2*ep0
    else:
        ep2 = ep0 - 1j*sigma2/omega
    
    mu1 = mur1*mu0
    mu2 = mur2*mu0
    
    recursion_eval = np.linspace(recursion_span[0], recursion_span[1], N)
    CDF, _ = get_CDF_PDF(recursion_eval, Rrms, distribution=distribution)
    
    Zs = []
    print('TL method running...')
    for idx, w in enumerate(omega):
        mmu = (mu2[idx] - mu1[idx])*CDF + mu1[idx]
        eep = (ep2[idx] - ep1[idx])*CDF + ep1[idx]
        gamma = np.sqrt(-w**2*mmu*eep)
        gamma = gamma*np.sign(gamma.real)  # Ensure positive square root
        Z = np.sqrt(mmu/eep)
        Z = Z*np.sign(Z.real)              # Ensure positive square root
        
        gamma = gamma[::-1]  # Reverse recursion order
        Z = Z[::-1]
        delta_L = np.diff(recursion_eval)[::-1]
        delta_L = np.hstack([[delta_L[0]], delta_L])  # Prepend to match length
        Zsi = Z[0]
        for g, z, dl in zip(gamma, Z, delta_L):            
            Zsi = z * (Zsi + z*np.tanh(g*dl)) / (z + Zsi*np.tanh(g*dl))
        Zs.append(Zsi)
        print(f'Frequency solved: {f[idx] * 1e-9:.5f} GHz')

    return np.array(Zs)

## below are all functions needed for the Gradient model [1] ( in addition to get_CDF_PDF(.) )
def conductivity(x, sigma1=0, sigma2=58e6, Rrms=1e-6, distribution='norm'):
    """
    Computes the conductivity profile as a function of distance from a given CDF and PDF.

    Args:
        x (float or array): Distance.
        sigma1 (float): Conductivity of the first medium (default is 0 S/m).
        sigma2 (float): Conductivity of the second medium (default is 58e6 S/m).
        Rrms (float): RMS roughness (standard deviation).
        distribution (str): Probability distribution of the roughness ('norm', 'rayleigh', 'uniform', etc.).

    Returns:
        tuple: Conductivity and its differential as functions of distance.
    """
    CDF, PDF = get_CDF_PDF(x, Rrms, distribution)
    sigma = (sigma2 - sigma1)*CDF + sigma1
    sigma_diff = (sigma2 - sigma1)*PDF
    return sigma, sigma_diff

def diff_eq(x, B, *args):
    """
    Differential equation describing B-fields in a conductor based on the gradient model [1]. 
    This function is solved by scipy.integrate.solve_ivp().

    Args:
        x (float): Current position in the integration.
        B (array): Array of B-field values.
        *args: Additional arguments passed to the function (frequency, conductivities, RMS roughness, distribution).

    Returns:
        list: Differential values for B-field.
    """
    f      = args[0]   # Frequency in Hz
    sigma1 = args[1]   # Conductivity of the first medium (S/m)
    sigma2 = args[2]   # Conductivity of the second medium (S/m)
    Rrms   = args[3]   # RMS roughness (standard deviation)
    distribution = args[4]  # Probability distribution of the roughness
    
    sigma, sigma_diff = conductivity(x, sigma1, sigma2, Rrms, distribution)
    
    mu0 = 4*np.pi*1e-7
    omega = 2*np.pi*f
    k = 1j*omega*mu0

    B_diff = B[1]
    B_diff_diff = sigma_diff/(sigma + np.finfo(float).eps)*B[1] + sigma*k*B[0]  # eps to avoid dividing by zero
    
    return [B_diff, B_diff_diff]

def Zs_B_gradient_model(f, sigma1=0, sigma2=58e6, Rrms=1e-6, integration_span=None, N=128, distribution='norm'):
    """
    Solves for the surface impedance and magnetic fields (normalized) of an interface between 
    a rough conductor and another material using the gradient model [1].

    Args:
        f (float or array): Frequency in Hz.
        sigma1 (float or array): Conductivity of the first medium (default is 0 S/m).
        sigma2 (float or array): Conductivity of the second medium (default is 58e6 S/m).
        Rrms (float): RMS roughness (standard deviation).
        integration_span (list of two floats): Range of integration (default is [-5*Rrms, 10*Rrms]).
        N (int): Number of points for integration (default is 128).
        distribution (str): Probability distribution of roughness ('norm', 'rayleigh', 'uniform', etc.).

    Returns:
        tuple: Surface impedance (1D array matching frequency) and magnetic field (2D array: freq_length x N).
    """
    mu0 = 4*np.pi*1e-7         # Permeability
    ep0 = 8.854187818814e-12   # Permittivity

    def gamma(f, sigma):
        # calculates the propagation constant in a conductor with conductivity sigma.
        omega = 2*np.pi*f
        ep = ep0 - 1j*sigma/omega
        val = omega*np.sqrt(-mu0*ep)
        return val*np.sign(val.real)  # Ensure positive square root
    
    use_smooth_model = False
    if Rrms < 0.0001e-6:
        use_smooth_model = True
        print('Rrms too small for ODE solver... using smooth model.')
    
    f = np.atleast_1d(f)
    sigma1 = np.atleast_1d(sigma1)*np.ones_like(f)
    sigma2 = np.atleast_1d(sigma2)*np.ones_like(f)
    
    integration_span = [-5*Rrms, 10*Rrms] if integration_span is None else integration_span
    integration_span = integration_span if integration_span[0] > integration_span[1] else integration_span[::-1]  # flip to match initial values
    
    # Gaussian Quadrature for accurate integration
    integration_eval, integration_weights = np.polynomial.legendre.leggauss(N)
    rescale_nodes  = lambda x, a, b: 0.5*(b - a)*x + 0.5*(b + a)
    rescale_weight = lambda w, a, b: 0.5*(b - a)*w
    integration_eval    = rescale_nodes(integration_eval, integration_span[0], integration_span[1])
    integration_weights = rescale_weight(integration_weights, integration_span[0], integration_span[1])
    
    BB = []
    ZZs = []
    print('Gradient method running...')
    for ff, sig1, sig2 in zip(f, sigma1, sigma2):
        if not use_smooth_model:
            # Solve ODE
            initial_values = [np.exp(-gamma(ff, sig2)*integration_span[0]), 
                              -gamma(ff, sig2)*np.exp(-gamma(ff, sig2)*integration_span[0])]
            
            sol = scipy.integrate.solve_ivp(fun=diff_eq, t_span=integration_span, 
                            y0=initial_values, method='DOP853', 
                            t_eval=integration_eval,
                            args=[ff, sig1, sig2, Rrms, distribution],
                            rtol=1e-8, atol=1e-12)  # change the tolerance for faster integration
            
            B = sol.y[0]/max(abs(sol.y[0]))  # Normalized
            B_diff = sol.y[1]/max(abs(sol.y[0]))
            J = B_diff/mu0
            omega = 2*np.pi*ff
            Zs = -1j*omega*np.sum(B*integration_weights)/np.sum(J*integration_weights)
        else:
            # Use smooth model
            omega = 2*np.pi*ff
            Zs = (1 + 1j)*np.sqrt(omega*mu0/(2*sig2))
            B = np.exp(-gamma(ff, sig2)*np.heaviside(integration_eval, 1)*integration_eval)
            B = B/max(abs(B))
        BB.append(B[::-1])
        ZZs.append(Zs)
        print(f'Frequency solved: {ff*1e-9:.5f} GHz')

    return np.array(ZZs), np.array(BB), integration_eval[::-1]


if __name__ == '__main__':
    mu0 = 4*np.pi*1e-7       # Permeability
    ep0 = 8.854187818814e-12 # Permittivity
    
    # frequency grid
    f   = np.logspace(-1, 2, 100)*1e9
    
    # Rough conductor (normal distribution)  
    Rrms = 1e-6  # roughness RMS value (standard deviation)
    Zs_rough_grad, B_rough, x = Zs_B_gradient_model(f, sigma1=0, sigma2=58e6, N=128, distribution='norm',
                                                    Rrms=Rrms, integration_span=[-5*Rrms, 10*Rrms])
    # Smooth conductor  
    Zs_smooth, B_smooth, x = Zs_B_gradient_model(f, sigma1=0, sigma2=58e6, N=128, distribution='norm', 
                                                Rrms=0, integration_span=[-5*Rrms, 10*Rrms])
    # TL model
    Zs_rough_TL = Zs_TL(f, sigma1=0, sigma2=58e6, Rrms=Rrms, N=2048, 
                        distribution='norm', recursion_span=[-5*Rrms, 10*Rrms])
    
    # Plot normalized magnetic field magnitude
    plt.figure()
    for ff in [1e9, 10e9, 100e9]:
        idx = np.isclose(f, ff).nonzero()[0][0]
        plt.plot(x*1e6, abs(B_rough[idx]), lw=2, label=f'Rough conductor, {Rrms*1e6:.2f}um, {ff*1e-9:.2f} GHz')
    
    for ff in [1e9, 10e9, 100e9]:
        idx = np.isclose(f, ff).nonzero()[0][0]
        plt.plot(x*1e6, abs(B_smooth[idx]), '--', lw=2, label=f'Smooth conductor, {ff*1e-9:.2f} GHz')
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

    ## Calculate and plot effective conductivity and permeability
    sigma_copper = 58e6
    # Gradient Model
    sigma_eff_grad = sigma_copper*(Zs_smooth.real/Zs_rough_grad.real)**2
    mur_eff_grad   = (Zs_rough_grad.imag/Zs_smooth.real)**2
    # TL Model
    sigma_eff_TL = sigma_copper*(Zs_smooth.real/Zs_rough_TL.real)**2
    mur_eff_TL   = (Zs_rough_TL.imag/Zs_smooth.real)**2

    # Plot effective conductivity
    plt.figure()
    plt.plot(f*1e-9, sigma_eff_grad/1e6, lw=2, label='Rrms=1um, Gradient model')
    plt.plot(f*1e-9, sigma_eff_TL/1e6, '--', lw=2, label='Rrms=1um, TL model')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Effective Conductivity (Ms/m)')
    plt.xlim([0, 100])
    plt.ylim([0, 60])
    plt.legend()

    # Plot relative effective permeability
    plt.figure()
    plt.plot(f*1e-9, mur_eff_grad, lw=2, label='Rrms=1um, Gradient model')
    plt.plot(f*1e-9, mur_eff_TL, '--', lw=2, label='Rrms=1um, TL model')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative Effective Permeability (Unitless)')
    plt.xlim([0, 100])
    plt.ylim([0, 1200])
    plt.legend()
    
    ## Example of other distributions
    Rrms = 1e-6 
    Zs_norm = Zs_TL(f, sigma1=0, sigma2=58e6, Rrms=Rrms, N=2048, 
                    distribution='norm', recursion_span=[-5*Rrms, 10*Rrms])
    Zs_rayleigh = Zs_TL(f, sigma1=0, sigma2=58e6, Rrms=Rrms, N=2048, 
                        distribution='rayleigh', recursion_span=[-5*Rrms, 10*Rrms])
    Zs_uniform  = Zs_TL(f, sigma1=0, sigma2=58e6, Rrms=Rrms, N=2048, 
                        distribution='uniform', recursion_span=[-5*Rrms, 10*Rrms])
    sigma_copper = 58e6
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
    plt.plot(f*1e-9, sigma_eff_norm/1e6, lw=2, label=f'Normal, Rrms={Rrms*1e6:.2f}um ')
    plt.plot(f*1e-9, sigma_eff_rayleigh/1e6, '--', lw=2, label=f'Rayleigh, Rrms={Rrms*1e6:.2f}um')
    plt.plot(f*1e-9, sigma_eff_uniform/1e6, '-.', lw=2, label=f'Uniform, Rrms={Rrms*1e6:.2f}um')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Effective Conductivity (Ms/m)')
    plt.xlim([0, 100])
    plt.ylim([0, 60])
    plt.legend()

    # Plot relative effective permeability
    plt.figure()
    plt.plot(f*1e-9, mur_eff_norm, lw=2, label=f'Normal, Rrms={Rrms*1e6:.2f}um ')
    plt.plot(f*1e-9, mur_eff_rayleigh, '--', lw=2, label=f'Rayleigh, Rrms={Rrms*1e6:.2f}um')
    plt.plot(f*1e-9, mur_eff_uniform, '-.', lw=2, label=f'Uniform, Rrms={Rrms*1e6:.2f}um')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative Effective Permeability (Unitless)')
    plt.xlim([0, 100])
    plt.ylim([0, 1200])
    plt.legend()

    plt.show()
    # EOF