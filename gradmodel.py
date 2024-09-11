"""
@Author: Ziad (https://github.com/ZiadHatab)

This is the original Gradient model based on the paper [1] to compute the surface impedance 
of a rough interface between a conductor and another material (usually a dielectric). 

I wrote this script just as a reference. It is not stable, and I would not recommend using it. Use the transmission line method instead.
NOTE:
 - The ODE solver can be unstable, especially for low roughness values. It is also very slow.
 - Imaginary part of the surface impedance for small roughness values is unreliable... use transmission method instead.

References:
[1] G. Gold and K. Helmreich, "A Physical Surface Roughness Model and Its Applications," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 65, no. 10, pp. 3720-3732, 
Oct. 2017, doi: https://doi.org/10.1109/TMTT.2017.2695192.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

def get_CDF_PDF(x, Rrms, boundary_loc, distribution='norm'):
    """
    Returns the CDF and PDF of the selected probability distribution.

    Args:
        x (float or array): Distance.
        Rrms (float): RMS roughness (standard deviation).
        boundary_loc (float): Boundary location (mean value).
        distribution (str): Probability distribution of the roughness ('norm', 'rayleigh', 'uniform', etc.).

    Returns:
        tuple: CDF and PDF of the specified distribution.
    
    Raises:
        ValueError: If an unknown distribution is specified.
    """
    Rrms = 1e-14 if np.isclose(Rrms, 0, atol=1e-14) else Rrms  # prevent division by zero in the CDFs and PDFs
    
    if distribution == 'norm':
        # https://en.wikipedia.org/wiki/Normal_distribution
        scale = Rrms
        loc = boundary_loc
        CDF = scipy.stats.norm.cdf(x, loc=loc, scale=scale)
        PDF = scipy.stats.norm.pdf(x, loc=loc, scale=scale)
    elif distribution == 'rayleigh':
        # https://en.wikipedia.org/wiki/Rayleigh_distribution
        scale = np.sqrt(2/(4-np.pi))*Rrms
        loc = scale*np.sqrt(np.pi/2) + boundary_loc
        CDF = scipy.stats.rayleigh.cdf(x, loc=loc, scale=scale)
        PDF = scipy.stats.rayleigh.pdf(x, loc=loc, scale=scale)
    elif distribution == 'uniform':
        # https://en.wikipedia.org/wiki/Continuous_uniform_distribution
        a = boundary_loc - 3*Rrms
        b = boundary_loc + 3*Rrms
        scale = 1/np.sqrt(12)*(b - a)
        loc = 0.5*(a + b)
        CDF = scipy.stats.uniform.cdf(x, loc=loc, scale=scale)
        PDF = scipy.stats.uniform.pdf(x, loc=loc, scale=scale)
    # Add more distributions here
    else: 
        raise ValueError(f"Unknown distribution: {distribution}")

    return CDF, PDF

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
    CDF, PDF = get_CDF_PDF(x, Rrms, 0, distribution)
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

def surface_impedance(f, sigma1=0, sigma2=58e6, Rrms=1e-6, integration_span=None, N=128, distribution='norm'):
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
    pass