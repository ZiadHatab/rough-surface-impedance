"""
@Author: Ziad (https://github.com/ZiadHatab)

A script to compute the surface impedance of a rough interface between a conductor and another material (usually a dielectric).
Also the ability to calculate the surface impedance of multiple stacked thin conductors.

Two approaches are implemented here:
    1. The original Gradient model based on the paper [1] (this is just as reference. Not stable. Don't use!).
    2. The transmission line taper approach, based on the paper [2] (I generalized it to any material property and multiple stacked materials).

NOTE:
 - If you just want to compute the surface impedance, I recommend method [2]. It is more general and you can do more with it.
 - The ODE solver for the gradient model [1] can be unstable, especially for low roughness values. It is also very slow.
 - Imaginary part of the surface impedance with gradient model [1] for small roughness values is unreliable... use method [2] instead.
 - I generalized the method of [2] to allow tapering of everything, i.e., permeability, permittivity, and conductivity.
 - Also, all parameters can be frequency-dependent.
 - I only included some probability distribution functions. You can add more if you want something specific—see the corresponding function below.
 - For method [2], you can also use multiple layers of materials with different permeabilities, permittivities, and conductivities, which use is based on the idea of [3].

References:
[1] G. Gold and K. Helmreich, "A Physical Surface Roughness Model and Its Applications," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 65, no. 10, pp. 3720-3732, 
Oct. 2017, doi: https://doi.org/10.1109/TMTT.2017.2695192.

[2] B. Tegowski, T. Jaschke, A. Sieganschin and A. F. Jacob, 
"A Transmission Line Approach for Rough Conductor Surface Impedance Analysis," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 71, 
no. 2, pp. 471-479, Feb. 2023, doi: https://doi.org/10.1109/TMTT.2022.3206440

[3] G. Gold and K. Helmreich, "Modeling of transmission lines with multiple coated conductors," 
2016 46th European Microwave Conference (EuMC), London, UK, 2016, pp. 635-638, 
doi: https://doi.org/10.1109/EuMC.2016.7824423.

[4] B. Schafsteller, M. Schwaemmlein, M. Rosin, G. Ramos, Z. Hatab, M. E. Gadringer, E. Schlaffer, 
"Investigating the Impact of Final Finishes on the Insertion Loss in As Received and After Aging," 
IMAPSource Proceedings, vol. 2023, no. Symposium. IMAPS - International Microelectronics Assembly 
and Packaging Society, Feb. 29, 2024. doi: https://doi.org/10.4071/001c.94519.

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

## This is the only function needed for the transmission line method [2] ( in addition to get_CDF_PDF(.) )
def Zs_TL(f, material_properties=None, Rrms=1e-6, boundary_loc=0, recursion_span=None, N=2048, distribution='norm'):
    """
    Computes the surface impedance using the transmission line taper approach [2].
    I updated the procedure to support multiconductor [3], where each boundary has its own roughness.
    
    Args:
        f (float or array): Frequency in Hz.
        
        material_properties (list of dict): Material properties for each boundary. Each dict contain:
        {'sigma': Conductivity,
        'mur': Relative Permeability,
        'er': Relative Permittivity}
        Any property you don't specify will be set to the default value. sigma = 0; mur = 1; er = 1
        Conductivity definition of material proceed definition using relative permittivity. 
        i.e., if you set a value for sigma, any value set for er will be ignored.
        
        Rrms (float or list): RMS roughness (standard deviation) of each boundary.
        boundary_loc (float or list): Boundary location (mean value) of each boundary.
        recursion_span (list of two floats): Range for the recursion (default is [-5*Rrms[0]+boundary_loc[0], 10*Rrms[-1]+boundary_loc[-1]]).
        N (int): Number of points for recursion evaluation (default is 2048).
        distribution (str): Probability distribution of roughness ('norm', 'rayleigh', 'uniform', etc.).

    Returns:
        numpy.ndarray: Surface impedance as a function of frequency.
    """
    # Constants
    mu0 = 4*np.pi*1e-7        # Permeability
    ep0 = 8.854187818814e-12  # Permittivity
    
    Rrms = np.atleast_1d(Rrms)
    boundary_loc = np.atleast_1d(boundary_loc)
    small_number = 1e-14  # this is to deal with cases when Rrms is zero and boundary_loc is zero, Otherwise the recursion bound would be zero.
    recursion_span = [-5*Rrms[0] + boundary_loc[0] - small_number, 10*Rrms[-1] + boundary_loc[-1] + small_number] if recursion_span is None else recursion_span
    
    # Ensure all inputs are arrays of the same length as frequency array
    f = np.atleast_1d(f)
    omega  = 2*np.pi*f
    material_properties = [{'sigma':0, 'mur':1-0j, 'er': None},
                           {'sigma':58e6, 'mur':1-0j, 'er': None}] if material_properties is None else material_properties
    
    # fill in the missing properties not provided by the user
    required_keys = ['sigma', 'mur', 'er']
    default_values = {'sigma': 0, 'mur': 1-0j, 'er': None}
    for material in material_properties:
        for key in required_keys:
          if key not in material:
            material[key] = default_values.get(key, None)
    
    # make frequency dependent
    for material in material_properties:
        material['sigma'] = np.atleast_1d(material['sigma'])*np.ones_like(f)
        material['mur'] = np.atleast_1d(material['mur']).astype(complex)*np.ones_like(f)
        
        # select between sigma or er for defining the permittivity
        if material['er'] is not None:
            material['er'] = np.atleast_1d(material['er']).astype(complex)*np.ones_like(f)
        else:
            material['er'] = 1 - 1j*material['sigma']/omega/ep0
    
    recursion_eval = np.linspace(recursion_span[0], recursion_span[1], N)
    CDF = []
    for r, offset in zip(Rrms, boundary_loc):
        cdf, _ = get_CDF_PDF(recursion_eval, r, offset, distribution=distribution)
        CDF.append(cdf)
        
    Zs = []
    print('TL method running...')
    for idx, w in enumerate(omega):
        
        mmu = mu0*( np.array([(material_properties[inx+1]['mur'][idx] - material_properties[inx]['mur'][idx])*cdf for inx,cdf in enumerate(CDF)]).sum(axis=0) + material_properties[0]['mur'][idx] )
        eep = ep0*( np.array([(material_properties[inx+1]['er'][idx] - material_properties[inx]['er'][idx])*cdf for inx,cdf in enumerate(CDF)]).sum(axis=0) + material_properties[0]['er'][idx] )
        
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
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    gamma2dbcm  = lambda x: mag2db(np.exp(x.real*1e-2))  # losses dB/cm
    time2distance = lambda x,er: x*c0/np.sqrt(er.real)
    Zsmooth = lambda f, sigma, mur: (1 + 1j)*np.sqrt(2*np.pi*f*mu0*mur/(2*sigma))
    
    
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
    material_properties = [{'sigma': 0}, {'sigma': 58e6}]
    Zs_rough_TL = Zs_TL(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='norm')
    
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
    material_properties = [{'sigma': 0}, {'sigma': 58e6}]
    Zs_norm = Zs_TL(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='norm')
    Zs_rayleigh = Zs_TL(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='rayleigh')
    Zs_uniform  = Zs_TL(f, material_properties, Rrms=Rrms, boundary_loc=0, distribution='uniform')
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
    
    ## Example of multiple stacked thin conductors, e.g., surface finish on PCB
    # see the paper [4] for effects of surface finish on losses on transmission line
    # in the example below the roughness is very small, near flat. I'm just using 5nm for the sake of including roughness effects.
    
    # ENIG example: Air-Gold-Nickel-Copper
    # thicknesses (first and last materials are base and assumed they extend to infinity):
        # Gold: 0.05um
        # Nickel: 4um
        
    # ENIPIG example: Air-Gold-Palladium-Nickel-Copper
    # thicknesses: 
        # Gold: 0.1um
        # Palladium: 0.1um
        # Nickel: 4um
    
    # For the material electrical properties, just google them (e.g., wikipedia)
    # Air: sigma = 0; mur = 1
    # Gold: sigma = 41.1e6 S/m; mur = 1
    # Palladium: sigma = 10e6 S/m; mur = 1
    # Nickel: sigma = 14.3e6 S/m; mur = 350
    # Copper: sigma = 58e6 S/m; mur = 1
    
    f = np.logspace(-1, 2.5, 100)*1e9  # frequency grid... up to roughly 300GHz

    # materials with mur=1 are default values in the function and thus not defined explicitly.
    # ENIG with thickness as given in [4]
    material_properties = [{'sigma': 0},  # air
                           {'sigma': 41.1e6},   # gold
                           {'sigma': 14.3e6, 'mur': 350},  # nickel
                           {'sigma': 58e6}  # copper
                           ]
    roughness = [5e-9]*(len(material_properties)-1)  # 5nm RMS roughness at each boundary
    thickness = [0, 0.05e-6, 4e-6]  # last one not mentioned as it is assumed infinity
    boundary_loc = np.cumsum(thickness)
    Zs_enig = Zs_TL(f, material_properties, Rrms=roughness, boundary_loc=boundary_loc)
    
    # ENIG with a thicker gold layer 0.15um
    material_properties = [{'sigma': 0},   # air
                           {'sigma': 41.1e6},   # gold
                           {'sigma': 14.3e6, 'mur': 350}, # nickel
                           {'sigma': 58e6}  # copper
                           ]
    roughness = [5e-9]*(len(material_properties)-1)  # 5nm RMS roughness at each boundary
    thickness = [0, 0.15e-6, 4e-6]  # last one not mentioned as it is assumed infinity
    boundary_loc = np.cumsum(thickness)
    Zs_enig_thick_gold = Zs_TL(f, material_properties, Rrms=roughness, boundary_loc=boundary_loc)
    
    # ENIG with even thicker gold layer 0.5um
    material_properties = [{'sigma': 0},   # air
                           {'sigma': 41.1e6},   # gold
                           {'sigma': 14.3e6, 'mur': 350}, # nickel
                           {'sigma': 58e6}  # copper
                           ]
    roughness = [5e-9]*(len(material_properties)-1)  # 5nm RMS roughness at each boundary
    thickness = [0, 0.5e-6, 4e-6]  # last one not mentioned as it is assumed infinity
    boundary_loc = np.cumsum(thickness)
    Zs_enig_thick_gold2 = Zs_TL(f, material_properties, Rrms=roughness, boundary_loc=boundary_loc)
    
    # ENIPIG with thickness as given in [4]
    material_properties = [{'sigma': 0},   # air
                           {'sigma': 41.1e6},   # gold
                           {'sigma': 10e6},   # palladium
                           {'sigma': 14.3e6, 'mur': 350},   # nickel
                           {'sigma': 58e6}   # copper
                           ]
    roughness = [5e-9]*(len(material_properties)-1)  # 5nm RMS roughness at each boundary
    thickness = [0, 0.1e-6, 0.1e-6, 4e-6]  # last one not mentioned as it is assumed infinity
    boundary_loc = np.cumsum(thickness)
    Zs_enipig = Zs_TL(f, material_properties, Rrms=roughness, boundary_loc=boundary_loc)
    
    sigma_copper = 58e6
    Zs_smooth = Zsmooth(f, sigma_copper, 1)
    # ENIG
    sigma_eff_enig = sigma_copper*(Zs_smooth.real/Zs_enig.real)**2
    mur_eff_enig   = (Zs_enig.imag/Zs_smooth.real)**2
    
    # ENIG with thicker gold layer 0.15um
    sigma_eff_enig_thick_gold = sigma_copper*(Zs_smooth.real/Zs_enig_thick_gold.real)**2
    mur_eff_enig_thick_gold   = (Zs_enig_thick_gold.imag/Zs_smooth.real)**2
    
    # ENIG with even thicker gold layer 0.5um
    sigma_eff_enig_thick_gold2 = sigma_copper*(Zs_smooth.real/Zs_enig_thick_gold2.real)**2
    mur_eff_enig_thick_gold2   = (Zs_enig_thick_gold2.imag/Zs_smooth.real)**2
    
    # ENIPIG
    sigma_eff_enipig = sigma_copper*(Zs_smooth.real/Zs_enipig.real)**2
    mur_eff_enipig   = (Zs_enipig.imag/Zs_smooth.real)**2
    
    # Plot effective conductivity
    plt.figure()
    plt.plot(f*1e-9, sigma_eff_enig/1e6, lw=2, label='ENIG, Gold=0.05um, Nickel=4um')
    plt.plot(f*1e-9, sigma_eff_enig_thick_gold/1e6, '--', lw=2, label='ENIG, Gold=0.15um, Nickel=4um')
    plt.plot(f*1e-9, sigma_eff_enig_thick_gold2/1e6, '--', lw=2, label='ENIG, Gold=0.5um, Nickel=4um')
    plt.plot(f*1e-9, sigma_eff_enipig/1e6, '-.', lw=2, label='ENIPIG, Gold=0.1um, Palladium=0.1um, Nickel=4um')
    plt.hlines(41.1, min(f)*1e-9, max(f)*1e-9, lw=2, linestyles='--', color='black', label='Gold conductivity')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Effective Conductivity (Ms/m)')
    plt.xlim([0, 300])
    plt.ylim([0, 70])
    plt.legend(loc='upper right')

    # Plot relative effective permeability
    plt.figure()
    plt.semilogy(f*1e-9, mur_eff_enig, lw=2, label='ENIG, Gold=0.05um, Nickel=4um')
    plt.semilogy(f*1e-9, mur_eff_enig_thick_gold, '--', lw=2, label='ENIG, Gold=0.15um, Nickel=4um')
    plt.semilogy(f*1e-9, mur_eff_enig_thick_gold2, '--', lw=2, label='ENIG, Gold=0.5um, Nickel=4um')
    plt.semilogy(f*1e-9, mur_eff_enipig, '-.', lw=2, label='ENIPIG, Gold=0.1um, Palladium=0.1um, Nickel=4um')
    plt.hlines(1, min(f)*1e-9, max(f)*1e-9, lw=2, linestyles='--', color='black', label='Gold relative permeability')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative Effective Permeability (Unitless)')
    plt.xlim([0, 300])
    plt.ylim([1e-1, 1e3])
    plt.legend()


    plt.show()
    # EOF
