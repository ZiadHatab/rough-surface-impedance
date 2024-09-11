"""
@Author: Ziad (https://github.com/ZiadHatab)

This script calculates the surface impedance of a rough interface between a conductor and another material (usually a dielectric).
It also provides the ability to calculate the surface impedance of multiple stacked thin conductors using the transmission line taper approach.
The script is based on the paper [1] and has been generalized to handle any material property and multiple stacked materials [2].

Features:
 - You can taper everything, i.e., permeability, permittivity, and conductivity.
 - All parameters can be frequency-dependent.
 - I included some probability distribution functions. You can add more.
 - Based on [2], you can use multiple layers of materials with different permeabilities, permittivities, and conductivities.
 - For multiple layers, each boundary can be of different distribution and roughness level.

References:
[1] B. Tegowski, T. Jaschke, A. Sieganschin and A. F. Jacob, 
"A Transmission Line Approach for Rough Conductor Surface Impedance Analysis," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 71, 
no. 2, pp. 471-479, Feb. 2023, doi: https://doi.org/10.1109/TMTT.2022.3206440

[2] G. Gold and K. Helmreich, "Modeling of transmission lines with multiple coated conductors," 
2016 46th European Microwave Conference (EuMC), London, UK, 2016, pp. 635-638, 
doi: https://doi.org/10.1109/EuMC.2016.7824423.
"""

import numpy as np
import scipy

def get_CDF(x, Rrms, boundary_loc, distribution='norm'):
    """
    Returns the CDF and PDF of the selected probability distribution.

    Args:
        x (float or array): Distance.
        Rrms (float): RMS roughness (standard deviation).
        boundary_loc (float): Boundary location (mean value).
        distribution (str): Probability distribution of the roughness ('norm', 'rayleigh', 'uniform', etc.).

    Returns:
        tuple: CDF of the specified distribution.
    
    Raises:
        ValueError: If an unknown distribution is specified.
    """
    Rrms = 1e-14 if np.isclose(Rrms, 0, atol=1e-14) else Rrms  # prevent division by zero in the CDFs and PDFs
    
    if distribution == 'norm':
        # https://en.wikipedia.org/wiki/Normal_distribution
        scale = Rrms
        loc = boundary_loc
        CDF = scipy.stats.norm.cdf(x, loc=loc, scale=scale)
        #PDF = scipy.stats.norm.pdf(x, loc=loc, scale=scale)
    elif distribution == 'rayleigh':
        # https://en.wikipedia.org/wiki/Rayleigh_distribution
        scale = np.sqrt(2/(4-np.pi))*Rrms
        loc = scale*np.sqrt(np.pi/2) + boundary_loc
        CDF = scipy.stats.rayleigh.cdf(x, loc=loc, scale=scale)
        #PDF = scipy.stats.rayleigh.pdf(x, loc=loc, scale=scale)
    elif distribution == 'uniform':
        # https://en.wikipedia.org/wiki/Continuous_uniform_distribution
        a = boundary_loc - 3*Rrms
        b = boundary_loc + 3*Rrms
        scale = 1/np.sqrt(12)*(b - a)
        loc = 0.5*(a + b)
        CDF = scipy.stats.uniform.cdf(x, loc=loc, scale=scale)
        #PDF = scipy.stats.uniform.pdf(x, loc=loc, scale=scale)
    # add more distributions here
    else: 
        raise ValueError(f"Unknown distribution: {distribution}")

    return CDF

def surface_impedance(f, material_properties=None, Rrms=1e-9, boundary_loc=0, distribution='norm',
                      recursion_span=None, N=2048, return_material_profile=False):
    """
    Computes the surface impedance using the transmission line taper approach [1].
    I updated the procedure to support multiconductor [2], where each boundary has its own roughness.
    
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
        distribution (str or list of str): Probability distribution of roughness ('norm', 'rayleigh', 'uniform', etc.).

    Returns:
        numpy.ndarray: Surface impedance as a function of frequency.

        If return_material_profile is True, also returns a list of material profiles as function of distance for each frequency.
    """
    # Constants
    mu0 = 4*np.pi*1e-7        # Permeability
    ep0 = 8.854187818814e-12  # Permittivity
    
    M = len(material_properties)  # number of materials
    Rrms = np.atleast_1d(Rrms)*np.ones(M-1)
    boundary_loc = np.atleast_1d(boundary_loc)*np.ones(M-1)
    distribution = np.resize(np.atleast_1d(distribution), len(Rrms)).tolist()

    small_number = 1e-10  # this is to deal with cases when Rrms is zero and boundary_loc is zero, Otherwise the recursion bound would be zero.
    recursion_span = [-5*Rrms[0] + boundary_loc[0] - small_number, 10*Rrms[-1] + boundary_loc[-1] + small_number] if recursion_span is None else recursion_span
    
    # force frequency to be an array
    f = np.atleast_1d(f)
    omega  = 2*np.pi*f
    
    # default value if nothing provided by the user
    material_properties = [{'sigma':0, 'mur':1-0j, 'er': None},
                           {'sigma':58e6, 'mur':1-0j, 'er': None}] if material_properties is None else material_properties
    
    # check input lengths
    error_message = "Input lengths do not match:\n"
    if len(Rrms) != M-1:
        error_message += f"Rrms: {len(Rrms)} (expected {M - 1})\n"
        raise ValueError(error_message)
    if len(boundary_loc) != M-1:
        error_message += f"boundary_loc: {len(boundary_loc)} (expected {M - 1})\n"
        raise ValueError(error_message)
    if len(distribution) != M-1:
        error_message += f"distribution: {len(distribution)} (expected {M - 1})"
        raise ValueError(error_message)

    # fill in the missing properties not provided by the user with default values
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
    
    # this define the poins at which the recursion will be performed
    recursion_eval = np.linspace(recursion_span[0], recursion_span[1], N)
    
    # compute the CDF of each boundary
    CDF = []
    for r, offset, dist in zip(Rrms, boundary_loc, distribution):
        cdf = get_CDF(recursion_eval, r, offset, dist)
        CDF.append(cdf)
    
    # this is the length difference for each small segment
    delta_L = np.diff(recursion_eval)[::-1]
    delta_L = np.hstack([[delta_L[0]], delta_L])  # prepend to match length
    
    # here is the main script for computing the surface impedance
    Zs = []
    mur_list = []
    er_list  = []
    print('TL method running...')
    for idx, w in enumerate(omega):
        # compute the material property profile as a multimodal from the individual CDFs
        mur = np.array([(material_properties[inx+1]['mur'][idx] - material_properties[inx]['mur'][idx])*cdf for inx,cdf in enumerate(CDF)]).sum(axis=0) + material_properties[0]['mur'][idx]
        er  = np.array([(material_properties[inx+1]['er'][idx] - material_properties[inx]['er'][idx])*cdf for inx,cdf in enumerate(CDF)]).sum(axis=0) + material_properties[0]['er'][idx]
        
        # scale the units
        mu = mu0*mur
        ep = ep0*er
        
        # propagation constant and intrinsic impedance at each step
        gamma = np.sqrt(-w**2*mu*ep)
        gamma = gamma*np.sign(gamma.real)  # ensure positive square root
        Z = np.sqrt(mu/ep)
        Z = Z*np.sign(Z.real)              # ensure positive square root
        
        gamma = gamma[::-1]  # reverse recursion order  (from last material to first)
        Z = Z[::-1]
        Zsi = Z[0]
        for g, z, dl in zip(gamma, Z, delta_L):
            tanh = np.tanh(g*dl)
            Zsi = z*(Zsi + z*tanh)/(z + Zsi*tanh)
        Zs.append(Zsi)
        mur_list.append(mur)
        er_list.append(er)
        print(f'Frequency solved: {f[idx] * 1e-9:.5f} GHz')
    
    if return_material_profile:
        return np.array(Zs), np.array(mur_list), np.array(er_list), recursion_eval
    else:
        return np.array(Zs)
    

if __name__ == '__main__':
    pass