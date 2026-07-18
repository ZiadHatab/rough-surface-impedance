"""
@author: Ziad Hatab (zi.hatab@gmail.com)

Surface impedance of rough conductors via the transmission line taper method [1], 
generalized to multiple stacked materials (e.g., coating layers) following [2], 
where each boundary has its own roughness level and probability distribution.

The roughness CDF of each boundary defines a material profile mu(x) and ep(x).
The profile is split into transmission line segments, each with propagation
constant gamma = j*omega*sqrt(mu*ep) and intrinsic impedance eta = sqrt(mu/ep),
and the input impedance is transformed from the deepest material up to the
surface (Eq. (9) in [1]):

    Z_i = eta_i*(Z_{i+1} + eta_i*tanh(gamma_i*dx))/(eta_i + Z_{i+1}*tanh(gamma_i*dx))

Segments are uniform over the [-5*Rq, +5*Rq] window of each boundary [1, 2];
the bulk between boundaries needs one segment each, being exact for a uniform
line. The material variation inside a segment is handled by the 4th-order
Magnus method (Eqs. (252)-(254) in [3]), and N is doubled until Zs converges
(Eq. (10) in [1]), combining the last two levels by Richardson extrapolation
[4]. The field is optional (Appendix of [1]).


References:
[1] B. Tegowski, T. Jaschke, A. Sieganschin and A. F. Jacob,
"A Transmission Line Approach for Rough Conductor Surface Impedance Analysis,"
in IEEE Transactions on Microwave Theory and Techniques, vol. 71,
no. 2, pp. 471-479, Feb. 2023, doi: https://doi.org/10.1109/TMTT.2022.3206440

[2] G. Gold and K. Helmreich, "Modeling of transmission lines with multiple coated conductors,"
2016 46th European Microwave Conference (EuMC), London, UK, 2016, pp. 635-638,
doi: https://doi.org/10.1109/EuMC.2016.7824423.

[3] S. Blanes, F. Casas, J. A. Oteo, and J. Ros, "The Magnus expansion and some
of its applications," Physics Reports, vol. 470, no. 5-6, pp. 151-238, 2009,
doi: https://doi.org/10.1016/j.physrep.2008.11.001.

[4] L. F. Richardson and J. A. Gaunt, "The deferred approach to the limit,"
Philosophical Transactions of the Royal Society A, vol. 226, pp. 299-361, 1927,
doi: https://doi.org/10.1098/rsta.1927.0008.
"""

import warnings
import numpy as np  # python -m pip install numpy
import scipy.stats  # python -m pip install scipy

# physical constants
mu0 = 1.25663706127e-6  # vacuum permeability in H/m (CODATA 2022)
ep0 = 8.8541878188e-12  # vacuum permittivity in F/m (CODATA 2022)

_RRMS_FLOOR = 1e-14  # minimum rms roughness considered.

def get_CDF(x, Rrms, boundary_loc, distribution='norm'):
    """CDF of the roughness profile, with mean ``boundary_loc`` and standard deviation ``Rrms``.

    Parameters
    ----------
    x : float or array_like
        Distance values at which to evaluate the CDF.
    Rrms : float
        RMS roughness (standard deviation). ``Rrms = 0`` gives a step transition.
    boundary_loc : float
        Boundary location (mean value).
    distribution : optional
        Name of a scipy.stats continuous distribution without shape parameters
        (``'norm'`` (default), ``'rayleigh'``, ``'uniform'``, ...), a frozen one
        for those with shape parameters (e.g. ``scipy.stats.gamma(2)``), or a
        function ``distribution(x, Rrms, boundary_loc)`` returning the CDF.
        Named and frozen distributions are shifted and scaled to the given mean
        and standard deviation.

    Returns
    -------
    ndarray
        CDF evaluated at ``x``, increasing from 0 to 1.
    """
    x = np.asarray(x, dtype=float)
    Rrms = max(float(Rrms), _RRMS_FLOOR)  # zero roughness --> practically a step transition

    if isinstance(distribution, str):
        dist = getattr(scipy.stats, distribution, None)
        if not isinstance(dist, scipy.stats.rv_continuous):
            raise ValueError(f"Unknown distribution '{distribution}'. "
                             "Use the name of a scipy.stats continuous distribution,\n"
                             "e.g., 'norm', 'rayleigh', 'uniform'.")
        try:
            m, v = dist.stats(moments='mv')
        except TypeError as e:
            raise ValueError(f"Distribution '{distribution}' requires shape parameters. "
                             f"Pass a frozen distribution instead, e.g., scipy.stats.{distribution}(...).") from e
        scale = Rrms/np.sqrt(float(v))
        return dist.cdf(x, loc=boundary_loc - float(m)*scale, scale=scale)

    if hasattr(distribution, 'cdf'):  # frozen scipy.stats distribution
        m, v = distribution.stats(moments='mv')
        s = np.sqrt(float(v))
        return distribution.cdf((x - boundary_loc)*(s/Rrms) + float(m))

    if callable(distribution):  # user-supplied CDF function
        return np.asarray(distribution(x, Rrms, boundary_loc))

    raise ValueError(f"Invalid distribution: {distribution!r}")

def smooth_surface_impedance(f, sigma, mur=1):
    """Surface impedance of a smooth conductor, (1+j)*sqrt(pi*f*mu0*mur/sigma),
    with the frequency f in Hz and the conductivity sigma in S/m."""
    f     = np.asarray(f, dtype=float)
    mur   = np.asarray(mur, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return (1 + 1j)*np.sqrt(np.pi*f*mu0*mur/sigma)

def _parse_materials(material_properties, omega):
    """Turn the list of material dicts into (M, nf) arrays of mur and complex er.
    A scalar value applies to all frequencies; an array must match the length of f."""
    ones = np.ones_like(omega)
    mur_list, er_list = [], []
    for k, mat in enumerate(material_properties):
        unknown = set(mat) - {'sigma', 'mur', 'er'}
        if unknown:  # catch typos, which would otherwise silently fall back to the defaults
            raise ValueError(f"material_properties[{k}] has unknown key(s) {sorted(unknown)}; "
                             "allowed keys are 'sigma', 'mur', and 'er'.")
        sigma = mat.get('sigma', None)
        mur   = mat.get('mur', None)
        er    = mat.get('er', None)
        sigma = np.asarray(0 if sigma is None else sigma, dtype=float)*ones
        er    = np.asarray(1 if er is None else er, dtype=complex)*ones
        # a conductor's loss is described by sigma; allowing a complex er next to it would count the loss twice
        if np.any(sigma != 0) and np.any(er.imag != 0):
            raise ValueError(f"material_properties[{k}]: a conductor (nonzero 'sigma') must have a real-valued 'er'; "
                             "for a lossy dielectric give a complex 'er' and no 'sigma'.")
        mur_list.append(np.asarray(1 if mur is None else mur, dtype=complex)*ones)
        er_list.append(er - 1j*sigma/(omega*ep0))  # conduction loss enters as the imaginary part
    return np.array(mur_list), np.array(er_list)

def _build_grid(Rrms, boundary_loc, x_start, x_end, N):
    """Segment edges: about N segments spread uniformly over the rough transitions
    [boundary_loc -+ 5*Rrms] (the window of [1], per boundary [2]). Each uniform bulk
    gap between them gets one segment, being exact for a uniform line (Eq. (9) in [1])."""
    # transition regions, merged where they overlap, clipped to [x_start, x_end]
    intervals = sorted([b - 5*max(r, _RRMS_FLOOR), b + 5*max(r, _RRMS_FLOOR)] for r, b in zip(Rrms, boundary_loc))
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    merged = [[max(a, x_start), min(b, x_end)] for a, b in merged if min(b, x_end) > max(a, x_start)]
    # spread N segments uniformly over the total transition width; each bulk gap gets one segment
    dx = sum(b - a for a, b in merged)/N
    edges = {x_start, x_end}
    for a, b in merged:
        edges.update(np.linspace(a, b, max(1, round((b - a)/dx)) + 1))
    return np.array(sorted(edges))

def surface_impedance(f, material_properties=None, Rrms=1e-9, boundary_loc=0, distribution='norm',
                      recursion_span=None, N=None, tol=1e-6, return_field=False, B_field=True):
    """Surface impedance of a rough interface via the transmission line taper approach [1, 2].

    In most cases only the first five parameters are needed; the rest have
    safe defaults.

    Parameters
    ----------
    f : float or array_like
        Frequency in Hz. Must be positive.
    material_properties : list of dict or dict, optional
        Materials ordered from the outside medium (air, dielectric) to the
        deepest conductor, each with any of the keys ``'sigma'`` (S/m),
        ``'mur'``, ``'er'`` (defaults 0, 1, 1). Describe a conductor by
        ``sigma`` (its ``er``, if given, must be real-valued) and a lossy
        dielectric by a complex ``er``. Values can be scalars or arrays
        matching ``f``. A single dict means a conductor below vacuum.
        Default is vacuum over copper (``sigma=58e6``).
    Rrms : float or array_like, optional
        RMS roughness of each boundary; a scalar applies to all. Default 1e-9.
    boundary_loc : float or array_like, optional
        Location (mean) of each boundary, in increasing depth order. Default 0.
    distribution : str, frozen distribution, callable, or list thereof, optional
        Roughness distribution of each boundary; see :func:`get_CDF`.
        Default ``'norm'``.
    recursion_span : sequence of two floats, optional
        Distance range [x_start, x_end] to discretize. Default reaches 5*Rrms
        past every boundary (the xs = -5*Rq convention of [1]). Widen it to see
        the returned field over more distance; Zs stays referenced to the same
        plane either way.
    N : int, optional
        Segments spread over the rough transitions (each bulk gap adds one
        more). Default doubles N until Zs changes by less than ``tol``, with
        Richardson extrapolation. Pass a value to fix the discretization.
    tol : float, optional
        Convergence tolerance of the refinement, roughly the relative accuracy
        of the result. Default 1e-6.
    return_field : bool, optional
        If True, also return the field along the transition. Default is False.
    B_field : bool, optional
        Magnetic field if True (default), otherwise electric field.

    Returns
    -------
    Zs : (len(f),) ndarray
        Surface impedance as a function of frequency.
    A : (len(f), n) ndarray
        Field amplitude at the n segment midpoints, normalized to its maximum
        magnitude per frequency. Only returned when ``return_field`` is True.
    x : (n,) ndarray
        The n segment midpoints where the field is evaluated.
        Only returned when ``return_field`` is True.
    """
    # ---------------- inputs and defaults ----------------
    f = np.atleast_1d(np.asarray(f, dtype=float))  # a scalar f must still give (M, nf) arrays below
    if np.any(f <= 0):
        raise ValueError("Frequency must be non-zero and positive.")
    omega = 2*np.pi*f
    nf = f.size

    if material_properties is None:
        material_properties = [{'sigma': 0}, {'sigma': 58e6}]  # vacuum over copper
    elif isinstance(material_properties, dict):
        material_properties = [{}, material_properties]        # single conductor below vacuum
    mur, er = _parse_materials(material_properties, omega)     # (M, nf) each

    # one value per boundary; a single value applies to all
    K = len(material_properties) - 1  # number of boundaries
    Rrms = np.broadcast_to(np.asarray(Rrms, dtype=float), K)
    boundary_loc = np.broadcast_to(np.asarray(boundary_loc, dtype=float), K)
    distribution = list(np.broadcast_to(np.asarray(distribution, dtype=object), K))
    if np.any(np.diff(boundary_loc) < 0):
        raise ValueError("boundary_loc must be sorted in increasing depth order (matching material_properties).")

    # ---------------- computation span ----------------
    # Reach 5*Rq past every boundary (xs = -5*Rq of [1], Sec. III-A), not just the outermost:
    # a deeper boundary that is rougher reaches further out. Zs is referenced to x_start.
    # Zs is referenced where the material is still the outside medium; a wider span (e.g. to
    # look at the field over more distance) is de-embedded back to it, so it cannot change Zs.
    reach = 5*np.maximum(Rrms, _RRMS_FLOOR)
    reference_plane = np.min(boundary_loc - reach)
    if recursion_span is None:
        x_start, x_end = reference_plane, np.max(boundary_loc + reach)
    else:
        x_start, x_end = (float(v) for v in recursion_span)

    # intrinsic impedances of the outermost materials
    eta_outer   = np.sqrt(mu0*mur[0]/(ep0*er[0]))
    eta_bulk    = np.sqrt(mu0*mur[-1]/(ep0*er[-1]))
    gamma_outer = np.sqrt(-omega**2*mu0*mur[0]*ep0*er[0])

    # ------- surface impedance, doubling the segments until it converges -------
    # The Magnus step size is set in advance by the material profile ([3], Sec. 6), leaving
    # only its scale N. Zs is compared between successive N (the error metric of Eq. (10) in [1]) 
    # until it settles. Doubling is the standard geometric refinement: total work stays ~2x the 
    # final level (vs O(N^2) for N += 1), and the fixed 2:1 ratio gives the clean Richardson 
    # factor 1/(2**4-1) below. N_min = 64 and N_max = 16384 are only where the doubling starts and stops.
    N_levels  = [int(N)] if N is not None else [64*2**k for k in range(9)]  # 64, 128, ... 16384
    Zs_levels = []             # surface impedance at each refinement level
    converged = N is not None  # a user-given N is taken as-is; only the automatic mode can fail
    for N_budget in N_levels:
        edges = _build_grid(Rrms, boundary_loc, x_start, x_end, N_budget)
        dx = np.diff(edges)
        n  = dx.size              # actual number of segments (N_budget in the transitions, plus bulk gaps)
        x  = edges[:-1] + 0.5*dx  # segment midpoints

        # material at the two Gauss points of each segment, x +/- dx/(2*sqrt(3))
        # (Gauss-Legendre nodes +/-1/sqrt(3) mapped to the segment; Eq. (252) in [3])
        x_gauss = np.concatenate([x - np.sqrt(3)/6*dx, x + np.sqrt(3)/6*dx])
        CDF = np.array([get_CDF(x_gauss, r, loc, dist) for r, loc, dist in zip(Rrms, boundary_loc, distribution)])
        mur_gauss = mur[0][:, None] + np.einsum('kf,kn->fn', np.diff(mur, axis=0), CDF)
        er_gauss = er[0][:, None] + np.einsum('kf,kn->fn', np.diff(er, axis=0), CDF)
        mu_1, mu_2 = mu0*mur_gauss[:, :n], mu0*mur_gauss[:, n:]
        ep_1, ep_2 = ep0*er_gauss[:, :n], ep0*er_gauss[:, n:]

        # Segment quantities of the 4th-order Magnus method (Eqs. (253)-(254) in [3]): the
        # electrical length s (= gamma*dx if uniform), the products eta*tanh(gamma*dx) and
        # tanh(gamma*dx)/eta, and the correction d for the material variation inside the
        # segment (d = 0 when mu_1 = mu_2 and ep_1 = ep_2).
        mu_sum, ep_sum = mu_1 + mu_2, ep_1 + ep_2
        d = -np.sqrt(3)/12*dx**2*omega[:, None]**2*(ep_1*mu_2 - mu_1*ep_2)  # commutator, Eq. (254) in [3]
        s = np.sqrt(d**2 - 0.25*dx**2*omega[:, None]**2*mu_sum*ep_sum)
        tanh_over_s = np.tanh(s)/s  # even in s, so the branch above does not matter
        eta_tanh = tanh_over_s*0.5j*dx*omega[:, None]*mu_sum       # eta*tanh(gamma*dx)
        tanh_over_eta = tanh_over_s*0.5j*dx*omega[:, None]*ep_sum  # tanh(gamma*dx)/eta
        d = tanh_over_s*d

        # Transform from the deepest material up to the surface (output at x_start). This is
        # Eq. (9) in [1] divided by eta, with the correction d; d = 0 and multiplying by eta
        # gives back Zin = eta*(Zin + eta*tanh(gamma*dx))/(eta + Zin*tanh(gamma*dx)).
        Zin = eta_bulk.copy()  # terminate with the intrinsic impedance of the deepest material
        for i in range(n - 1, -1, -1):
            Zin = ((1 - d[:, i])*Zin + eta_tanh[:, i])/((1 + d[:, i]) + tanh_over_eta[:, i]*Zin)

        # de-embed the outside medium between x_start and the reference plane, so that
        # widening the span (to see more of the field) leaves Zs unchanged
        length = reference_plane - x_start
        if length > 0:
            t = np.tanh(gamma_outer*length)
            Zin = eta_outer*(Zin - eta_outer*t)/(eta_outer - Zin*t)
        Zs_levels.append(Zin)

        # Richardson extrapolation [4]: the 4th-order error drops by 2**4 when n doubles, so
        # 1/(2**4 - 1) of the change cancels it. Stop once the extrapolated Zs settles within tol.
        if len(Zs_levels) >= 3:
            Zs = Zs_levels[-1] + (Zs_levels[-1] - Zs_levels[-2])/(2**4 - 1)
            Zs_before = Zs_levels[-2] + (Zs_levels[-2] - Zs_levels[-3])/(2**4 - 1)
            change = np.max(np.abs(Zs - Zs_before)/np.abs(Zs))
            if change < tol:
                converged = True
                break
    if N is not None:
        Zs = Zs_levels[-1]  # fixed number of segments: no extrapolation
    if not converged:
        warnings.warn(f"Surface impedance did not converge to tol={tol:.1e} with {n} segments "
                      f"(last change {change:.1e}); returning the last result.", stacklevel=2)
    if not return_field:
        return Zs

    # ---------------- field along the transition (optional) ----------------
    # Zs needs only one segment per uniform bulk gap, but the field is meant to be looked at
    # over the whole span, so add uniform coverage of the gaps to the converged grid.
    edges = np.unique(np.concatenate([edges, np.linspace(x_start, x_end, n)]))
    dx = np.diff(edges)
    n = dx.size
    x = edges[:-1] + 0.5*dx

    # superposition of the traveling waves on each segment, at the segment midpoints
    CDF = np.array([get_CDF(x, r, loc, dist) for r, loc, dist in zip(Rrms, boundary_loc, distribution)])
    mur_mid = mur[0][:, None] + np.einsum('kf,kn->fn', np.diff(mur, axis=0), CDF)
    er_mid  = er[0][:, None] + np.einsum('kf,kn->fn', np.diff(er, axis=0), CDF)
    gamma   = np.sqrt(-omega[:, None]**2*mu0*mur_mid*ep0*er_mid)
    eta     = np.sqrt(mu0*mur_mid/(ep0*er_mid))
    tanh    = np.tanh(gamma*dx)
    E       = np.exp(-gamma*dx)      # one-way propagation over a full segment
    Eh      = np.exp(-0.5*gamma*dx)  # half segment (midpoint evaluation)
    sgn     = -1 if B_field else 1   # (-1)^m with m = 1 for magnetic waves, m = 0 for electric waves

    # reflection at the right interface of each segment, looking right (Eq. (A.4a))
    Gamma_right = np.empty((nf, n), dtype=complex)
    Zright = eta_bulk.copy()
    for i in range(n - 1, -1, -1):
        Gamma_right[:, i] = sgn*(Zright - eta[:, i])/(Zright + eta[:, i])
        Zright = eta[:, i]*(Zright + eta[:, i]*tanh[:, i])/(eta[:, i] + Zright*tanh[:, i])

    # reflection at the left interface of each segment, looking left (Eq. (A.4b))
    Gamma_left = np.empty((nf, n), dtype=complex)
    Zleft = eta_outer.copy()
    for i in range(n):
        Gamma_left[:, i] = sgn*(Zleft - eta[:, i])/(Zleft + eta[:, i])
        Zleft = eta[:, i]*(Zleft + eta[:, i]*tanh[:, i])/(eta[:, i] + Zleft*tanh[:, i])

    # transmission factor tau (Eq. (A.2)) and wave superposition (Eqs. (11) and (A.1))
    A = np.empty((nf, n), dtype=complex)
    eta_previous = eta_outer
    tau = 1
    for i in range(n):
        r = sgn*(eta[:, i] - eta_previous)/(eta[:, i] + eta_previous)  # reflection between segments (Eq. (A.3))
        tau = 1 + r if i == 0 else tau*(1 + r)*E[:, i-1]/(1 - Gamma_left[:, i-1]*r*E[:, i-1]**2)
        A[:, i] = tau*(Eh[:, i] + Gamma_right[:, i]*E[:, i]**2/Eh[:, i])/(1 - Gamma_right[:, i]*Gamma_left[:, i]*E[:, i]**2)
        eta_previous = eta[:, i]
    A = A/np.abs(A).max(axis=1, keepdims=True)  # normalize per frequency

    return Zs, A, x

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f = np.logspace(-1, 2, 200)*1e9  # 0.1 GHz to 100 GHz
    sigma_copper = 58e6

    plt.figure()
    for Rrms in [0.5e-6, 1e-6, 2e-6]:
        Zs = surface_impedance(f, [{'sigma': 0}, {'sigma': sigma_copper}], Rrms=Rrms)
        plt.plot(f*1e-9, Zs.real, lw=2, label=f'Re(Zs), Rrms = {Rrms*1e6:.1f} um')
        plt.plot(f*1e-9, Zs.imag, '--', lw=2, label=f'Im(Zs), Rrms = {Rrms*1e6:.1f} um')
    Zs_smooth = smooth_surface_impedance(f, sigma_copper)
    plt.plot(f*1e-9, Zs_smooth.real, 'k', lw=2, label='Smooth copper')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Surface Impedance (Ω)')
    plt.legend()
    plt.show()

# EOF
