import numpy as np
from astropy.visualization import ZScaleInterval as zs

#===================================================
#=== plotting

def default_colors():
    import matplotlib.pyplot as plt
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return colors

def set_tick_params_ax(ax, kwargs):
    ax = ax.tick_params(**kwargs)
    return ax

def set_ax(ax, kwargs):
    ax = ax.set(**kwargs)
    return ax

def set_ax_xticks(ax, ticks, minor):
    ax = ax.set_xticks(ticks, minor=minor)
    return ax

def set_ax_yticks(ax, ticks, minor):
    ax = ax.set_yticks(ticks, minor=minor)
    return ax


def set_tick_params(axes, idxs, kwargs):
    if isinstance(idxs, list) | isinstance(idxs, np.ndarray):
        axes_mod = [set_tick_params_ax(ax, kwargs) for ax in axes.flatten()[idxs]]
        axes.flatten()[idxs] = axes_mod
    elif idxs == 'all':
        axes = [set_tick_params_ax(ax, kwargs) for ax in axes.flatten()]
        axes = np.array(axes)
    return axes

def set_axes(axes, idxs, kwargs):
    if isinstance(idxs, list) | isinstance(idxs, np.ndarray):
        axes_mod = [set_ax(ax, kwargs) for ax in axes.flatten()[idxs]]
        axes.flatten()[idxs] = axes_mod
    elif idxs == 'all':
        axes = [set_ax(ax, kwargs) for ax in axes.flatten()]
        axes = np.array(axes)
    return axes

def set_axes_xticks(axes, idxs, ticks, minor=False):
    if isinstance(idxs, list) | isinstance(idxs, np.ndarray):
        axes_mod = [set_ax_xticks(ax, ticks, minor) for ax in axes.flatten()[idxs]]
        axes.flatten()[idxs] = axes_mod
    elif idxs == 'all':
        axes = [set_ax_xticks(ax, ticks, minor) for ax in axes.flatten()]
        axes = np.array(axes)
    return axes

def set_axes_yticks(axes, idxs, ticks, minor=False):
    if isinstance(idxs, list) | isinstance(idxs, np.ndarray):
        axes_mod = [set_ax_yticks(ax, ticks, minor) for ax in axes.flatten()[idxs]]
        axes.flatten()[idxs] = axes_mod
    elif idxs == 'all':
        axes = [set_ax_yticks(ax, ticks, minor) for ax in axes.flatten()]
        axes = np.array(axes)
    return axes


#===================================================
#=== integration

def get_cfs(counts, cfs=[0.025, 0.13, 0.50, 0.87, 0.975]):
        ''' Find the counts that correspond to the confidence intervals provided.
        Params:
        ------
        counts: array, list; binned data from plt.hist function or np.histogram (or 2d).
        cf : scalar, the confidence intervals requested -  e.g., [0.68,0.95,0.99]
            
        Returns:
        -------
        levels : scalar, the count values corresponding to the requested confidence regions
        '''
        
        counts_sorted = np.copy(np.sort(counts, axis=None))[::-1].astype(np.int64)
        total = counts_sorted.sum()
        cumsum = np.cumsum(counts_sorted)
        counts_cfs = np.zeros_like(cfs)
        for i, cf in enumerate(cfs):
            count = counts_sorted[cumsum>=cf*total]
            if len(count) > 1:
                count = counts_sorted[cumsum>=cf*total][0]
            counts_cfs[i] = count
        return counts_cfs

def get_cfs_cont(x, cfs=[0.67, 0.95, 0.99], mask_values=np.nan, axis=None):
        ''' Find the x-levels closests to the confidence intervals provided.
        
        Params:
        ------
        x: ndarray, list; binned data from plt.hist function or np.histogram (or 2d).
        mask_values: scalar; values to mask out in 'x'.
        cf : scalar; the confidence intervals requested -  e.g., [0.68,0.95,0.99]
        axis : scalar, None; axis along which to calculate confidence intervals
            
        Returns:
        -------
        levels : scalar, the count values corresponding to the requested confidence regions
        '''
        
        # sort, mask bad values and integrate
        x_sorted = np.copy(np.sort(x, axis=axis)[::-1]) # sort zero-level xs
        total = x_sorted.sum(axis=axis)
        cumsum = np.cumsum(x_sorted-x.min(), axis=axis)
        cumsum /= cumsum.max() # CDF
        x_cfs = np.zeros(shape=(len(cfs),))
        
        # compute levels by integration
        for i, cf in enumerate(cfs):
            idxs = np.where(cumsum >= cf)
            """idxs_unique = np.unique(idxs[0], return_index=True)
            if len(idxs) > 2:
                idxs_coords = idxs[0][idxs_unique[1]], idxs[1][idxs_unique[1]]
            elif len(idxs) == 1:
                idxs_coords = idxs[0][idxs_unique[1]]
            """
            level = x_sorted[idxs[0][0]]
            x_cfs[i] = level
        return x_cfs

def get_zscale(data):
    # zscale image limits
    zScale = zs()
    interval = zScale.get_limits(data.flatten())
    vmin, vmax = interval
    return vmin, vmax


#===================================================
#=== table manipulation

def get_matches(strings, allcols, get_idxs=False, exclude=None):
    # select a subsample of columns
    matches = []
    strings = np.atleast_1d(strings)
    if not (isinstance(allcols, list) | \
            isinstance(allcols, np.ndarray) |\
            isinstance(allcols, str)):
        allcols = list(allcols)
    allcols = np.atleast_1d(allcols)
    if get_idxs:
        # get matches and idxs
        for j, s in enumerate(strings):
            for i, col in enumerate(allcols):
                if s.lower() in col.lower():
                    matches.append([j, i, col])
        matches = np.array(matches)
    else:
        # get matches only
        for s in strings:
            for i, col in enumerate(allcols):
                if s.lower() in col.lower():
                    matches.append(col)
        matches = np.array(matches)
    if exclude is not None:
        exclude = np.atleast_1d(exclude)
        for s_exc in exclude:
            for s in matches:
                if s_exc.lower() in s.lower():
                    matches = np.delete(matches, np.where(matches==s))
    return matches


def match_catalogs(samp_x, samp_y, cat_x, cat_y, max_sep=1.0):
    import astropy
    import astropy.units as u
    
    from astropy.coordinates import SkyCoord
    
    if not isinstance(max_sep, astropy.units.quantity.Quantity):
        max_sep = max_sep * u.degree
    
    sample = SkyCoord(ra=samp_x*u.degree, dec=samp_y*u.degree)
    catalog = SkyCoord(ra=cat_x*u.degree, dec=cat_y*u.degree)
    idx, d2d, d3d = sample.match_to_catalog_sky(catalog)
    samp_sel = d2d < max_sep
    return samp_sel, idx

#===================================================
#=== astro 

#=== magnitude zeropoints
# Bessel+1998 ZPs for f_lambda
zps_lam = {'U': -0.152, 
           'B': -0.602,
           'V': 0.000, 
           'R': 0.555, 
           'I': 1.271, 
           'J': 2.655, 
           'H': 3.760, 
           'K': 4.906, 
           'Kp': 4.780, 
           'L': 6.775, 
           'L*': 7.177}

zps_nu = {'U': 0.770, 
          'B': -0.120,
          'V': 0.000, 
          'R': 0.186, 
          'I': 0.444, 
          'J': 0.899, 
          'H': 1.379, 
          'K': 1.886, 
          'Kp': 1.826, 
          'L': 2.765, 
          'L*': 2.961}

#=== eazy-py FILTER.RES filter dictinary

filts = { #  NIRCam,     HST-WFC3   
         'F090W': 363, 'F105W': 202,
         'F115W': 364, 'F125W': 203,
         'F150W': 365, 'F140W': 204,
         'F182M': 370, 'F160W': 205,
         'F200W': 366, 'F435W': 233,
         'F210M': 371, 'F606W': 214,
         'F277W': 375, 'F775W': 216,
         'F335M': 381, 'F814W': 239,
         'F356W': 376, 'F850LP': 240,
         'F410M': 383,
         'F430M': 384,
         'F444W': 358,
         'F460M': 385,
         'F480M': 386,
         }

#==========================================
#=== redshifts: photo-zs vs spec-zs
def phot_spec_zs_stats(x, y):
    """ 
    Basic Statistics of photo-zs vs spec-zs comparison.
    
    x: spec-zs
    y: photo-zs
    """
    # eta: frac of catastrophic outliers
    above = y > x + (1 + x) * 0.15 # outliers
    below = y < x - (1 + x) * 0.15
    outlier = above | below
    eta = outlier.sum() / len(outlier)

    # sigma_nmad: normalized median absolute deviation
    dz = y - x
    dz_abs = np.abs(dz - np.median(dz))
    nmad = 1.48 * np.median(dz_abs / (1 + x))

    # bias
    b = np.median(y - x)
    
    return {'eta': eta, 'nmad': nmad, 'bias': b}

#=== radii: galaxy profiles

def r80(r50, n):
    """ 
    80% light radius (Miller+2019)
    
    r50:    50%-light radius
    n:      Sersic index
    """
    return r50 * (0.0012*n**3 - 0.0123*n**2 + 0.5092*n + 1.2646)

def r20(r50, n):
    """ 
    20% light radius (Miller+2019) 
    
    r50:    50%-light radius
    n:      Sersic index
    """
    return r50 * (-0.0008*n**3 + 0.0178*n**2 - 0.1471*n + 0.6294)

#===========================================
#=== stellar masses: completeness limits (COSMOS2020)

def mass_compl_sf_imf(z):
    # Rusakov+2023, 95% mass completeness
    from astropy.modeling.powerlaws import Schechter1D
    p = [9.56, -1.63, -1.01]
    model = Schechter1D(*p)
    m = model(z)
    m[m <= 0] = 0.0
    return m

def mass_compl_qs_imf(z):
    # Rusakov+2023, 95% mass completeness
    from astropy.modeling.powerlaws import Schechter1D
    p = [9.10, -1.89, -1.03]
    model = Schechter1D(*p)
    m = model(z)
    m[m <= 0] = 0.0
    return m

def mass_compl_sf_w22(z):
    # Weaver+2022, 95% mass completeness
    from astropy.modeling.powerlaws import Schechter1D
    p = [10.65, -1.03, -1.01]
    model = Schechter1D(*p)
    m = model(z)
    m[m <= 0] = 0.0
    return m

def mass_compl_qs_w22(z):
    # Weaver+2022, 95% mass completeness
    from astropy.modeling.powerlaws import Schechter1D
    p = [8.30, -2.37, -1.05]
    model = Schechter1D(*p)
    m = model(z)
    m[m <= 0] = 0.0
    return m
