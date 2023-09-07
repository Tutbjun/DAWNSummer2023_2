#Courtesy of Vadim & Gabe

import os, sys, copy

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import grizli.utils
import eazy
import multiprocessing as mp

#crazy hack to get fsps to be multithreaded
THREADCNT = 0
try:
    import fsps0
    import eazy.sps0 as sps0
    THREADCNT += 1
    try:
        import fsps1
        import eazy.sps1 as sps1
        THREADCNT += 1
        try:
            import fsps2
            import eazy.sps2 as sps2
            THREADCNT += 1
            try:
                import fsps3
                import eazy.sps3 as sps3
                THREADCNT += 1
                try:
                    import fsps4
                    import eazy.sps4 as sps4
                    THREADCNT += 1
                    try:
                        import fsps5
                        import eazy.sps5 as sps5
                        THREADCNT += 1
                        try:
                            import fsps6
                            import eazy.sps6 as sps6
                            THREADCNT += 1
                            try:
                                import fsps7
                                import eazy.sps7 as sps7
                                THREADCNT += 1
                            except:
                                pass
                        except:
                            pass
                    except:
                        pass
                except:
                    pass
            except:
                pass
        except:
            pass
    except:
        pass
except:
    import fsps
    import eazy.sps as sps
    FSPS_PATH = os.getenv('SPS_HOME')
if THREADCNT >= 1:
    fsps = fsps0
    sps = sps0
    FSPS_PATH = os.getenv('SPS0_HOME')

#THREADCNT = 1#threadcap

from astropy.cosmology import WMAP9 as cosmology
from collections import OrderedDict
from eazy.templates import read_templates_file
from scipy.interpolate import CubicSpline

#TEMP = sys.argv[-1] # IMF temperature [K]
TEMP = 45

#=========================================
# check eazy and fsps setup
grizli.utils.set_warnings()
eazy_path = os.getenv('EAZYCODE')

if not os.path.exists('templates'):
    eazy.symlink_eazy_inputs()

# Gabe's FSPS version:
# '0.4.2.dev9+g71fb3e5.d20220502'

# My FSPS version (for pdva;basel templates):
# ('0.4.2.dev9+g71fb3e5', '0.5.2.dev63')

# Alternatively, its possible to install FSPS with 
# a different set of stellar evolution tracks
# eg., mist;miles fsps version

print("=== Starting up ===================")
print(f"fsps: {fsps.__version__}\neazy: {eazy.__version__}\n")

#=========================================
# read in C2020 templates as an example
# load C2020 templates

def dict_lower_keys(d):
    # change all keys to lowercase
    keys = [k.lower() for k in d.keys()]
    vals = [v for v in d.values()]
    d_new = dict(zip(keys, vals))
    return d_new

def compare_sp_meta(my_templ, orig_templ, print_all=False):
    # loop over params
    diff = {}
    #test_dict = dict_lower_keys(my_templ.params._params)
    test_dict = dict_lower_keys(my_templ.meta)
    target_dict = dict_lower_keys(orig_templ.meta)
    for k, v in target_dict.items():
        try:
            v_sp = test_dict[k]
            if v_sp != v:
                diff[k] = [v_sp, v] # order: my, original
                print(f"{k} (test vs target): {v_sp} --- {v}")
        except:
            print(f"{k} (test vs target): nan --- {v}")

    if print_all:
        for k, v in target_dict.items():
            try:
                v_sp = test_dict[k]
                print(f"{k} (test vs target): {v_sp} --- {v}")
            except:
                pass
    return diff

def compare_templ_meta(my_templ, orig_templ, print_all=False):
    # loop over params
    diff = {}
    test_dict = dict_lower_keys(my_templ.meta)
    target_dict = dict_lower_keys(orig_templ.meta)
    for k, v in target_dict.items():
        v_my = test_dict[k]
        if v_my != v:
            diff[k] = [v_my, v] # order: my, original
            print(f"{k} (test vs target): {v_my} --- {v}")

    if print_all:
        for k, v in target_dict.items():
            v_my = test_dict[k]
            print(f"{k}:{v_my} --- {v}")
    return diff

def compare_meta(my_templ, orig_templ, print_all=False):
    diff = {}
    if isinstance(my_templ, eazy.sps.ExtendedFsps):
        diff = compare_sp_meta(my_templ, orig_templ, print_all=print_all)
    elif isinstance(my_templ, eazy.templates.Template):
        diff = compare_templ_meta(my_templ, orig_templ, print_all=print_all)
    return diff

def get_param_dict(d):
    kwargs_in = dict_lower_keys(copy.deepcopy(d))
    keys = np.array(['imf_upper_limit', 'imf_lower_limit', 'imf1', 'imf2', 'imf3',
       'vdmc', 'mdave', 'dell', 'delt', 'sbss', 'fbhb', 'pagb',
       'add_stellar_remnants', 'tpagb_norm_type', 'add_agb_dust_model',
       'agb_dust', 'redgb', 'agb', 'masscut', 'fcstar', 'evtype',
       'smooth_lsf', 'smooth_velocity', 'redshift_colors',
       'compute_light_ages', 'nebemlineinspec', 'dust_type',
       'add_dust_emission', 'add_neb_emission', 'add_neb_continuum',
       'cloudy_dust', 'add_igm_absorption', 'zmet', 'sfh', 'wgp1', 'wgp2',
       'wgp3', 'tau', 'const', 'tage', 'fburst', 'tburst', 'dust1',
       'dust2', 'logzsol', 'zred', 'pmetals', 'dust_clumps',
       'frac_nodust', 'dust_index', 'dust_tesc', 'frac_obrun', 'uvb',
       'mwr', 'dust1_index', 'sf_start', 'sf_trunc', 'sf_slope',
       'duste_gamma', 'duste_umin', 'duste_qpah', 'sigma_smooth',
       'min_wave_smooth', 'max_wave_smooth', 'gas_logu', 'gas_logz',
       'igm_factor', 'fagn', 'agn_tau', '_zcontinuous'])
    vals = []
    for key in keys:
        vals.append(kwargs_in[key])
    
    kwargs_out = dict(zip(keys, vals))
    kwargs_out['zcontinuous'] = kwargs_out.pop('_zcontinuous')
    return kwargs_out


#=========================================
### Set params for FSPS (based on COSMOS2020 templates)

templates_c2020 = read_templates_file('templates/spline_templates_v3/c2020_spline.param')#!only 17 long, but should match the gridlength
#templates_c2020 = read_templates_file('templates/templates-c2020/chabrier/fsps.param')
c2020_meta = templates_c2020[0].meta
#diff = compare_meta(sp, templates_c2020[idx_temp], print_all=True)
#print([templates_c2020[i].meta['libraries'] for i in range(len(templates_c2020))])

kws_c2020 = get_param_dict(templates_c2020[0].meta)
kws_c2020['pagb'] = 0.0 # reset some params

#=========================================
# Initialize FSPS (with eazy.sps)

SP = []
for t in range(THREADCNT):
    module = eval(f'sps{t}')
    SP.append(module.ExtendedFsps(**kws_c2020))
    print("Loaded FSPS module", t)
if THREADCNT == 0:
    SP.append(sps.ExtendedFsps(**kws_c2020))
for t,sp in enumerate(SP):
    print(f"Setting FSPS{t} parameters...")
    FSPS_PATH = eval(f'os.getenv("SPS{t}_HOME")')
    sp.set_fir_template()
    sp.set_dust(dust_obj_type='KC13')
    sp.params['dust_index'] = 0.0
    #sp.set_dust(dust_obj_type='R15')
    #sp.set_dust(dust_obj_type='C00')
    #sp.set_dust(dust_obj_type='WG00x')

    # IR extrapolation
    sp.dust_obj.extra_params['beta'] = -3.2
    sp.dust_obj.extra_params['extra_uv'] = -0.4
    x_dust_index = [0, 0.5, 1, 2, 3]
    y_dust_index = [-0.1, -0.1, 0.1, 0.2, 0.4]

    lum_filters = [153, 155, 161, 163, 270, 274]
    lum_names = ['u','v','j','k','1400','2800']

    res = eazy.filters.FilterFile(os.getenv('EAZYCODE')+'/inputs/FILTER.RES.latest')
    uvj_res = [res[153],res[155],res[161]]

    magdis = grizli.utils.read_catalog('templates/magdis/README.txt')
    bands = ['u','v','2mass_j', 'i1500', 'i2800']
    meta_pars = ['stellar_mass', 'formed_mass', 'sfr100','energy_absorbed', 
                'rline [OII]3726', 'rline [OII]3729', 'rline Hbeta4861', 
                'rline [OIII]4960', 'rline [OIII]5007', 'rline Halpha6563',
                'rline [NII]6549', 'rline [NII]6585', 
                'rline Pabeta12819', 'rline Paalpha18752', 'line [CII]157.7m', 
                'dust_index','Av', 'gas_logu', 'logzsol', 'imf_type',
                'zred']

    centers = list(np.logspace(np.log10(0.031), 1, 6)) # VR adjusted to match C2020

    step = 0.001
    ages = (10**np.arange(np.log10(3.e-4), np.log10(14.127) + step, step))
    NZ = 14 # number of redshift bins

    # load SFH from a file
    #sfh = np.loadtxt(os.path.join(os.path.dirname(__file__), 'sfh_alpha_c2020.txt'))
    sfh = np.loadtxt(os.path.join(os.path.dirname(__file__), 'sfh_custom.txt'))
    #N = sfh.shape[0]

    #=========================================
    # set the IMF.dat file
    # place correct imf.dat file in FSPS location
    imf_file = f"imf{TEMP}.dat"
    #FSPS_PATH = os.getenv('SPS_HOME')
    FSPS_PATH = os.path.join(FSPS_PATH, "data")
    IMF_CUSTOM_PATH = f"{FSPS_PATH}/fsps_imfs/{imf_file}"
    IMF_FSPS_PATH = f"{FSPS_PATH}/imf.dat"

    print("=== IMF ===========================")
    print(f" Setting IMF temperature to T={TEMP} K")
    print(f"  {IMF_CUSTOM_PATH.split('/')[-1]} ---> {IMF_FSPS_PATH.split('/')[-1]}\n")
    os.system(f'cp {IMF_CUSTOM_PATH} {IMF_FSPS_PATH}')

# fsps kwargs
kwargs = {'scale_lyman_series': 0.1, 'force_recompute': True, #!! lines scaling here!! fix balmer
        'oversample': 2.5, 'zmet': 1}

scale_lines_templ = {
    '[OII]3726': 1., '[OII]3729': 1.,
    'Halpha6563':1., 'Hbeta4861':1., 'Hgamma4340':1., 'Hdelta4102':1., 'H3970':1., 'H3889':1., 'H3835':1., 'H3798':1.,
    '[OIII]4960': 1., '[OIII]5007': 1.
}
scale_lines = scale_lines_templ
kwargs['scale_lines'] = scale_lines

# Metallicity-dependent blue templates
# Cresci, logM = 9
# https://www.researchgate.net/figure/Evolution-of-the-mass-metallicity-relation-from-local-to-high-redshift-galaxies-from_fig2_47637888
# Padova
met_redshift = [0., 0.07, 0.7, 2.2, 3.5, 6, 12, 15, 30]
met_logZ = [0., 0., -0.1, -0.5, -0.7, -0.9, -0.9, -0.9, -0.9]
met_logZ = np.array(met_logZ)
met_logZ = met_logZ / (met_logZ.max() - met_logZ.min()) * 0.6 - 0.3
met_logU = [-1, -1, -1, -1, -1, -1, -1, -1, -1]

met_bump = [2., 2., 2., 1.8, 1.4, 1.3, 1, 1, 1]
met_bump = [2., 2., 2., 1.8, 1.6, 1.3, 1, 1, 1]
met_bump = (np.array(met_bump) - 1) * 2

met_Avmax = [1.8, 1.8, 2.5, 4.2, 4., 2, 1, 1, 1]
met_Avmax = [2.0, 2.0, 2.5, 4.2, 4., 2, 1, 1, 1]
met_Avmax = np.array(met_Avmax)

met_logZ_spl = CubicSpline(np.array(met_redshift)*1.5, met_logZ)
met_logU_spl = CubicSpline(np.array(met_redshift)*1.5, met_logU)
met_bump_spl = CubicSpline(np.array(met_redshift)*0.8, met_bump)
met_Avmax_spl = CubicSpline(np.array(met_redshift), met_Avmax)

met_redshift_grid = eazy.utils.log_zgrid([0, 13], 0.2)
met_logU_grid = met_logU_spl(met_redshift_grid)
met_logZ_grid = met_logZ_spl(met_redshift_grid)
met_bump_grid = met_bump_spl(met_redshift_grid)
met_Avmax_grid = met_Avmax_spl(met_redshift_grid)

full_meta_params = ['stellar_mass', 'formed_mass', 'sfr100', 'energy_absorbed', 
                    'rline [OII]3726', 'rline [OII]3729', 'rline Hbeta4861', 
                    'rline [OIII]4960', 'rline [OIII]5007', 'rline Halpha6563', 
                    'rline [NII]6549', 'rline [NII]6585', 'rline Pabeta12819', 
                    'rline Paalpha18752', 'line [CII]157.7m', 'dust_index', 
                    'Av', 'gas_logu', 'logzsol', 'imf_type', 'zred', 'lwAgeV', 
                    'mwAge', 'lwAgeR', 'Lu', 'Lv', 'Lj', 'Lk', 'L1400', 'L2800']
delta_age_max = 0.01 # Maximum delta age relative to age of universe for SF


#=========================================
# set the stellar pop parameters for 
# individual templates
#                     Age,   Av,    Scale Em. Line Hb4861+OIII,  extra_uv?,  beta?, dust_index?
"""sp_props = np.array([[0.1,   0.005, 1.0,                         0.0,        -3.2,  0.0],
                     [0.1,   0.5,   1.0,                         0.0,        -3.2,  0.0],
                     [0.1,   0.005, 1.0,                         0.0,        -3.2,  -0.3],
                     [0.1,   0.5,   1.0,                         0.0,        -3.2,  0.0],
                     [0.1,   1.0,   1.0,                         0.0,        -3.2,  0.0],
                     [0.1,   2.0,   1.0,                         0.0,        -3.2,  0.0],
                     [0.1,   3.0,   1.0,                         -0.3,       -3.2,  0.0]])"""
#! new set
"""sp_props = np.array([[0.01,   0.005, 0.5,                         0.0,        -3.2,  0.0],#young
                     [0.01,   0.5,   0.5,                         0.0,        -3.2,  0.0],#young
                     [0.01,   3,   0.5,                         0.0,        -3.2,  0.0],#dusty, young
                     [0.01,   0.005, 0.1,                         0.0,        -3.2,  0.0],#low em. line, young
                     [0.1,   0.005, 0.5,                         0.0,        -3.2,  -0.3],
                     [0.1,   0.5,   0.5,                         0.0,        -3.2,  0.0],
                     [0.1,   3,   0.5,                         0.0,        -3.2,  0.0],#dusty
                     [0.1,   0.005, 0.1,                         0.0,        -3.2,  0.0],#low em. line
                     [0.5,   0.005,   0.5,                         0.0,        -3.2,  0.0],
                     [0.5,   0.5,   0.5,                         0.0,        -3.2,  0.0],
                     [0.5,   3,   0.5,                         -0.3,       -3.2,  0.0],#dusty
                     [0.5,   0.005, 0.1,                         0.0,        -3.2,  0.0],#low em. line
                     ])"""
parameterSpace = [
    np.logspace(np.log10(0.01), np.log10(0.5), 3), # age
    np.logspace(np.log10(0.005), np.log10(0.5), 2), # Av
    np.logspace(np.log10(0.01), np.log10(1), 3), # Hb4861+OIII
    [-0.3], # extra_uv #TODO: figure out use
    [-3.2], # beta #TODO: figure out use
    [0,-0.3], # dust_index #TODO: figure out use
    list(range(4))#index of SFH
]#TODO: add Oxygen scaling and HII scaling
#create grid of parameters
sp_props = np.array(np.meshgrid(*parameterSpace)).T.reshape(-1,7)


print("=== Template SP parameters ========")
print("\t\t Age, \t Av, \t Scale Em. Line Hb4861+OIII,  extra_uv,  beta, dust_index, sfh_index")
for i, (tage_, Av, hb_boost, extra_uv, beta, dust_index, sfh_index) in enumerate(sp_props):
    print(f"template {i+1}: \t {tage_:.3f} \t{Av:.3f} \t\t{hb_boost} \t\t\t {extra_uv}\t  {beta} \t {dust_index} \t {sfh_index}")
print()

#=========================================
# produce SP with FSPS

# loop over params sets for each template


templates = [[] for e in sp_props]
workinVars = list(enumerate(sp_props))
def worker(vars, threadID=None):
    #global templates
    if threadID == None: threadID = int(mp.current_process().name.split("-")[-1])-1
    i, props = vars
    tage_, Av, hb_boost, extra_uv, beta, dust_index, sfh_index = props
    #TODO: make sure all parameters are in use
    #!extra_uv
    #!beta
    #!dust_index
    print(f"\n === template {i} - thread {threadID} ===")

    sfh_index = int(sfh_index)
    sfh_cur = sfh[sfh_index]
    #t_c2020 = templates_c2020[i]
    #t_c2020.meta = dict_lower_keys(t_c2020.meta)
    #d = dict_lower_keys(t_c2020.meta)
    d = dict_lower_keys(c2020_meta)
    try:
        logz = np.array([d[f'logz{j}'] for j in range(NZ)])
        logu = np.array([d[f'logu{j}'] for j in range(NZ)])
    except:
        logz = np.full((NZ,), 0.0)
        logu = np.full((NZ,), -2.75)
    
    #kws_c2020 = get_param_dict(t_c2020.meta)
    module = eval(f'sps{threadID}')
    sp = module.ExtendedFsps(**kws_c2020)
    sp.set_fir_template()
    sp.set_dust(dust_obj_type='KC13')
    sp.dust_obj.extra_params['extra_uv'] = extra_uv
    sp.dust_obj.extra_params['beta'] = beta
    sp.params['dust_index'] = dust_index
    #!removed temporarily:
    """sp.params['tau'] = t_c2020.meta['tau']#! whyyyyyyyy is sfh also 1?????!?!
    sp.params['pagb'] = t_c2020.meta['pagb']
    sp.params['sfh'] = 1"""
    #!added:
    sp.params['sfh'] = 3
    sp.set_tabular_sfh(ages, sfh_cur)
    
    sp.wg00lim = np.isfinite(sp.wavelengths)
    sp.get_full_spectrum(tage=0.2) # initialization run
    #sp.emline_sigma = [sig for s in sp.emline_sigma]
    #sp.scale_lines['Lyalpha1216'] = 1e3
    #sp.scale_lines['Lybeta1025'] = 11e10
    #sp.scale_lyman_series = 1e2

    #=== scale emission lines ===============================
    scale_lines = {k:scale_lines_templ[k]*hb_boost for k in scale_lines_templ if "OIII" in k or "H " in k}
    #add rest without scaling
    for k in scale_lines_templ:
        if k not in scale_lines:
            scale_lines[k] = scale_lines_templ[k]
    kwargs['scale_lines'] = scale_lines
    
    # metallicity
    logzsol_ = 0.0
    zsol = 0.019 # mist/miles library
    met_ = 10**logzsol_ * zsol
    sp.params._params['metallicity'] = zsol
    
    kwargs['tstart'] = 0.03
    kwargs['t0'] = 0.035
    kwargs['degree'] = 1
    kwargs['end_clip'] = 1.e-3
        
    #=== scale emission lines ===============================
    sp.wg00lim = np.isfinite(sp.wavelengths)
    templ = sp.get_full_spectrum(tage=tage_, Av=Av, 
                                 get_template=True, 
                                 set_all_templates=True, **kwargs)

    templ.name = f'fsps_{TEMP}k_t{tage_:.3f}_Av{Av:.3f}'

    templ.meta['NZ'] = len(met_logZ_grid) # VR: muted
    flux_grid = []
    cont_grid = []
    unred_grid = []
    templ.meta_dict = OrderedDict()
    for k in full_meta_params:
        templ.meta_dict[k] = []

    # loop over redshifts
    print(f"\n  === {i} - template {templ.name} - thread {threadID} ===")
    print("with variables ", props)
    for iz in range(templ.meta['NZ']):
        z_i = met_redshift_grid[iz]
        sp.params['zred'] = z_i

        _mix = np.where(z_i >= magdis['zmin'])[0][-2]
        sp.set_fir_template(file='templates/magdis/'+magdis['file'][_mix])
        #sp.set_fir_template(file='templates/magdis/'+magdis['file'][_mix]), scale_pah3=0.3)

        #=== SFH setting ===================
        sp.params['sfh'] = 3
        sp.set_tabular_sfh(ages, sfh_cur)

        #logz_i = np.minimum(met_logZ_grid[iz] + ((i/(df-i0-1))**0.5)*0.5, 0.2)
        #logu_i = met_logU_grid[iz]
        uv_bump_i = np.maximum(met_bump_grid[iz], 0.) * 0.1
        if hasattr(sp.dust_obj, 'bump_ampl'):
            sp.dust_obj.bump_ampl = uv_bump_i
            templ.meta[f'BUMP{iz}'] = uv_bump_i
        elif 'delta' in sp.dust_obj.param_names:
            del sp.dust_obj.extra_params['extra_bump']
            sp.dust_obj.extra_params['extra_bump'] = uv_bump_i #*3./2

        templ.meta[f'z{iz}'] = met_redshift_grid[iz]
        templ.meta[f'logz{iz}'] = logz[iz] # logz[iz] #met_logZ_grid[iz] / 4#(i+1)**2
        templ.meta[f'logu{iz}'] = logu[iz] # logu[iz] #-1 #met_logU_grid[iz]
        sp.params['logzsol'] = logzsol_
        sp.params['gas_logz'] = met_logZ_grid[0] # templ.meta[f'LOGZ{iz}']
        sp.params['gas_logu'] = logu[iz] #logu_i        
        sp.params['compute_light_ages'] = False
        sp.meta['nebemlineinspec'] = True    
        sp.dust_obj.bump_ampl = met_bump_grid[iz]

        #=== IMF setting ===========
        #sp.params['imf_type'] = 1 # Chabrier IMF
        sp.params['imf_type'] = 5 # custom IMF
        _x = sp.get_full_spectrum(tage=tage_, Av=Av, 
                                  get_template=True, 
                                  set_all_templates=True, **kwargs)
        #print(f"logZsol={sp.params['logzsol']:.2f}, tage={sp.params['tage']} Gyr, Av={sp.Av}, imf={sp.meta['imf_type']} \n")

        #=== Template parameters ===
        for k in meta_pars:
            if k == 'sfr100':
                templ.meta_dict[k].append(sp.meta['sfr'])
            else:
                templ.meta_dict[k].append(sp.meta[k])

        # Luminosities
        for f_i, f_n in zip(lum_filters, lum_names):
            f_dens = _x.integrate_filter(res[f_i], flam=False)
            f_flux = f_dens*eazy.utils.CLIGHT*1.e10/res[f_i].pivot
            templ.meta_dict['L'+f_n].append(f_flux)

        #templ.ageV = band_ages[i,1]
        sp.params['compute_light_ages'] = True
        lw_ages = sp.get_mags(tage=tage_, bands=['v', 'sdss_r'])
        templ.meta_dict['mwAge'].append(sp.stellar_mass*1)
        templ.meta_dict['lwAgeV'].append(lw_ages[0])
        templ.meta_dict['lwAgeR'].append(lw_ages[1])
        sp.params['compute_light_ages'] = False

        # VR
        templ.meta['_zcontinuous'] = sp._zcontinuous
        templ.meta['tage'] = sp.params['tage']
        templ.meta['metallicity'] = sp.params['metallicity']

        # Identifying attributes
        templ.meta['hb_boost'] = hb_boost
        templ.meta['extra_uv'] = extra_uv
        templ.meta['beta'] = beta
        templ.meta['dust_index'] = dust_index
        templ.meta['sfh_index'] = sfh_index

        flux_grid.append(_x.flux[0,:])
        cont_grid.append(sp.templ_cont.flux[0,:])
        unred_grid.append(sp.templ_unred.flux[0,:])

    templ.NZ = templ.meta['NZ']
    templ.redshifts = met_redshift_grid
    templ.flux = np.array(flux_grid)
    templ.continuum = np.array(cont_grid)
    templ.unred = np.array(unred_grid)
    templ.IS_ALPHA = False
    #templates[i] = templ
    #templates[i].append(templ)
    return templ

#!multithreading region
pool = mp.Pool(processes=THREADCNT)
templates = pool.map(worker, workinVars, chunksize=1)
pool.close()
#!end multithreading region

print("\n=== The templates were produced: ===")
[print(f"  {i}. {t}") for i, t in enumerate(templates)]
identifiers = []


#=========================================
# plot comparison of the current templates 
# with C2020 templates


"""fig, axes = plt.subplots(2 * len(templates), 1, 
                         figsize=(8, 6*len(templates)), 
                         dpi=100, gridspec_kw={'hspace': 0.1})

axes = axes.flatten()

idxs1 = np.arange(0, 22, 2)
idxs2 = np.arange(1, 21, 2)
for i, t in enumerate(templates):
    wave_target = templates_c2020[i].wave
    flux_target = templates_c2020[i].flux
    
    n = t.flux.shape[0]
    color1 = iter(plt.cm.Blues(np.linspace(0.5, 1, n))[::-1])
    color2 = iter(plt.cm.Reds(np.linspace(0.5, 1, n))[::-1])
    color3 = iter(plt.cm.Greys(np.linspace(0.5, 1, n))[::-1])
    
    for j in range(n):
        c1 = next(color1)
        c2 = next(color2)
        c3 = next(color3)
        for k in range(t.meta['NZ']):
            try:
                t.meta[f'Z{k}'] = t.meta.pop(f'z{k}') # sps.py-compatible format
            except:
                pass
        t_resampled = t.resample(wave_target, in_place=False) # resample to common wavelength grid
        axes[idxs2[i]].plot(t_resampled.wave, t_resampled.flux[j, :], color=c1, lw=1, 
                     label='My templates') # test template
        axes[idxs2[i]].plot(wave_target, flux_target[j, :], color=c2, lw=1, 
                     label='C2020') # target template
        axes[idxs1[i]].plot(t_resampled.wave, t_resampled.flux[j, :]/flux_target[j, :], 
                     color=c3, lw=1) # test template
        x = flux_target[j, :]
        mu = t_resampled.flux[j, :]
        sig = np.sqrt(t_resampled.flux[j, :])
        chi = (x - mu) / sig
        chi_cond = np.isinf(chi) | np.isnan(chi)
        chi = np.ma.masked_where(chi_cond, chi)
        chi2 = np.sum(chi)**2
    axes[idxs1[i]].annotate(f"{chi2:.2f}", xy=(0.8, 0.9), xycoords='axes fraction')
    axes[idxs1[i]].axhline(1.0, ls=':', color='k')
    axes[idxs1[i]].loglog()
    axes[idxs1[i]].tick_params(axis="x", which='both', direction="in", pad=-22)
    axes[idxs2[i]].loglog()
    axes[idxs1[i]].set_xlim(50, 2e8)
    #axes[idxs1[i]].set_xlim(5e2, 5e3) # lya window
    axes[idxs2[i]].set_xlim(50, 2e8)
    axes[idxs1[i]].set_ylim(1e-1, 1e1)
    axes[idxs2[i]].set_ylim(1e-18, 5)
    
    nzeros = (t.flux == 0.0).all(axis=0).sum()
    #print(f"{t.name}: \t {t.flux.shape},  zero fluxes (along axis=1): {nzeros}")
plt.savefig(f'../../docs/figures-templates/{TEMP}k_alpha.png', dpi=100)"""


#=========================================
# save templates

output_dir = f'templates-custom/{TEMP}k/raw'
#output_dir = f'test-templ/{TEMP}k'
dirs = output_dir.split('/')
for i in range(len(dirs)):
    if i != len(dirs)-1: _dir = '/'.join(dirs[:i+1])
    else: _dir = output_dir
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    

templ_root = f"fsps_{TEMP}k"
param_file = f"{output_dir}/{templ_root}_alpha.param"

print("\n=== Writing params file ===========")
fp = open(param_file, 'w')
for i, templ in enumerate(templates):
    if not templ: continue
    tab = templ.to_table()
    tab_verbose = templ.to_table()
    tab['flux'] = tab['flux'].astype(np.float32)
    def foldOutTuple(tab_in, tab_out, key:str):
        inTable = np.asarray(tab_in[key]).T
        for i in range(len(tab_in[key][0])):
            tab_out[key + str(i)] = inTable[i]
        return tab_out
    tab_verbose = foldOutTuple(tab,tab_verbose,'flux')
    
    if hasattr(templ, 'continuum'):
        tab['continuum'] = templ.continuum.T.astype(np.float32)
        tab_verbose = foldOutTuple(tab,tab_verbose,'continuum')
    
    if hasattr(templ, 'unred'):
        tab['dered'] = templ.unred.T.astype(np.float32)
        tab_verbose = foldOutTuple(tab,tab_verbose,'dered')
    
    templ.ageV = tage_
    
    identifier = f'{TEMP}k_index{i}_t{templ.meta["tage"]}_Av{np.round(templ.meta["Av"], 2):.3f}_EmHIIOIII{np.round(templ.meta["hb_boost"], 2):.3f}_extrauv{np.round(templ.meta["extra_uv"], 2):.3f}_beta{templ.meta["beta"]:.3f}_dust{templ.meta["dust_index"]:.3f}_sfh{sfh_index}'
    if i < 2:
        name = f"fsps_{identifier}_bin0"
    else:
        name = f"fsps_{identifier}_bin1"
    templ.name = name
    tab.meta['ageV'] = templ.ageV
    tab_verbose.meta['ageV'] = templ.ageV
    line = f'{i+1} templates/{output_dir}/{name}.fits 1.0 0.0 1.0'
    fp.write(line+'\n')
    tab.write(f'{output_dir}/{name}.fits', overwrite=True)
    tab_verbose.write(f'{output_dir}/{name}_verbose.fits', overwrite=True)
    print(f" {output_dir}/{name}.fits")
fp.close()

# Metadata
max_NZ = np.max([templ.NZ for templ in templates if templ])
cols = ['file']
rows = []
for j, templ in enumerate(templates):
    if not templ: continue
    row = []
    row.append(templ.name)
    for k in full_meta_params:
        _val = [0.]*max_NZ
        _val[:templ.NZ] = templ.meta_dict[k]
        row.append(_val)
        if j == 0:
            cols.append(k)        
    rows.append(row)

par = grizli.utils.GTable(names=cols, rows=rows)
par['LOIII'] = par['rline [OIII]4960'] + par['rline [OIII]5007']
par['LOII'] = par['rline [OII]3726'] + par['rline [OII]3729']
par['LHaNII'] = par['rline Halpha6563'] + par['rline [NII]6549'] + par['rline [NII]6585']
par.rename_column('rline Halpha6563', 'LHa')
par.rename_column('rline Hbeta4861', 'LHb')
par.rename_column('stellar_mass', 'mass')
par.rename_column('energy_absorbed', 'energy_abs')
par.rename_column('sfr100', 'sfr')
par['sfr'].description = 'SFR over last 100 Myr'
par['LIR'] = par['energy_abs']
par['LIR'].description = 'IR luminosity = energy_abs'
cols = list(par.columns)
for c in cols:
    if 'rline' in c:
        cnew = copy.deepcopy(c)
        cnew = cnew.replace('rline', 'line')
        par.rename_column(c, cnew)

for c in par.colnames:
    try:
        par[c] = np.squeeze(par[c])
    except:
        pass

par.write(param_file+'.fits', overwrite=True)
print(f" {param_file}")
