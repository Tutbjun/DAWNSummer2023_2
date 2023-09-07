import eazy, os

import numpy as np

if not os.path.exists('templates'):
    eazy.symlink_eazy_inputs() # symlinks to FILTER.RES and templates/
if not os.path.exists('eazy-output'):
    os.mkdir('eazy-output')
    

def get_atten_dict(filts, mw_ebv=0.0061):
    """ Corrects for Galactic extinction """
    
    import eazy, extinction

    if not os.path.exists('templates'):
        eazy.symlink_eazy_inputs()

    filt_names = list(filts.keys())
    filt_num = list(filts.values())
    
    # load .res filters
    flt_eazy = eazy.filters.FilterFile()
    filts_eazy = np.array(flt_eazy.filters)

    # get filter names and idxs from .res
    myfilts = filts_eazy[filt_num]
    myfilt_pivot = np.array([f.pivot for f in myfilts])

    # compute reddening
    Av = 3.1 * mw_ebv
    atten_mag = extinction.fm07(myfilt_pivot, Av) 
    atten_factor = 10 ** (-0.4 * atten_mag)
    atten_dict = dict(zip(filt_names, atten_factor))
    
    return atten_dict

def write_tran_file(d, keys_id=['id id', 'z_spec z_spec'], 
                    fext='zphot.translate', out_path='./', fwrite=True):
    n_extra = len(keys_id)
    outfile = np.empty(len(d)*2+n_extra, dtype='U200')
    outfile[0:n_extra] = keys_id
    for i, (name, idx) in enumerate(d.items()):
        idx_out = i * 2 + n_extra
        outfile[idx_out] = f"{name}_flux F{idx}"
        outfile[idx_out+1] = f"{name}_err E{idx}"
    fname = os.path.join(out_path, f'{fext}.translate')
    if fwrite:
        np.savetxt(fname, outfile, fmt='%s')
    return fname

def write_zp_file(d, fext='zphot.zeropoint', out_path='./', fwrite=True):
    outfile = np.empty(len(d)*2, dtype='U200')
    for i, (name, [idx, zp]) in enumerate(d.items()):
        outfile[i] = f"F{int(idx)}  {zp}  # {name}"
    fname = os.path.join(out_path, f'{fext}.zeropoint')
    if fwrite:
        np.savetxt(fname, outfile, fmt='%s')
    return fname

def get_param_file(fext='zphot.param', out_path='./', fwrite=True):
    fdir = os.path.dirname(eazy.__file__)
    fpath = os.path.join(fdir, 'data/zphot.param.default')
    fname = os.path.join(out_path, f'{fext}.param')
    if fwrite:
        os.system(f'cp {fpath} {fname}')
    return fname

def eazy_init_photoz(params, fparam=None, ftran=None, fzp=None, 
                     kwargs={'load_prior': True, 
                             'load_products': False}):
    
    # file names an paths
    catalog_name = params['cat_path'].split('/')[-1].split('.')[0]
    if not os.path.exists(params['cat_path']):
        os.mkdir(params['cat_path'])
    
    # params for eazy
    out_path = os.path.join(params['out_path'], catalog_name)
    params = {"CATALOG_FILE": params['cat_path'],
              "FIX_ZSPEC": params['FIX_ZSPEC'],
              "TEMPLATES_FILE": params['templ_path'],
              "MAIN_OUTPUT_FILE": out_path}
    
    # photoz config
    kwargs = dict(param_file=fparam, 
                  translate_file=ftran, 
                  zeropoint_file=fzp, 
                  params=params,
                  **kwargs)
    pz = eazy.photoz.PhotoZ(**kwargs)
    return pz

def write_config(cat_name, filts, zps, keys_id=['id id', 'z_spec z_spec'], 
                 out_path='./', fwrite=True):
    
    # check outpath exists
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    # filter keys and idx from .res
    myfilts_keys = list(filts.keys())
    myfilts_idx = list(filts.values())    
    
    # load .res filters
    flt = eazy.filters.FilterFile()
    filt = np.array(flt.filters)
    filt_names = np.array([f.name for f in filt])

    # get filter names and idxs from .res
    myfilt = filt[myfilts_idx]
    myfilt_wave = np.array([f.pivot for f in myfilt])

    # write .translate and .zeropoint files
    translate_dict = dict(zip(myfilts_keys, myfilts_idx))
    zp_dict = dict(zip(filts.keys(), zip(filts.values(), zps)))
    fnames = {}
    fnames['ftran'] = write_tran_file(translate_dict, keys_id, fwrite=fwrite,
                                      fext=cat_name, out_path=out_path)
    #fnames['fparam'] = get_param_file(fext=cat_name, out_path=out_path, 
    #                                  fwrite=fwrite)
    fnames['fzp'] = write_zp_file(zp_dict, fext=cat_name, 
                                  out_path=out_path, fwrite=fwrite,)

    return myfilts_idx, fnames

def run_eazy(params, fnames_cfg, n_proc=-1, idx=None):
    
    #=== fit the catalog & save output
    pz = eazy_init_photoz(params, **fnames_cfg) # get a photoz object

    # fix z_spec
    pz.fit_catalog(idx=idx, n_proc=n_proc)
    zout, hdu = pz.standard_output(idx=idx)
    
    return zout, hdu

def run_eazy_full(catalog_path, filts, keys_id, n_proc=-1, idx=None):
    
    # file names
    catalog_name = catalog_path.split('/')[-1].split('.')[0]
    filt_num, fnames_cfg = write_config(catalog_name, filts, keys_id)
    
    # fit catalog
    zout, hdu = run_eazy(catalog_path, fnames_cfg, n_proc, idx)
    
    return zout, hdu

if __name__ == '__main__':
    
    from astropy.cosmology import Planck18
    
    cosmo = Planck18
    
    # log all camera filters
    flt = eazy.filters.FilterFile()

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

    mw_reddening = get_atten_dict(filts)

    # get zeropoints
    zps = [1.0]*len(filts)
    
    #=== set up paths for eazy

    # catalog paths
    cat_name = 'gds_jades_eazy'
    cat_path = f'../data/{cat_name}.fits'
    keys_id = ['ID id', 'RA ra', 'DEC dec', 'z_spec z_spec']

    # template paths
    templ_paths = ["templates/sfhz/corr_sfhz_13.param",
                   "templates/sfhz/blue_sfhz_13.param",
                   "templates/templates-c2020/45k/fsps_45k.param"]
    out_names = [f.split('/')[-1].split('.')[0] for f in templ_paths]
    out_paths = [f"eazy-output/{f}" for f in out_names]
    paths = np.array([templ_paths, out_paths]).T

    # iterate over tempalte sets
    for tpath, opath in paths[2:]:
        
        params = {"cat_path": cat_path,
                  "templ_path": tpath,
                  "out_path": opath,
                  "FIX_ZSPEC": 'n',
                  "USE_ZSPEC_FOR_REST": 'n',
                  "Z_MAX": 12.0,
                  "H0": cosmo.H0,
                  "OMEGA_M": cosmo.Om0,
                  "OMEGA_L": cosmo.Ode0,
                  "CATALOG_FORMAT": 'fits'}
        
        # write eazy config files
        filt_num, fnames = write_config(cat_name, filts, zps, keys_id,
                                        out_path=opath)

        # run eazy
        #idx = np.array([0])
        idx = None
        zout, hdu = run_eazy(params, fnames, n_proc=-1, idx=idx)