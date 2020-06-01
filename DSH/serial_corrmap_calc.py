import sys
import os
from DSH import Config, MIfile, CorrMaps
from multiprocessing import Process


if __name__ == '__main__':
    
    inp_fnames = []
    cmd_list = []
    for argidx in range(1, len(sys.argv)):
        # If it's something like -cmd, add it to the command list
        # Otherwise, assume it's the path of some input file to be read
        if (sys.argv[argidx][0] == '-'):
            cmd_list.append(sys.argv[argidx])
        else:
            inp_fnames.append(sys.argv[argidx])
    if (len(inp_fnames)<=0):
        inp_fnames = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serial_corrmap_config.ini')]
        
    for cur_inp in inp_fnames:
        conf = Config.Config(cur_inp)
        kernel_specs = conf.Get('global_settings', 'kernel_specs')
        lag_list = conf.Get('global_settings', 'lag_list', [], int)
        froot = conf.Get('global_settings', 'root', '')
        for cur_sec in conf.GetSections():
            if (cur_sec != 'global_settings'):
                mi_fname = os.path.join(froot, conf.Get(cur_sec, 'mi_file'))
                meta_fname = os.path.join(froot, conf.Get(cur_sec, 'meta_file'))
                out_folder = os.path.join(froot, conf.Get(cur_sec, 'out_folder'))
                img_range = conf.Get(cur_sec, 'img_range', None, int)
                crop_roi = conf.Get(cur_sec, 'crop_roi', None, int)
                mi_file = MIfile.MIfile(mi_fname, meta_fname)
                corr_maps = CorrMaps.CorrMaps(mi_file, out_folder, lag_list, kernel_specs, img_range, crop_roi)
                if ('-skip_cmap' not in cmd_list):
                    corr_maps.Compute(silent=True, return_maps=False)
                if ('-skip_vmap' not in cmd_list):
                    num_proc = conf.Get('vmap', 'n_proc', 1, int)
                    vmap_trange = conf.Get('vmap', 'tRange', None, int)
                    vmap_kw = {'qValue':conf.Get('vmap', 'qValue', 1.0, float),\
                               'tRange':vmap_trange,\
                               'lagRange':conf.Get('vmap', 'lagRange', None, int),\
                               'signed_lags':conf.Get('vmap', 'signed_lags', False, bool),\
                               'consecutive_only':conf.Get('vmap', 'consecutive_only', True, bool),\
                               'allow_max_holes':conf.Get('vmap', 'allow_max_holes', 0, int),\
                               'mask_opening_range':conf.Get('vmap', 'mask_opening_range', None, int),\
                               'conservative_cutoff':conf.Get('vmap', 'conservative_cutoff', 0.3, float),\
                               'generous_cutoff':conf.Get('vmap', 'generous_cutoff', 0.15, float),\
                               'silent':conf.Get('vmap', 'silent', True, bool),\
                               'debug':conf.Get('vmap', 'debug', False, bool)}
                    if (num_proc == 1):
                        corr_maps.ComputeVelocities(**vmap_kw)
                    else:
                        if (vmap_trange is None):
                            vmap_trange = [0, corr_maps.outputShape[0], 1]
                        else:
                            if (vmap_trange[1]<0):
                                vmap_trange[1] = corr_maps.outputShape[0]
                            if (len(vmap_trange) < 3):
                                vmap_trange.append(1)
                        start_t = vmap_trange[0]
                        end_t = vmap_trange[1]
                        num_t = (end_t-start_t) // num_proc
                        step_t = vmap_trange[2]
                        all_tranges = []
                        for pid in range(num_proc):
                            all_tranges.append([start_t, start_t+num_t, step_t])
                        all_tranges[-1][1] = end_t
                        proc_list = []
                        for pid in range(num_proc):
                            cur_kw = vmap_kw.copy()
                            cur_kw['tRange'] = all_tranges[pid]
                            cur_p = Process(target=corr_maps.ComputeVelocities, args=('Parabolic', cur_kw['qValue'], cur_kw['tRange'], cur_kw['lagRange'],\
                                                                                      cur_kw['signed_lags'], cur_kw['consecutive_only'], cur_kw['allow_max_holes'],\
                                                                                      cur_kw['mask_opening_range'], cur_kw['conservative_cutoff'], cur_kw['generous_cutoff'],\
                                                                                      cur_kw['silent'], False, cur_kw['debug'], '_'+str(pid).zfill(2)))
                            proc_list.append(cur_p)
                        for cur_p in proc_list:
                            cur_p.join()