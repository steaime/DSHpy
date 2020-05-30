import sys
import os
from DSH import Config, MIfile, CorrMaps


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
                corr_maps.Compute(silent=True, return_maps=False)