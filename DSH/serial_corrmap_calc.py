import sys
import os
from DSH import Config, MIfile, CorrMaps, VelMaps, SharedFunctions


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
    
    if ('-silent' not in cmd_list):
        print('\n\nBATCH CORRELATION MAP CALCULATOR\nWorking on {0} input files'.format(len(inp_fnames)))
    
    # Loop through all configuration files
    for cur_inp in inp_fnames:
        if ('-silent' not in cmd_list):
            print('Current input file: ' + str(cur_inp))
            
        # Read global section
        conf = Config.Config(cur_inp)
        num_proc = conf.Get('global', 'n_proc', 1, int)
        px_per_chunk = conf.Get('global', 'px_per_proc', 1, int)
        kernel_specs = conf.Get('global', 'kernel_specs')
        lag_list = conf.Get('global', 'lag_list', [], int)
        froot = conf.Get('global', 'root', '')
        
        # Loop through all 'input_N' sections of the configuration file
        for cur_sec in conf.GetSections():
            if (cur_sec[:len('input_')]=='input_'):
                
                # Read current input section
                mi_fname = os.path.join(froot, conf.Get(cur_sec, 'mi_file'))
                if ('-silent' not in cmd_list):
                    print(' - ' + str(cur_sec) + ': working with ' + str(mi_fname) + '...')
                meta_fname = os.path.join(froot, conf.Get(cur_sec, 'meta_file'))
                out_folder = os.path.join(froot, conf.Get(cur_sec, 'out_folder'))
                img_range = conf.Get(cur_sec, 'img_range', None, int)
                crop_roi = conf.Get(cur_sec, 'crop_roi', None, int)
                
                # Initialize image and correlation files
                mi_file = MIfile.MIfile(mi_fname, meta_fname)
                corr_maps = CorrMaps.CorrMaps(mi_file, out_folder, lag_list, kernel_specs, img_range, crop_roi)
                
                # Calculate correlation maps
                if ('-skip_cmap' not in cmd_list):
                    if ('-silent' not in cmd_list):
                        print('    - Computing correlation maps (multiprocess mode to be implemented)')
                    corr_maps.Compute(silent=True, return_maps=False)
                    
                if (('-skip_vmap' not in cmd_list) or (num_proc > 1 and '-skip_vmap_assemble' not in cmd_list) or ('-skip_displ' not in cmd_list) or ('-skip_grad' not in cmd_list)):
                    
                    # Read options for velocity calculation
                    vmap_kw = VelMaps._get_kw_from_config(conf, section='velmap_parameters')
                    
                    # Initialize MelMaps object
                    vel_maps = VelMaps.VelMaps(corr_maps, **SharedFunctions.filter_kwdict_funcparams(vmap_kw, VelMaps.VelMaps.__init__))
                    
                # Calculate velocity maps
                if ('-skip_vmap' not in cmd_list):
                    
                    if (num_proc == 1):
                        if ('-silent' not in cmd_list):
                            print('    - Computing velocity maps (single process)')
                        vel_maps.Compute(**SharedFunctions.filter_kwdict_funcparams(vmap_kw, VelMaps.VelMaps.Compute))
                    else:
                        if ('-silent' not in cmd_list):
                            print('    - Computing velocity maps (splitting computaton in {0} processes)'.format(num_proc))
                        vel_maps.ComputeMultiproc(num_proc, px_per_chunk, assemble_after=False,\
                                                  **SharedFunctions.filter_kwdict_funcparams(vmap_kw, VelMaps.VelMaps.ComputeMultiproc))
                                            
                if (num_proc > 1 and '-skip_vmap_assemble' not in cmd_list):
                    
                    if ('-silent' not in cmd_list):
                        print('    - Assembling velocity maps from multiprocess outputs')
                    vel_maps.AssembleMultiproc(os.path.join(out_folder, '_vMap.dat'))
                    
                if ('-skip_displ' not in cmd_list):
                    vel_maps.CalcDisplacements()
                    
                if ('-skip_grad' not in cmd_list):
                    vel_maps.CalcGradients()
                    
                if ('-silent' not in cmd_list):
                    print('   ...all done!')
