import sys
import os
import logging
from DSH import Config, MIfile, CorrMaps, VelMaps, NonAffMaps, SharedFunctions


if __name__ == '__main__':
    
    inp_fnames = []
    cmd_list = []
    g_params = {'log_suffix':''}
    param_kw = None
    for argidx in range(1, len(sys.argv)):
        # If it's something like -cmd, add it to the command list
        # Otherwise, assume it's the path of some input file to be read
        if (sys.argv[argidx][0] == '-'):
            # Special case: combinations like --param value
            if (sys.argv[argidx][:2] == '--'):
                param_kw = sys.argv[argidx][2:]
            else:
                param_kw = None
                cmd_list.append(sys.argv[argidx])
        else:
            if (param_kw is None):
                inp_fnames.append(sys.argv[argidx])
            else:
                g_params[param_kw] = sys.argv[argidx]
                param_kw = None
    if (len(inp_fnames)<=0):
        inp_fnames = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'serial_corrmap_config.ini')]
    
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
        
        if ('-skip_cmap' not in cmd_list) or ('-skip_vmap' not in cmd_list) or ('-skip_vmap_assemble' not in cmd_list) or ('-skip_displ' not in cmd_list) or ('-skip_grad' not in cmd_list):
            
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
                    
                    SharedFunctions.CheckCreateFolder(out_folder)
                    logging.basicConfig(filename=os.path.join(out_folder, 'DSH' + str(g_params['log_suffix']) + '.log'),\
                                        level=logging.DEBUG, format='%(asctime)s | %(levelname)s:%(message)s')
                    logging.info('Now starting analysis in folder ' + str(out_folder))
                    
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
    
                    # Free up memory
                    logging.info('Analysis in folder ' + str(out_folder) + ' completed. Freeing up memory and moving on...')
                    mi_file = None
                    corr_maps = None
                    vel_maps = None
        
        
        
        if ('-skip_naff' not in cmd_list):
            
            # Read options for nonaffine displacement calculation
            naffmap_kw = NonAffMaps._get_kw_from_config(conf, section='naffmap_parameters')
            
            # Loop through all 'nonaff_N' sections of the configuration file
            for cur_sec in conf.GetSections():
                if (cur_sec[:len('nonaff_')]=='nonaff_'):
                    
                    cur_outFolder = os.path.join(froot, conf.Get(cur_sec, 'out_folder'))
                    fw_corr_folder = os.path.join(froot, conf.Get(cur_sec, 'cmaps_fw_folder'))
                    bk_corr_folder = os.path.join(froot, conf.Get(cur_sec, 'cmaps_bk_folder'))
                    
                    if ('-silent' not in cmd_list):
                        print(' - ' + str(cur_sec) + ': working in folder ' + str(cur_outFolder) + '...')
            
                    logging.basicConfig(filename=os.path.join(out_folder, 'DSH' + str(g_params['log_suffix']) + '.log'),\
                                        level=logging.DEBUG, format='%(asctime)s | %(levelname)s:%(message)s')
                    logging.info('Correlation maps from folders ' + str(fw_corr_folder) + ' and ' + str(bk_corr_folder) +\
                                 ' processed to compute NonAffMaps to be saved in folder ' + str(cur_outFolder))
                    
                    # Initialize NonAffMaps object
                    naff_maps = NonAffMaps.NonAffMaps(CorrMaps.LoadFromConfig(os.path.join(fw_corr_folder, 'CorrMapsConfig.ini')),\
                                                      CorrMaps.LoadFromConfig(os.path.join(bk_corr_folder, 'CorrMapsConfig.ini')),\
                                                      cur_outFolder, **SharedFunctions.filter_kwdict_funcparams(naffmap_kw, NonAffMaps.NonAffMaps.__init__))
                    
                    logging.info('NonAffMaps object initialized. Now computing maps')

                    naff_maps.Compute()
                    
                    if ('-silent' not in cmd_list):
                        print('   ...all done!')
                    
                    logging.info('All NonAffMaps saved to folder ' + str(cur_outFolder))
                    naff_maps = None
