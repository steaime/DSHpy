import sys
import os
import numpy as np
import SharedFunctions
import DSHfunctions

ProgramRoot = os.path.dirname(sys.argv[0]) + '/'
print('Program root: ' + str(ProgramRoot))

config = None

# READ CONFIG FILE
def ReadConfig(filename='config_CorrMaps.ini'):
    config = SharedFunctions.ReadConfig(str(ProgramRoot) + filename)
    return config

#exec(open(str(ProgramRoot) + 'var_names.py').read())
#exec(open(str(ProgramRoot) + 'functions.py').read())

#initialize variables and dictionaries
#exec(open(str(ProgramRoot) + 'config/config_mv_corr_map.py').read())


def ComputeCorrMaps():
    
    # EXPORT CONFIGURATION FILE
    #open(OutFolder + 'Config.txt', 'a').close()
    copyfile(ProgramRoot + 'config/ConfigFile_mv_corr_map.txt', OutFolder + 'config.txt')
    
    
    print('Calculating correlations...')
    
    raw_filename = InputFolder + MIfile_name
    
    
    if Analysis_info[K_AN_USE_PADDING]:
        pad_width   = int(Analysis_info[K_AN_SIGMA]*Analysis_info[K_AN_CUTOFF])
        I_pad       = np.empty([Analysis_info[K_AN_NUM_LAGS]+1, Analysis_info[K_AN_IMGS_NUM], Analysis_info[K_AN_ROI_SIZE][1] + 2*pad_width,\
                      Analysis_info[K_AN_ROI_SIZE][0] + 2*pad_width])
        ROIs_pad    = np.empty([Analysis_info[K_AN_IMGS_NUM], Analysis_info[K_AN_ROI_SIZE][1] + 2*pad_width,\
                      Analysis_info[K_AN_ROI_SIZE][0] + 2*pad_width])
        
    ROIs    = np.empty([Analysis_info[K_AN_IMGS_NUM], Analysis_info[K_AN_ROI_SIZE][1], Analysis_info[K_AN_ROI_SIZE][0]])
    G       = np.empty([Analysis_info[K_AN_NUM_LAGS]+1, Analysis_info[K_AN_IMGS_NUM], Analysis_info[K_AN_ROI_SIZE][1], Analysis_info[K_AN_ROI_SIZE][0]])
    
    mask    = np.pad(np.ones([Analysis_info[K_AN_ROI_SIZE][1], Analysis_info[K_AN_ROI_SIZE][0]]), pad_width, 'constant')
    I_mean  = np.empty_like(ROIs)
    norm    = np.empty_like(ROIs[0])
    
    for t in range(Analysis_info[K_AN_IMGS_NUM]):  
        temp = Load_single_img(InputFolder+MIfile_name, Analysis_info[K_AN_IMAGE_SIZE], data_format='B', pix_depth = 1,\
                           header_size = 0, image_pos = Analysis_info[K_AN_FIRST_IMG]+t, gap = 0)
    
        temp_crop = temp[Analysis_info[K_AN_ROI][1]:Analysis_info[K_AN_ROI][3], Analysis_info[K_AN_ROI][0]:Analysis_info[K_AN_ROI][2]]
        ROIs_pad[t]      = np.pad(temp_crop, pad_width, 'constant')  
        I_pad[0][t]      = np.square(ROIs_pad[t])
        
    
    for n in range(Analysis_info[K_AN_NUM_LAGS]):
        for t in range(Analysis_info[K_AN_IMGS_NUM]-Analysis_info[K_AN_LAG_LIST][n]):
    
                    I_pad[n+1][t] = np.multiply(ROIs_pad[t], ROIs_pad[t+Analysis_info[K_AN_LAG_LIST][n]])
            
        
    
    x = np.asarray(range(-pad_width, pad_width+1))
    y = np.asarray(range(-pad_width, pad_width+1))
    
    grid    = np.meshgrid(x,y)
    weights = np.exp(np.divide(np.square(grid[0])+np.square(grid[1]),-np.square(Analysis_info[K_AN_SIGMA])))    
    
    
    
    for i in range(Analysis_info[K_AN_ROI_SIZE][1]):
        for j in range(Analysis_info[K_AN_ROI_SIZE][0]):
    
            norm[i][j] = np.sum(np.multiply(weights, mask[i:i+2*pad_width+1, j:j+2*pad_width+1]))
    
    print('tau = 0...')
    for t in range(Analysis_info[K_AN_IMGS_NUM]):
        for i in range(Analysis_info[K_AN_ROI_SIZE][1]):
            for j in range(Analysis_info[K_AN_ROI_SIZE][0]):
                    
                temp_num = np.sum(np.multiply(weights, I_pad[0][t][i:i+2*pad_width+1, j:j+2*pad_width+1]))/norm[i][j]
                I_mean[t][i][j] = np.sum(np.multiply(weights, ROIs_pad[t][i:i+2*pad_width+1, j:j+2*pad_width+1]))/norm[i][j]
                
                G[0][t][i][j]   = temp_num/np.square(I_mean[t][i][j])-1 
                    
            
    filename = ('mv_corr_map_d0.dat') 
        
    WriteFile(G[0], 'f', OutFolder, filename, overwrite=True)
    
    for n in range(Analysis_info[K_AN_NUM_LAGS]):
        print('tau = ' + str(Analysis_info[K_AN_LAG_LIST][n]) + '...')
        for t in range(Analysis_info[K_AN_IMGS_NUM]-Analysis_info[K_AN_LAG_LIST][n]):
            I_pad[n+1][t] = np.multiply(ROIs_pad[t], ROIs_pad[t+Analysis_info[K_AN_LAG_LIST][n]])
            for i in range(Analysis_info[K_AN_ROI_SIZE][1]):
                for j in range(Analysis_info[K_AN_ROI_SIZE][0]):
                    
                    temp_num    = np.sum(np.multiply(weights, I_pad[n+1][t][i:i+2*pad_width+1, j:j+2*pad_width+1]))/norm[i][j]
                    
                
                    G[n+1][t][i][j]      = (temp_num/(I_mean[t][i][j]*I_mean[t+Analysis_info[K_AN_LAG_LIST][n]][i][j])-1)/G[0][t][i][j]
                    
            
        filename = ('mv_corr_map_d%s.dat') % Analysis_info[K_AN_LAG_LIST][n]   
        
        WriteFile(G[n+1], 'f', OutFolder, filename, overwrite=True)
    
    

            
    

    

