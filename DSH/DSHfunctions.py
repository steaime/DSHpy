import sys
import os
import numpy as np
import re
import math
import struct
from PIL import Image
import cv2
import warnings

import scipy as sp
import scipy.special as ssp
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import figure
import cmath
import bisect

from matplotlib.pyplot import quiver


def is_comment(s):
    
    condition = False
    
    if s.startswith('#'):
        condition = True
    elif s.startswith('\n'):
        condition = True
        
    return condition



def ReadConfigFile(f_conf):

    line = f_conf.readline()
    
    while is_comment(line):
        line = f_conf.readline()
        
        if not is_comment(line):
       
            line = line.strip() #remove spaces initial spaces from line
            
            #if it is a string remove the " " and return the content
            if line[0]=='"':
                value = line.strip('"')
        
            #if it starts with '[' it is an array, strip the parenthesis and turn it into an array 
            elif line[0]=='[':
                value = np.asarray(line[1:len(line)-1].split(','), dtype=int)
        
            #If it is none of the above it is a number, a float if there is a point
            elif line.find('.') > 0:
                value = float(line)
                
            else:
                value = int(line)
        
    return value



def Check_Path(path):
    
    if not os.path.isdir(path):
        if not os.path.isfile(path):
            raise IOError('Error in searching for input folder: ' + path + '\nno such file or directory')




def Load_from_file(path, filename, struct_format, start_idx=0, pix_depth=1):
    
    Check_Path(path + filename)
    
    read_handle = open(path + filename, 'rb')
    
    bytes_num = int(struct_format.strip('Bf')) * pix_depth
    
    read_handle.seek(start_idx*pix_depth)
    
    bytes_read = read_handle.read(bytes_num)
        
    bytes_array = np.asarray(struct.unpack(struct_format, bytes_read))
        
    return bytes_array



def WriteFile(data, dtype, OutFolder, filename, overwrite=True):
    
    d_shape = np.shape(data)
    
    entries = 1
    
    for i in range(len(d_shape)):
        entries *= d_shape[i]
        
    d_array = data.reshape(entries)
       
    if not os.path.isdir(OutFolder):
        os.mkdir(OutFolder)    
        
    if overwrite == True:
        f = open(OutFolder + filename, 'wb+')
        
    else:
        f = open(OutFolder + filename, 'ab+')
               
    
    for i in range(entries):
        f.write(struct.pack(dtype, d_array[i]))
        
    f.close()
    
def Check_overwrite(file_path):
    
    if os.path.isfile(file_path):
        over_all = input('\nOne or more files already exist. Do you wish to overwrite all of them? (y or n) \n')
        if over_all == 'y':
            over_bool = True
            
        else:
            over_bool = False
            
    else:
        over_bool = True
        
    if not over_bool:
    
        if os.path.isfile(file_path):
            over = input('\nFile ' + file_path + 'already exists! Do you wish to overwrite it? (y or n) \n')

    
        if over == 'n' or over == 'no':
            raise IOError('Change file name and run program again')
            
    return over_bool


def get_single_ROILine_MIfile(MIfile_handle, MI_info, Analysis_info, line_n, img_n, reshape=False):
       
    ROIsize = Analysis_info[K_AN_ROI_SIZE]
    
    #checking line number
    if (line_n < 0):
        raise IOError('MI file read error: line number index cannot be negative')
    elif (line_n * ROIsize[1] > MI_info[K_MI_IMAGE_HEIGHT]):
        raise IOError('MI file read error: line number has to be smaller than the total number of lines (' + str(MI_info[K_MI_IMAGE_HEIGHT]/ROIsize[1]) + ')')
    
    #calculating number of pixels to be read
    pixels_per_ROIline = MI_info[K_MI_IMAGE_WIDTH] * ROIsize[1]
    
    #caluclating position of line to be read inside file
    image_pos = MI_info[K_MI_TOT_HEADER_SIZE] + (img_n-1) * MI_info[K_MI_PIXELS_PER_IMAGE] * MI_info[K_MI_PIXEL_DEPTH]
    
    #accounting for line position
    line_pos = image_pos + line_n * pixels_per_ROIline * MI_info[K_MI_PIXEL_DEPTH]

    # move reading pointer at line_pos
    MIfile_handle.seek(line_pos)
    
    #number of bytes to be read is given by number of pixels times depth of each pixel
    bytes_to_read = pixels_per_ROIline * MI_info[K_MI_PIXEL_DEPTH]

    lineContent = MIfile_handle.read(bytes_to_read)
    
    if len(lineContent) < bytes_to_read:
        raise IOError('MI file read error: EOF encountered when reading image stack starting from ' + str(start_idx) +\
                        ': ' + str(len(lineContent)) + ' instead of ' + str(bytes_to_read) + ' bytes returned')
        return None
    
    #If reshape is True the content is given as a 3D array whose components are [current ROI, y, x]
    #otherwise the byte information is returner (and will be reshaped later)
    
    if reshape == True:
        
        # get data type from the depth in bytes
        struct_format = ('%s' + MI_info[K_MI_VARIABLE_FORMAT]) % pixels_per_ROIline
    
        # unpack data structure in a tuple (than converted into 1D array) of float32
        res_arr = np.asarray(struct.unpack(struct_format, lineContent))
    
        #reshape data to create the 3D array
        ROI_per_line = MI_info[K_MI_IMAGE_WIDTH]//ROIsize[0]
        data_resh = res_arr.reshape([ROIsize[1], ROI_per_line, ROIsize[0]])
        data_resh = np.swapaxes(data_resh, 0, 1)
    
        return data_resh
    
    
    return lineContent



#gets a stack of
def getROI_stack_MIfile(MIfile_handle, MI_info, Analysis_info, line_n, reshape=True):
    
    ROIsize = Analysis_info[K_AN_ROI_SIZE]
    
    start_idx = Analysis_info[K_AN_FIRST_IMG]
    
    imgs_num = Analysis_info[K_AN_IMGS_NUM]
    
    # Checking for errors in starting position (in bytes):
    if (start_idx < 0):
        raise IOError('MI file read error: starting image index cannot be negative')
    elif (start_idx >= MI_info[K_MI_IMAGE_NUMBER]):
        raise IOError('MI file read error: starting image index (' + str(start_idx) + ') has to be smaller than the number of images (' + str(MI_info[K_MI_IMAGE_NUMBER]) + ')')

    
    # Total number of bytes to read (taking into account the different channels and the number of images)
    if (imgs_num > 0):
        if (imgs_num > MI_info[K_MI_IMAGE_NUMBER] - start_idx):
            raise IOError('MI file read error: image number too large')
    else:
        imgs_num = MI_info[K_MI_IMAGE_NUMBER] - start_idx
    
    
    #initializing variables for reading data
    fileContent = tuple()
    
    pixels_per_ROIline = MI_info[K_MI_IMAGE_WIDTH] * ROIsize[1]
    
    bytes_to_read = pixels_per_ROIline * imgs_num
    
    struct_format = ('%s' + MI_info[K_MI_VARIABLE_FORMAT]) % pixels_per_ROIline
    
    #reading the selected line and storing it in a tuple
    for i in range(imgs_num):
        temp_ROIline = get_single_ROILine_MIfile(MIfile_handle, MI_info, Analysis_info, line_n, start_idx+i)
        fileContent += struct.unpack(struct_format, temp_ROIline)
    
    
    #check if the number of bytes read is correct
    if len(fileContent) < bytes_to_read:
        raise IOError('MI file read error: EOF encountered when reading image stack starting from ' + str(start_idx) +\
                        ': ' + str(len(fileContent)) + ' instead of ' + str(bytes_to_read) + ' bytes returned')
        return None
    
    #turning list first into an array, reshaping it to get a 4D array whose dimensions are [current ROI][curent frame][y][x]
    res_arr = np.asarray(fileContent)    
    
    #reshape content of 
    if reshape == True:
        ROI_per_line = MI_info[K_MI_IMAGE_WIDTH]//ROIsize[0]
        data_resh = res_arr.reshape([imgs_num, ROIsize[1], ROI_per_line, ROIsize[0]])
        data_resh = np.swapaxes(data_resh, 1, 2)
        data_resh = np.swapaxes(data_resh, 0, 1)

        return data_resh
    
    else:
        return fileContent
    
    
    
def CalculateCorrelation_ROIline(MIfile_handle, MI_info, Analysis_info, line_n):
    
    NumLags = Analysis_info[K_AN_NUM_LAGS]
    
    LagList = Analysis_info[K_AN_LAG_LIST]
    
    ROIsize = Analysis_info[K_AN_ROI_SIZE]
    
    start_idx = Analysis_info[K_AN_FIRST_IMG]
    
    imgs_num = Analysis_info[K_AN_IMGS_NUM]
    
    #load 4D array with data according to structure [current ROI][curent frame][y][x]
    ROIs = getROI_stack_MIfile(MIfile_handle, MI_info, Analysis_info, line_n)
    
    matrix_dim = np.shape(ROIs)
    
    #Initialize variables 
    I_mean  = np.zeros([matrix_dim[0], matrix_dim[1]])
    I_sq    = np.zeros([matrix_dim[0], matrix_dim[1]])
    d0      = np.zeros([matrix_dim[0], matrix_dim[1]])
    
    #fill variables
    for i in range(matrix_dim[0]):
        for j in range(matrix_dim[1]):
            I_mean[i][j]    = np.mean(ROIs[i][j])
            I_sq[i][j] = np.mean(np.square(ROIs[i][j]))
            d0[i][j]        = (I_sq[i][j]/(I_mean[i][j]*I_mean[i][j]))-1
     
    #writing I_mean to file    
    if not os.path.isdir(OutFolder):
        os.mkdir(OutFolder)    
        
    f = open(OutFolder + 'I_mean.dat', 'ab+')
        
    I_mean = I_mean.astype(float)
    
    for i in range(matrix_dim[0]):
        for j in range(matrix_dim[1]):
            f.write(struct.pack('f', I_mean[i][j]))
            
                
    #Initialize variables for calculating correlations        
    g = []
    
    for i in range(NumLags):
        
        if imgs_num-LagList[i] < 0:
            print('\nWARNING! I cannot calculate correlation with lag %s because it is larger than images number %s' % (str(LagList[i]), str(imgs_num))) 
            break
        
        g_temp = np.zeros([matrix_dim[0], imgs_num-LagList[i]])
           
        for j in range(matrix_dim[0]):
            
            
            for k in range(imgs_num-LagList[i]):
                
                g_temp[j][k] = (np.mean(np.multiply(ROIs[j][k], ROIs[j][k+LagList[i]])) / (I_mean[j][k] * I_mean[j][k+LagList[i]]) - 1) / ((d0[j][k] + d0[j][k+LagList[i]]) / 2)
        
        g.append(g_temp)
    
    return g



def CalculateCorrelation(MIfile_handle, MI_info, Analysis_info):
    
    #initialize list to contain all correlations lines
    G = []
    
    lines_num = Analysis_info[K_AN_ROI_LINES_NUM]
    
    ROI_per_line = Analysis_info[K_AN_ROI_per_line]
    
    imgs_num = Analysis_info[K_AN_IMGS_NUM]
    
    LagList = Analysis_info[K_AN_LAG_LIST]
    
    for i in range(Analysis_info[K_AN_NUM_LAGS]):
        
        G.append(np.zeros([lines_num, ROI_per_line, imgs_num - LagList[i]]))
        
    
    #loop over number of lines and store images in a list
    for i in range(lines_num):
        
        current_g = CalculateCorrelation_ROIline(MIfile_handle, MI_info, Analysis_info, i)
        
        for j in range(Analysis_info[K_AN_NUM_LAGS]):
        
            G[j][i] = current_g[j]
            
        print('{:.1%}'.format((i+1)/lines_num))
        
    for i in range(Analysis_info[K_AN_NUM_LAGS]):
        
        G[i] = np.swapaxes(G[i], 0, 2)
        G[i] = np.swapaxes(G[i], 1, 2)
        
    return G



#function that turns array into frames, i.e. arrays with uint8 values between 0 and 255
def CreateFrames(data, mode='L'):
    
    frames = 255 * np.clip((data + data.min()) / (data.max()+data.min()), 0, 1)
    
    return frames.astype(np.uint8)



def ApplyMask(frames, filename):
    
    if os.path.isfile(ProgramRoot + filename):
        f = open(ProgramRoot + filename, 'rb')
    else:
        raise IOError('Unable to create mask! No file named ' + filename + ' found')
    
    f.seek(0)
    
    bytes_read = f.read()
    
    struct_format = ('%s' + MI_info[K_MI_VARIABLE_FORMAT]) % (Analysis_info[K_AN_ROI_per_line]*Analysis_info[K_AN_ROI_LINES_NUM])
    
    mask_array = np.asarray(struct.unpack(struct_format, bytes_read))
    
    mask = mask_array.reshape([Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line]])
    
    mask = mask / mask.max()
    
    frames = np.multiply(frames, mask)
 
    return frames
    


#Create video from frames contained in data.size of frames is the same as size of data
def CreateVideo(data, FPS, filename, mask=True):

    if not os.path.isdir(OutFolder):
        os.mkdir(OutFolder)    
        
    frame_number, height, width = np.shape(data)
    
    out = cv2.VideoWriter( OutFolder + filename, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (width, height), False)
    
    if mask == True:
    
        mask_data = ApplyMask(data, Analysis_info[K_AN_MASK_FILENAME])
        norm_data = CreateFrames(mask_data)
        frames = ApplyMask(norm_data, Analysis_info[K_AN_MASK_FILENAME])
        
    else:        
        frames = CreateFrames(data)
        
    #frames = flip(frames, 255)
    frames = frames.astype(np.uint8)
    
    for i in range(frame_number):
        
        out.write(frames[i])        
        cv2.imshow('frame', frames[i])
                
    out.release()
    
    print(filename + ' released!')
    
    cv2.destroyAllWindows() 
 
    
    
#flip frames, so that high correlation is dark, for better visualization
def flip(data, rangewidth):
    
    return rangewidth - data






def Calculate_single_g_ROI(MIfile_handle, MI_info, Analysis_info, t_0):
    
    line_n  = Analysis_info[K_AN_LINE_N]
    ROI_n   = Analysis_info[K_AN_ROI_N]
        
    temp_g = CalculateCorrelation_ROIline(MIfile_handle, MI_info, Analysis_info, line_n-1)
    
    g = []
    
    for i in range(Analysis_info[K_AN_NUM_LAGS]):
        g.append(temp_g[i][ROI_n-1][t_0])

    
    return np.asarray(g)
    
    

def Calculate_g_t(MIfile_handle, MI_info, Analysis_info):
    
    G = []
    
    for i in range(Analysis_info[K_AN_t_0_NUM]):
        
        t_0 = Analysis_info[K_AN_t_0][i]
        
        G.append(Calculate_single_g_ROI(MIfile_handle, MI_info, Analysis_info, t_0))
        
        print('{:.1%}'.format((i+1)/Analysis_info[K_AN_t_0_NUM]))
    
    return G



def g_dr_relation():
    
    dr_g = np.zeros([2,4500])

    dr_g[0] = np.linspace(4.5, 0.001, num=4500)
    
    norm = 4/np.pi

    dr_g[1] = (np.square(abs(ssp.erf(sp.sqrt(-dr_g[0]*1j))))/dr_g[0])/norm
    
    dr_g[1][4499] = 1 #to prevent correlation to be higher than highest value in array
    
    return dr_g



def extract_dr_t(data, dr_g):
    
    dr = []
    
    for j in range(len(data)):
        
        dr.append(np.zeros(len(data[j])))
        
        for i in range(len(data[j])):
            
            index = bisect.bisect_left(dr_g[1], data[j][i])
            
            dr[j][i] = dr_g[0][index]
        
    return dr

def extract_dr_t_array(data, dr_g):
    
    dr = np.zeros(len(data))
        
    for i in range(len(data)-1):
            
        index = bisect.bisect_left(dr_g[1], data[i])
            
        dr[i] = dr_g[0][index]
        
    return dr



def palette():
    
    palette = []
    palette.append('b')
    palette.append('g')
    palette.append('r')
    palette.append('c')
    palette.append('m')
    palette.append('y')
    palette.append('k')
    
    return palette



def multi_plot(data, Analysis_info, fit = False, line = [], cutoff = False, x = [], save = False, filename = 'plot', location = 'best'):
    
    figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
            
    for i in range(Analysis_info[K_AN_t_0_NUM]):
        labels = 't_0 = ' + str(Analysis_info[K_AN_t_0][i])
        if fit == True:
            slope = (line[i][1] - line[i][0])/(Analysis_info[K_AN_LAG_LIST][1] - Analysis_info[K_AN_LAG_LIST][0])
            labels = labels + '\nv = ' + '{:.5f}'.format(slope)
        
        if cutoff == True:
            plt.plot(x[i], data[i], '*', label = labels, color = palette()[i % len(palette())])
            if fit == True:
                plt.plot(x[i], line[i], color = palette()[i % len(palette())])
            
        else:
            plt.plot(Analysis_info[K_AN_LAG_LIST], data[i], '*', label = labels, color = palette()[i % len(palette())])
            if fit == True:
                plt.plot(Analysis_info[K_AN_LAG_LIST], line[i], color = palette()[i % len(palette())])
        
    plt.legend(loc= location)
    plt.xlabel('tau')
    plt.ylabel('q*dr')
    
    
    if save==True:        
        
        file_number = 0

        while os.path.isfile(OutFolder + filename + '_' + str(file_number) + '.png'):
            file_number += 1
       
            
        plt.savefig(OutFolder + filename + '_' + str(file_number) +'.png')
        


def linear_fit(data, Analysis_info, cutoff = False):
    
    slope       = np.zeros(Analysis_info[K_AN_t_0_NUM])
    intercept   = np.zeros(Analysis_info[K_AN_t_0_NUM])
    r_value     = np.zeros(Analysis_info[K_AN_t_0_NUM])
    p_value     = np.zeros(Analysis_info[K_AN_t_0_NUM])
    std_err     = np.zeros(Analysis_info[K_AN_t_0_NUM])
    line        = []
    
    if cutoff == True:
        
        x_cut = []
        
        for i in range(Analysis_info[K_AN_t_0_NUM]):
            x_cut.append(np.zeros(len(data[i])))
            
            for j in range(len(data[i])):
                x_cut[i][j] = Analysis_info[K_AN_LAG_LIST][j]
            
            slope[i], intercept[i], r_value[i], p_value[i], std_err[i] = stats.linregress(x_cut[i], data[i])
            line.append(slope[i] * x_cut[i] + intercept[i])
            
        return slope, intercept, r_value, p_value, std_err, line, x_cut
        
    else:
        slope[i], intercept[i], r_value[i], p_value[i], std_err[i] = stats.linregress(Analysis_info[K_AN_LAG_LIST], dat[i])
        line.append(slope[i] * Analysis_info[K_AN_LAG_LIST] + intercept[i])

        return slope, intercept, r_value, p_value, std_err, line



def cutoff(data, cutoff_value, max_index = -1):
    
    
    cutoff_index = bisect.bisect_left(data, cutoff_value)
        
    if max_index > 0:
        if cutoff_index > max_index:
            cutoff_index = max_index
    
    data_cut = np.zeros(cutoff_index)
    
    for i in range(cutoff_index):
        data_cut[i] = data[i]
        
    return data_cut



def Calculate_v_heatmap(data, Analysis_info, err = False):
    
    num_lags = Analysis_info[K_AN_NUM_LAGS]
    
    frames_number = Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][num_lags-1]
    
    dr_g = g_dr_relation()
    
    cutoff_corr = Analysis_info[K_AN_CORR_CUTOFF]
    
    v_heatmap = np.zeros([frames_number, Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line]])
    
    v_err = np.zeros([frames_number, Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line]])
    
    for i in range(frames_number-1):
        for j in range(Analysis_info[K_AN_ROI_LINES_NUM]-1):
            for k in range(Analysis_info[K_AN_ROI_per_line]-1):
                
                t_cutoff = 0
                
                while data[t_cutoff][i][j][k] > cutoff_corr and t_cutoff < len(data)-1:
                    t_cutoff += 1
                
                g_temp  = np.zeros(t_cutoff + 1)
                x       = np.zeros(t_cutoff + 1)
                
                for l in range(t_cutoff):
                    g_temp[l]   = data[l][i][j][k]
                    x[l]        = Analysis_info[K_AN_LAG_LIST][l]
                
                dr = extract_dr_t_array(g_temp, dr_g)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, dr)
            
                v_heatmap[i][j][k] = slope
                v_err[i][j][k]     = std_err
                
        print('{:.1%}'.format((i+1)/(frames_number-1)))
    
    filename = 'v_heatmap.dat'
    
    #WriteFile(v_heatmap, 'f', OutFolder, filename)
    
    if err == True:
        return v_heatmap, v_err
    
    return v_heatmap

def g2m1(v,x):
    
    norm = 4/np.pi
    
    return (np.square(abs(ssp.erf(sp.sqrt(-x*v*1j))))/(x*v))/norm

def Calculate_v_heatmap2(data, Analysis_info, q = 1, seed = False, v0 = 0):
    
    num_lags = Analysis_info[K_AN_NUM_LAGS]
    
    frames_number = Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][num_lags-1]
    
    v_heatmap = np.zeros([frames_number, Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line]])
    
    for i in range(frames_number-1):
        for j in range(Analysis_info[K_AN_ROI_LINES_NUM]-1):
            for k in range(Analysis_info[K_AN_ROI_per_line]-1):
                
                
                g_temp  = np.zeros(len(data))
                t       = np.zeros(len(data))
                
                for l in range(len(data)-1):
                    g_temp[l]   = data[l][i][j][k]
                    t[l]        = Analysis_info[K_AN_LAG_LIST][l]
                
                if not seed:
                    v_heatmap[i][j][k] = np.float(sp.optimize.curve_fit(g2m1,q*t,g_temp)[0])
                else:
                    v_heatmap[i][j][k] = np.float(sp.optimize.curve_fit(g2m1,q*t,g_temp,p0=v0[i][j][k])[0])
                
        print('{:.1%}'.format((i+1)/(frames_number-1)))
    
    filename = 'v_heatmap.dat'
    
    #WriteFile(v_heatmap, 'f', OutFolder, filename)
    
    return v_heatmap
               
 
    
#return modulus velocity heatmap
def modulus_heatmap(data_x, data_y):
    
    modulus = sp.sqrt(np.square(data_x)+ np.square(data_y))
    
    return modulus



#returns the angle of the deformation as an angle in units of pi between -1 and 1 wrt angle theta_0
def angle_heatmap(data_x, data_y, theta_0 = 0):
    
    angle = np.zeros(np.shape(data_x))
        
    for i in range(np.shape(angle)[0]-1):
        for j in range(np.shape(angle)[1]-1):
            for k in range(np.shape(angle)[2]-1):
                    
                angle[i][j][k] = math.atan2(data_y[i][j][k], data_x[i][j][k])
                    
                if theta_0 != 0:
                    
                    angle[i][j][k] = angle[i][j][k] - theta_0
                    
                    if angle[i][j][k] < -1*(np.pi):
                        angle[i][j][k] += 2*(np.pi)
                    
                    if angle[i][j][k] > np.pi:
                        angle[i][j][k] -= 2*(np.pi)
                        
    angle = angle/(np.pi)
    
    return angle


def ImageTimes(ImgNumber, fps, ImgStart=0, t0=0):
    
    Times = np.zeros(ImgNumber)
    
    for i in range(ImgNumber):
        Times[i] = (ImgStart - t0 + i)/fps
        
    return Times
    
    
def AnimateCorrMap(Frames, FileName=None, Times=None, fLog=None, Title=None, Comment=None, ColorMap=None, OverlayFrames=None,\
                   Quiver=None, VelocityMaps=None, OverlayCmap=None, BadValue=None, MapShape=None):

   
    if Times is None:

        if Analysis_info[K_AN_IMGS_NUM] is not None:
            Times = np.asarray(range(0, len(Frames)))/int(MI_info[K_MI_ACQUISITION_FPS])
            
        elif Ani_config[K_ANI_TIME_BOOL] == True:
            raise IOError("Timestamps or acquisition framerate must be provided!")    
            

            
   
    if len(Times)<len(Frames):
        raise IOError("Time list has fewer elements (" + str(len(Times)) + ") than the number of frames (" + str(len(Frames)) + ")!")

    if OverlayFrames is not None:

        if len(OverlayFrames)<len(Frames):
            raise IOError('\nWARNING: Number of overlay frames (' + str(len(OverlayFrames)) + ') is smaller than number of frames (' +\
                        str(len(Frames)) + '). Overlay disabled')

            OverlayFrames = None

        elif len(OverlayFrames)<len(Frames):
            raise IOError('\nWARNING: Number of overlay frames (' + str(len(OverlayFrames)) + ') is larger than number of frames (' +\
                        str(len(Frames)) + '). Only the first ' + str(len(Frames)) + ' overlay frames will be displayed')

        OverlayFrames_arr = np.asarray(OverlayFrames)

        if OverlayCmap is None:
            OverlayCmap = Ani_config[K_ANI_OVRL_CMAP]

   

    if MapShape is None:
        MapShape = [np.shape(Frames[0])[1],np.shape(Frames[0])[0]]
        
    FigExtent = [0, MapShape[0], 0, MapShape[1]]

    if (Title==None):
        Title = Ani_config[K_ANI_TITLE]
        
    if (ColorMap==None):
        ColorMap = 'gray'

    if (BadValue==None):
        global_min = np.min(Frames)
        global_max = np.max(Frames)

        if OverlayFrames is not None:
            g_min_ovr = np.min(OverlayFrames_arr)
            g_max_ovr = np.max(OverlayFrames_arr)

    else:

        Frames_arr = np.asarray(Frames)
        global_min = np.min(Frames_arr[np.nonzero(np.subtract(Frames_arr, BadValue))])
        global_max = np.max(Frames_arr[np.nonzero(np.subtract(Frames_arr, BadValue))])

        if OverlayFrames is not None:
            g_min_ovr = np.min(OverlayFrames_arr[np.nonzero(np.subtract(OverlayFrames_arr, BadValue))])
            g_max_ovr = np.max(OverlayFrames_arr[np.nonzero(np.subtract(OverlayFrames_arr, BadValue))])

#check that the quiver parameter has the right shape
            
    if Ani_config[K_ANI_QUIV_BOOL] is True:
        print('Creating vector field... ')
        if Quiver is None:
            
            if VelocityMaps is None:
                VelocityMaps = Ani_config[K_ANI_VEL_FILENAMES]
                
#            if VelocityMaps[0] is None or VelocityMaps[1] is None:
#        vel_maps_filenames = FindFileNames(g_config[K_VELMAP_FOLDER_PATH], Prefix=g_config[K_VELMAP_NAME_PREFIX], Ext=g_config[K_MIFILE_NAME_EXT])
#        PrintAndLog(str(len(vel_maps_filenames)) + ' Velocity maps ' + g_config[K_VELMAP_NAME_PREFIX] + '*' + g_config[K_MIFILE_NAME_EXT] +\
#                    ' found in folder ' + g_config[K_VELMAP_FOLDER_PATH] + '.', fLog)
#            for i in [0, 1]:
#                if VelocityMaps[i] is None and len(vel_maps_filenames) > i:
#                    VelocityMaps[i] = g_config[K_VELMAP_FOLDER_PATH] + vel_maps_filenames[i]
  
            if VelocityMaps[0] is None or VelocityMaps[1] is None:
                raise IOError('Unable to identify maps for velocity components')
                
            else:
                
                bytes_to_read = Ani_config[K_ANI_NUM_FRAMES] * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line]
                struct_format = ('%sf') % bytes_to_read
                ani_start_idx = (Ani_config[K_ANI_START_IDX]-Analysis_info[K_AN_FIRST_IMG]) * 4 * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line]
                
                vel_x_array = Load_from_file(CorrFolder, Ani_config[K_ANI_VEL_FILENAMES][0], struct_format, start_idx=ani_start_idx)
                U = vel_x_array.reshape(Ani_config[K_ANI_NUM_FRAMES], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])

                vel_y_array = Load_from_file(CorrFolder, Ani_config[K_ANI_VEL_FILENAMES][1], struct_format, start_idx=ani_start_idx)
                V = vel_y_array.reshape(Ani_config[K_ANI_NUM_FRAMES], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])

                coarse_grid = Ani_config[K_ANI_COARSE]
                
                U = U[:, ::-coarse_grid[1], ::coarse_grid[0]]
                V = V[:, ::-coarse_grid[1], ::coarse_grid[0]]
                dx = np.asarray(range(0, int(Analysis_info[K_AN_IMAGE_SIZE][0]), int(Analysis_info[K_AN_ROI_SIZE][0])*coarse_grid[0]))
                dy = np.asarray(range(0, int(Analysis_info[K_AN_IMAGE_SIZE][0]), int(Analysis_info[K_AN_ROI_SIZE][1])*coarse_grid[1]))
                X,Y = np.meshgrid(dx,dy)
                X_map = []
                Y_map = []
                for i in range(len(U)):
                    X_map.append(X)
                    Y_map.append(Y)
                
                Quiver = [X_map, Y_map, U, V]
                print('X shape: ' + str((len(X_map), len(X_map[0]), len(X_map[0][0]))))
                print('Y shape: ' + str((len(Y_map), len(Y_map[0]), len(Y_map[0][0]))))
                print('U shape: ' + str((len(U), len(U[0]), len(U[0][0]))))
                print('V shape: ' + str((len(V), len(V[0]), len(V[0][0]))))
            
        if Quiver is not None:
            if len(Quiver) == 2:
                qvr_idx_uv = [0,1]
            elif len(Quiver) == 4:
                #print(Quiver[0][0][0][:10])
                #print(Quiver[1][0][:10][0])
                qvr_idx_uv = [2,3]
            else:
                raise ValueError('\nWARNING: Unexpected shape for Quiver parameter: list length should be either 2 or 4 (' +\
                            str(len(Quiver)) + ' detected): quiver option disabled')
                Quiver = None
                
        if Quiver is not None:
            qvr_len = len(Quiver[0])
            for i in range(len(Quiver)):
                if qvr_len != len(Quiver[i]):
                    raise ValueError('\nWARNING: Unexpected shape for Quiver parameter: all elements should share ' +\
                                'the same length (' + str(qvr_len) + ' vs ' + str(len(Quiver[i])) + ')')
                    Quiver = None
                    break     
                
        if Quiver is not None:
            if len(Quiver[0])<len(Frames):
                raise ValueError('\nWARNING: Number of quiver frames (' + str(len(Quiver[0])) + ') is smaller than number of frames (' +\
                            str(len(Frames)) + '). Overlay disabled')
                OverlayFrames = None
            elif len(Quiver[0])>len(Frames):
                raise ValueError('\nWARNING: Number of quiver frames (' + str(len(Quiver[0])) + ') is larger than number of frames (' +\
                            str(len(Frames)) + '). Only the first ' + str(len(Frames)) + ' overlay frames will be displayed')
            
       
    fig = plt.figure(Title, frameon=False)

#    else:
#          fig = plt.figure(Title, figsize=(g_config[K_CORRMAP_VIDEO_SIZE]*g_config[K_CORRMAP_IMGSIZE][0],g_config[K_CORRMAP_VIDEO_SIZE]*g_config[K_CORRMAP_IMGSIZE][1]))    
#          PrintAndLog(str((g_config[K_CORRMAP_VIDEO_SIZE]*g_config[K_CORRMAP_IMGSIZE][0],g_config[K_CORRMAP_VIDEO_SIZE]*g_config[K_CORRMAP_IMGSIZE][1])) + ' figure created', fLog)

   
    if (OverlayFrames is not None):
#        if g_config[K_CORRMAP_OVERLAY_ALPHA_CLIP] is None:
        g_ovr_bounds = np.asarray((g_min_ovr, g_max_ovr))

#        else:
#            g_ovr_bounds = np.asarray(g_config[K_CORRMAP_OVERLAY_ALPHA_CLIP])
        if (Ani_config[K_ANI_OVRL_ALPHA_EXP]==0):
            AlphaValues = np.full(OverlayFrames_arr.shape, Ani_config[K_CORRMAP_OVERLAY_ALPHA_NORM], dtype=np.uint8)
            alpha_weight = None

        else:

            if (Ani_config[K_ANI_OVRL_WEIGHT] == 0):
                alpha_weight = OverlayFrames_arr
                alpha_ovr_bounds = g_ovr_bounds

            elif (Ani_config[K_ANI_OVRL_WEIGHT] == 1):
                alpha_weight = 1-OverlayFrames_arr
                alpha_ovr_bounds = 1-g_ovr_bounds

            elif (Ani_config[K_ANI_OVRL_WEIGHT] == 2):
                alpha_weight = - np.log(OverlayFrames_arr)
                alpha_ovr_bounds = - np.log(g_ovr_bounds)

            else:
                raise ValueError('Unsupported value for configuration parameter ' + str(K_ANI_OVRL_WEIGHT) + ': ' + str(Ani_config[K_ANI_OVRL_WEIGHT]))

            if (np.isnan(alpha_ovr_bounds).any()):
                print('WARNING: non-numeric boundaries for normalization ')
                
            print('max1: '+ str(np.nanmax(alpha_ovr_bounds)))
            print('min: '+ str(np.nanmin(alpha_ovr_bounds)))
            
            alpha_weight_bounds = np.nan_to_num(np.power(alpha_ovr_bounds, Ani_config[K_ANI_OVRL_ALPHA_EXP]))            
            alpha_nan_mask = np.isnan(alpha_weight)
            alpha_weight[alpha_nan_mask] = np.max(alpha_weight_bounds)
            alpha_weight = np.nan_to_num(np.power(alpha_weight, Ani_config[K_ANI_OVRL_ALPHA_EXP]))
            alpha_weight = mpl.colors.Normalize(vmin=np.min(alpha_weight), vmax=np.max(alpha_weight), clip=True)(alpha_weight)
            AlphaValues = np.multiply(alpha_weight, Ani_config[K_ANI_OVRL_ALPHA_NORM])
    

#            print(alpha_weight_bounds)
#            print(OverlayFrames_arr.flatten()[:30])

            #print(alpha_weight.flatten()[:30])
            #print(AlphaValues.flatten()[:30])

        ovr_cmap = plt.cm.get_cmap(OverlayCmap)
#        if alpha_weight is not None and g_config[K_CORRMAP_OVERLAY_CMAP_ALPHAW]:
#            OverlayFramesBlend = alpha_weight
#
#        else:
        OverlayFramesBlend = mpl.colors.Normalize(vmin=np.min(alpha_weight), vmax=np.max(alpha_weight), clip=True)(OverlayFrames_arr)
        OverlayFramesBlend = ovr_cmap(OverlayFramesBlend)
        OverlayFramesBlend[..., -1] = AlphaValues

    
    
    print('Setting frame builder...')
    ax = fig.add_subplot(111, aspect='auto', autoscale_on=False, xlim=(0, MapShape[0]), ylim=(0, MapShape[1]))   

#    if g_config[K_CORRMAP_VIDEO_CMAP_CLIP] is None:
    im_vmin = global_min
    im_vmax = global_max    
#    else:
#        im_vmin = g_config[K_CORRMAP_VIDEO_CMAP_CLIP][0]
#        im_vmax = g_config[K_CORRMAP_VIDEO_CMAP_CLIP][1]
    
    im = ax.imshow(Frames[0], animated=True, vmin=im_vmin, vmax=im_vmax, cmap=ColorMap, extent=FigExtent)

    if Quiver is not None:
        quiveropts = dict(color=Ani_config[K_ANI_ARROW_COLOR],\
                          scale=Ani_config[K_ANI_ARROW_SCALE],\
                          units=Ani_config[K_ANI_ARROW_UNITS],\
                          headlength=Ani_config[K_ANI_ARROW_HEADLENGTH],\
                          pivot=Ani_config[K_ANI_ARROW_PIVOT],\
                          headwidth=Ani_config[K_ANI_ARROW_HEADWIDTH],\
                          linewidth=Ani_config[K_ANI_ARROW_LINEWIDTH],\
                          width=Ani_config[K_ANI_ARROW_WIDTH])
        
        print(quiveropts)
        
        if len(Quiver) > 2:
            qvr = ax.quiver(Quiver[0][0], Quiver[1][0], Quiver[2][0], Quiver[3][0], **quiveropts)
        else:
            qvr = ax.quiver(Quiver[0][0], Quiver[1][0], **quiveropts)
            
    
    if OverlayFrames is not None:
        ovr = ax.imshow(OverlayFramesBlend[0], animated=True, vmin=g_min_ovr, vmax=g_max_ovr, cmap=OverlayCmap, extent=FigExtent)
#        ovr = ax.imshow(OverlayFramesBlend[0], animated=True, extent=FigExtent)
   
#    if (g_config[K_CORRMAP_VIDEO_T_LABEL_POS]==None):
#        text = None
#
#    else:
#        text = ax.text(g_config[K_CORRMAP_VIDEO_T_LABEL_POS][0], g_config[K_CORRMAP_VIDEO_T_LABEL_POS][1],\
#                        't=' + str('{:0.2f}'.format(Times[0])) + 's', fontsize=g_config[K_CORRMAP_VIDEO_LABEL_FONT], color=g_config[K_CORRMAP_VIDEO_LABEL_COLOR])
 
    if Ani_config[K_ANI_TIME_BOOL] is True:
        text = ax.text(Ani_config[K_ANI_TIME_POS][0], Ani_config[K_ANI_TIME_POS][1],\
                     't=' + str('{:0.2f}'.format(Times[0])) + 's', fontsize=16, color= 'r')

#    if (g_config[K_CORRMAP_VIDEO_SCALE_POS]!=None):
#        ax.plot([g_config[K_CORRMAP_VIDEO_SCALE_POS][0], g_config[K_CORRMAP_VIDEO_SCALE_POS][1]],\
#                [g_config[K_CORRMAP_VIDEO_SCALE_POS][2], g_config[K_CORRMAP_VIDEO_SCALE_POS][3]],\
#                '-', linewidth=g_config[K_CORRMAP_VIDEO_SCALE_THICK], color=g_config[K_CORRMAP_VIDEO_LABEL_COLOR])
#        ax.text(g_config[K_CORRMAP_VIDEO_SCALE_L_POS][0], g_config[K_CORRMAP_VIDEO_SCALE_L_POS][1],\
#                g_config[K_CORRMAP_VIDEO_SCALE_LABEL], fontsize=g_config[K_CORRMAP_VIDEO_LABEL_FONT], color=g_config[K_CORRMAP_VIDEO_LABEL_COLOR])


    plt.axis('off')
    fig.tight_layout()
    #plt.show()
    
    def init():
        return 
    
    def animate(idx):
        if (Frames is not None):
            im.set_array(Frames[idx])
        if (OverlayFrames is not None):
            ovr.set_array(OverlayFramesBlend[idx])
#        if (Quiver is not None and Ani_config[K_ANI_QUIV_BOOL]==1):
#            qvr.set_UVC(Quiver[qvr_idx_uv[0]][idx], Quiver[qvr_idx_uv[1]][idx])
        if (text!=None):
            text.set_text('t=' + str('{:0.2f}'.format(Times[idx])) + 's')
        return 
    
#    if len(Frames) > 1:
#        
#        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Frames), interval=1.0/Ani_config[K_ANI_FPS], repeat_delay=1000)
#        ExportAnimation(ani, CorrFolder, Ani_config[K_ANI_FILENAME], FPS=Ani_config[K_ANI_FPS], ForceOverwrite=False, Title=Title, Artist=Analysis_info[K_USER_NAME], Comment=Comment)
#
#    else:
    
    print('Saving frames to: ' + OutFolder)
    
    for i in range(len(Frames)):

        animate(i)
        metadata = {'Title':Title, 'Artist':Analysis_info[K_USER_NAME], 'Comment':Comment}
        figopts = dict(dpi=Ani_config[K_ANI_DPI], 
                       bbox_inches='tight', 
                       metadata=metadata)
        fname = OutFolder+'frames/'+Ani_config[K_ANI_FILENAME]+'_'+str(i).zfill(4)+Ani_config[K_ANI_FRAME_EXT]
        fig.savefig(fname, transparent=True, **figopts)
        print(str('{:0.2%}'.format((i+1)/len(Frames))))



# warning: problem if fps<10!

def ExportAnimation(ani, MovieFolder, MovieFilename, FPS=10, ForceOverwrite=False, Title=None, Artist=None, Comment=None):


#    if (g_config[K_FFMPEG_PATH]==None):
#        raise IOError('path for ffmpeg.exe application not specified in global configuration file')

    if (Title==None):
        Title='My animation'


    if (Artist==None):
        Artist=g_config[K_USER_NAME]

   
    if (Comment==None):
        Comment=''

#    while True:
#        if (os.path.isfile(os.path.join(MovieFolder, MovieFilename)) and not ForceOverwrite):
#            if query_yes_no('File ' + MovieFilename + ' already present in folder ' + MovieFolder + '. Overwrite it?'):
#                ForceOverwrite = True
#
#            else:
#                MovieFilename = query_string('Specify new filename (*.mp4)', accept_if_substr='.mp4')
#
#        else:
#            break

#    if g_config[K_FFMPEG_PATH]!=None and g_config[K_FFMPEG_PATH]!='conda':
#        unicode_path = str(g_config[K_FFMPEG_PATH])
#
#        plt.rcParams['animation.ffmpeg_path'] = unicode_path

#    unicode_path = Ani_config[K_ANI_FFMPEG_PATH]
#    plt.rcParams['animation.ffmpeg_path'] = unicode_path   
    metadata = dict(title=Title, artist=Artist, comment=Comment)
    mywriter = animation.FFMpegWriter(fps=FPS, metadata=metadata) # warning: problem if fps<10!
    ani.save(MovieFolder + MovieFilename, writer=mywriter)

    print('Animation successfully exported to file {0}'.format(MovieFolder + MovieFilename))

 
    
def ImgAffineTransform(input_arr2D, Coefficients=None, FillValue=None):

    if Coefficients is None:

        Coefficients = Analysis_info[K_CORRTODISPL_NONAFF_LINTR]

    if Coefficients is None:

        return input_arr2D

    else:

        if FillValue is None:

            FillValue = Analysis_info[K_CORRTODISPL_NONAFF_CVAL]

        matrix = np.zeros((2, 2), dtype=float)

        matrix[0,0] = Coefficients[3]

        matrix[0,1] = Coefficients[2]

        matrix[1,0] = Coefficients[1]

        matrix[1,1] = Coefficients[0]

        if (len(Coefficients) >= 6):

            offset = [Coefficients[5], Coefficients[4]]

        else:

            offset = 0.0

        #print(matrix)

        #print(offset)

        return sp.ndimage.affine_transform(input_arr2D, matrix, offset=offset, output_shape=input_arr2D.shape,\

                                           order=Analysis_info[K_CORRTODISPL_NONAFF_INTERP], mode='constant', cval=FillValue)
        
        

def remove_noise(image, cutoff):
    
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if image[i][j] < cutoff:
                image[i][j] = 0
                
    return image
    
def nan_weighted_avg(data, weights, err_bool = False):

    nan_weights = np.where(np.isnan(data), 0, weights)   
    avg = (np.nansum(np.multiply(data, nan_weights)))/np.sum(nan_weights)
    
    if err_bool:
        err = 1/np.sqrt(np.sum(nan_weights))   
        return avg, err
    
    else:
        return avg

                


def average_velocity(x1,x2,y1,y2,img_num, MI_info):
    
    read_v = open('/Users/matteo/Documents/Uni/TESI/_analysis/calibration/ROI16x16/10ul-h/forward/v_heatmap.dat', 'rb')
   
    speed_conversion = 0.661*MI_info[K_MI_ACQUISITION_FPS]/(2*np.pi)
    
    v_bin = read_v.read()
    
    bytes_to_read = img_num * 80 * 64
    
    struct_format = ('%sf') % bytes_to_read
    v_array = np.asarray(struct.unpack(struct_format, v_bin))
    
    v_heatmap = v_array.reshape([img_num,80,64])
    
    channel = np.zeros([img_num,y2-y1,x2-x1])
    
    for i in range(img_num):
        channel[i] = v_heatmap[i][y1:y2, x1:x2]
    
    print('Average speed = ' + '{:.3}'.format(channel.mean()*speed_conversion) + 'um/s')
    
    return channel.mean()*speed_conversion

def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    :INPUT:
      data   - whatever data you are radially averaging.  Data is
              binned into a series of annuli of width 'annulus_width'
              pixels.

      annulus_width - width of each annulus.  Default is 1.

      working_mask - array of same size as 'data', with zeros at
                        whichever 'data' points you don't want included
                        in the radial data computations.

      x,y - coordinate system in which the data exists (used to set
               the center of the data).  By default, these are set to
               integer meshgrids

      rmax -- maximum radial value over which to compute statistics
    
    :OUTPUT:
        r - a data structure containing the following
                   statistics, computed across each annulus:

          .r      - the radial coordinate used (outer edge of annulus)

          .mean   - mean of the data in the annulus

          .sum    - the sum of all enclosed values at the given radius

          .std    - standard deviation of the data in the annulus

          .median - median value in the annulus

          .max    - maximum value in the annulus

          .min    - minimum value in the annulus

          .numel  - number of elements in the annulus

    :EXAMPLE:        
      ::
        
        import numpy as np
        import pylab as py
        import radial_data as rad

        # Create coordinate grid
        npix = 50.
        x = np.arange(npix) - npix/2.
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)
        fake_psf = np.exp(-(r/5.)**2)
        noise = 0.1 * np.random.normal(0, 1, r.size).reshape(r.shape)
        simulation = fake_psf + noise

        rad_stats = rad.radial_data(simulation, x=xx, y=yy)

        py.figure()
        py.plot(rad_stats.r, rad_stats.mean / rad_stats.std)
        py.xlabel('Radial coordinate')
        py.ylabel('Signal to Noise')
    """
    
# 2012-02-25 20:40 IJMC: Empty bins now have numel=0, not nan.
# 2012-02-04 17:41 IJMC: Added "SUM" flag
# 2010-11-19 16:36 IJC: Updated documentation for Sphinx
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)
    
    if working_mask.all()==None:
        working_mask = ny.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x.all()==None or y.all()==None:
        x1 = ny.arange(-npix/2.,npix/2.)
        y1 = ny.arange(-npiy/2.,npiy/2.)
        x,y = ny.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.sum = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad, dtype=int)
    radialdata.max = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      #import pylab as py
      #pdb.set_trace()
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.sum[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = 0
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
      else:
        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.sum[irad] = data[r<maxrad].sum()
        radialdata.std[irad]  = data[thisindex].std()
        radialdata.median[irad] = ny.median(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()
    
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata

def azimuthalAverage(image, center=None, rmax = None, stddev=False, returnradii=False, return_nr=False, return_sym=False, theta_0 = None,
                     
                     binsize=2, weights=None, steps=False, interpnan=False, left=None, right=None):

    """

    Calculate the azimuthally averaged radial profile.

 

    image - The 2D image

    center - The [x,y] pixel coordinates used as the center. The default is

             None, which then uses the center of the image (including

             fractional pixels).

    stddev - if specified, return the azimuthal standard deviation instead of the average

    returnradii - if specified, return (radii_array,radial_profile)

    return_nr   - if specified, return number of pixels per radius *and* radius

    binsize - size of the averaging bin.  Can lead to strange results if

        non-binsize factors are used to specify the center and the binsize is

        too large

    weights - can do a weighted average instead of a simple average if this keyword parameter

        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't

        set weights and stddev.

    steps - if specified, will return a double-length bin array and radial

        profile so you can plot a step-form radial profile (which more accurately

        represents what's going on)

   interpnan - Interpolate over NAN values, i.e. bins where there is no data?

        left,right - passed to interpnan; they set the extrapolated values

 

    If a bin contains NO DATA, it will have a NAN value because of the

    divide-by-sum-of-weights component.  I think this is a useful way to denote

    lack of data, but users let me know if an alternative is prefered...

   

    """       

    

    # Calculate the indices from the image

    y, x = np.indices(image.shape)

 
    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
 
    if weights is None:
        weights = np.ones(image.shape)

    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper) 

    if rmax == None:    
        nbins = (np.round( r.max() / binsize)+1)
        
    else:        
        nbins = (np.round( rmax / binsize)+1)

    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)

    # but we're probably more interested in the bin centers than their left or right sides...

    bin_centers = (bins[1:]+bins[:-1])/2.0

 
    # Find out which radial bin each point in the map belongs to

    whichbin = np.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1

    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape

    if stddev:
        radial_prof = np.array([image.flat[whichbin==b].std() for b in range(1,nbins+1)])

    else:
        radial_prof = np.array([np.sum((image*weights).flat[whichbin==b]) / weights.flat[whichbin==b].sum() for b in range(1,int(nbins+1))])

    if return_sym:        
        if theta_0 == None:          
            raise ValueError("Angle for x-axis is needed in order to calculate symmetry of radial profile.")
        
        sin = np.sin(np.arctan2( y - center[1], x - center[0] ) + theta_0 + np.pi)        
        sym = np.array([np.nansum((image*weights*sin*bin_centers[b-1]).flat[whichbin==b]) / (weights.flat[whichbin==b].sum()*2*np.pi) for b in range(1,int(nbins+1))])

    #import pdb; pdb.set_trace()

 
    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel()
        yarr = np.array(zip(radial_prof,radial_prof)).ravel()
        return xarr,yarr

    elif returnradii:                
        return bin_centers,radial_prof

    elif return_nr:
        return nr,bin_centers,radial_prof
    
    elif return_sym:        
        return bin_centers,radial_prof,sym,sin

    else:
        return radial_prof


def Load_single_img(filename, img_size, data_format, pix_depth = 1, header_size = 0, image_pos = 0, gap = 0):
      
    Check_Path(filename)      
    
    read_handle = open(filename, 'rb')  
        
    pix_per_img = img_size[0] * img_size[1]
    bytes_per_image = pix_per_img * pix_depth    
    start_pos = header_size + (bytes_per_image+gap)*(image_pos-1) 
    
    read_handle.seek(start_pos)      
    
    bytes_read = read_handle.read(bytes_per_image)    
    struct_format = ('%s' + data_format) % pix_per_img    
    bytes_array = np.asarray(struct.unpack(struct_format, bytes_read))
    
    img = bytes_array.reshape([img_size[1], img_size[0]])
    
    return img



    
    
    
    



