import os
import numpy as np
import sys
import struct
import collections
import logging

from DSH import Config as cf
from DSH import SharedFunctions as sf

_data_depth = {'b':1, 'B':1, '?':1, 'h':2, 'H':2, 'i':4, 'I':4, 'f':4, 'd':8}
_data_types = {'b':np.int8, 'B':np.uint8, '?':bool, 'h':np.int16, 'H':np.uint16, 'i':np.int32, 'I':np.uint32, 'f':np.float32, 'd':np.float64}


def MergeMIfiles(MergedFileName, MIfileList, MergedMetadataFile=None, MergeAxis=0, MoveAxes=[], FinalShape=None):
    """Merge multiple image files into one image file
    
    Parameters
    ----------
    MergedFileName : full path of the destination merged MIfile
    MIfileList : list of MIfile objects or of 2-element list [MIfilename, MedatData]
                 Image shape and pixel format must be the same for all MIfiles
    MergedMetadataFile : if not None, export metadata of merged file
    MergeAxis :  Stitch MIfiles along specific axis.
                 Default is 0 (z axis or time). MIfiles need to have the same shape
                 along the other axes
    MoveAxes :   list of couples of indices, of the form (ax_pos, ax_dest)
                 If not empty, after merging and before writing to output file, do a series of
                 np.moveaxis(res, ax_pos, ax_dest) moves
    FinalShape : if not None, reshape final output (eventually after moving axes) to given shape
                 Shape along the merged axis will be disregarded and automatically computed
                 (can set this to -1)
                 
    Returns
    -------
    outMIfile : merged MIfile 
    """
    
    if (len(MoveAxes)>0):
        if (type(MoveAxes[0]) is int):
            MoveAxes = [MoveAxes]
    logging.info('MIfile.MergeMIfiles() procedure started. ' + str(len(MIfileList)) + ' MIfiles to be merged into ' + str(MergedFileName))
    strLog = 'Merging along Axis ' + str(MergeAxis)
    if len(MoveAxes)>0:
        strLog += '; followed by np.moveaxis moves: ' + str(MoveAxes)
    if FinalShape is None:
        strLog += '. No final reshaping'
    else:
        strLog += '. Reshape output to ' + str(FinalShape)
    logging.info(strLog)
    
    # Load all MIfiles and generate output metadata
    mi_in_list = []
    out_meta = {
        'hdr_len' : 0,
        'shape' : [0, 0, 0],
        'px_format' : None,
        'fps' : 0.0,
        'px_size' : 0.0
        }
    for midx in range(len(MIfileList)):
        if (type(MIfileList[midx]) is list):
            add_mi = MIfile(MIfileList[midx][0], MIfileList[midx][1])
            logging.debug('MergeMIfiles(): adding new MIfile object with filename ' + str(MIfileList[midx][0]))
        else:
            add_mi = MIfileList[midx]
            logging.debug('MergeMIfiles(): adding existing MIfile object (' + str(MIfileList[midx].FileName) + ')')
        mi_in_list.append(add_mi)
        cur_format = add_mi.DataFormat()
        cur_MIshape = add_mi.GetShape()
        if (MergeAxis < 0):
            MergeAxis += len(cur_MIshape)
        out_meta['shape'][MergeAxis] += cur_MIshape[MergeAxis]
        if (midx == 0):
            for ax in range(len(cur_MIshape)):
                if ax!=MergeAxis:
                    out_meta['shape'][ax] = cur_MIshape[ax]
            out_meta['px_format'] = cur_format
            out_meta['fps'] = add_mi.GetFPS()
            out_meta['px_size'] = add_mi.GetPixelSize()
        else:
            for ax in range(len(cur_MIshape)):
                if (ax!=MergeAxis and out_meta['shape'][ax]!=cur_MIshape[ax]):
                    raise IOError('Cannot merge MIfiles of shapes ' + str(out_meta['shape']) +\
                                  ' and ' + str(cur_MIshape) + ' along axis ' + str(MergeAxis) +\
                                  ' (sizes on axis ' + str(ax) + ' do not match)')
            assert out_meta['px_format'] == cur_format, 'MIfiles should all have the same pixel format'
        logging.debug('Current shape is ' + str(cur_MIshape) + '. Output shape updated to ' + str(out_meta['shape']))
    
    if (FinalShape is not None):
        re_shape = list(FinalShape.copy())
        re_shape[MergeAxis] = int(np.prod(out_meta['shape']) / (re_shape[MergeAxis-1]*re_shape[MergeAxis-2]))
        #for move in MoveAxes:
        #    re_shape = sf.MoveListElement(re_shape, move[0], move[1])
        assert np.prod(re_shape)==np.prod(out_meta['shape']), 'An error occurred trying to reshape MIfile of shape ' + str(out_meta['shape']) +\
                             ' into shape ' + str(re_shape) + ': pixel number is not conserved (' + str(np.prod(re_shape)) +\
                             '!=' + str(np.prod(out_meta['shape'])) + ')!'
        out_meta['shape'] = list(re_shape)
        logging.debug('Output shape should be ' + str(re_shape))
        
    if (MergedMetadataFile is not None):
        conf = cf.Config()
        conf.Import(out_meta, section_name='MIfile')
        conf.Export(MergedMetadataFile)
    
    outMIfile = MIfile(MergedFileName, out_meta)
    if (MergeAxis==0 and len(MoveAxes)==0):
        for cur_mifile in mi_in_list:
            outMIfile.WriteData(cur_mifile.Read(closeAfter=True), closeAfter=False)
            logging.debug('MIfile ' + str(cur_mifile.FileName) + ' read and directly transfered to output MIfile')
        outMIfile.Close()
    else:
        write_data = mi_in_list[0].Read(closeAfter=True)
        logging.debug('Writing buffer initialized with first MIfile (' + str(mi_in_list[0].FileName) + '). Shape is ' + str(write_data.shape))
        for midx in range(1, len(mi_in_list)):
            cur_buf = mi_in_list[midx].Read(closeAfter=True)
            write_data = np.append(write_data, cur_buf, axis=MergeAxis)
            logging.debug('MIfile #' + str(midx) + ' (' + str(mi_in_list[midx].FileName) + ') with shape ' + str(cur_buf.shape) +\
                          ' appended to writing buffer along axis ' + str(MergeAxis) + '. Current shape is ' + str(write_data.shape))
        for move in MoveAxes:
            write_data = np.moveaxis(write_data, move[0], move[1])
            logging.debug('Axis ' + str(move[0]) + ' moved to position ' + str(move[1]) + '. Current shape is ' + str(write_data.shape))
        outMIfile.WriteData(write_data, closeAfter=True)
        logging.debug('Final buffer with shape ' + str(write_data.shape) + ' written to output MIfile ' + str(outMIfile.FileName))
    
    return outMIfile

def MIcrop(source_file, source_metadata, dest_file, dest_metadata, crop_zRange=None, crop_ROI=None):
    return MIcopy(source_file, source_metadata, dest_file, dest_metadata, crop_zRange=crop_zRange, crop_ROI=crop_ROI)

def MIcopy(source_file, source_metadata, dest_file, dest_metadata, crop_zRange=None, crop_ROI=None):
    """Crops or reslices a MIfile and saves it in a different file
    
    Parameters
    ----------
    source_file: full path of source MIfile
    source_metadata: full path of source metadata or dictionary with metadata
    dest_file: full path of destination MIfile
    dest_metadata: full path where to export destination metadata
    crop_zRange: crop zRange
    crop_ROI: crop ROI
    """
    cur_mi = MIfile(source_file, source_metadata)
    cur_mi.Export(dest_file, dest_metadata, zRange=crop_zRange, cropROI=crop_ROI)
    
def ValidateROI(ROI, ImageShape, replaceNone=False):
    """Validates a Region Of Interest (ROI)
    
    Parameters
    ----------
    ROI : [topleftx (0-based), toplefty (0-based), width, height]
              width and/or height can be -1 to signify till the end of the image
    ImageShape : shape of image [row_number, column_number]
    replaceNone : if True, converts None input into full image range
    """
    if (ROI is None):
        if replaceNone:
            return [0, 0, int(ImageShape[1]), int(ImageShape[0])]
        else:
            return None
    else:
        assert (ROI[0] >= 0 and ROI[0] < ImageShape[1]), 'Top left coordinate (' + str(ROI[0]) + ') must be in the range [0,' + str(ImageShape[1]-1) + ')'
        assert (ROI[1] >= 0 and ROI[1] < ImageShape[0]), 'Top left coordinate (' + str(ROI[1]) + ') must be in the range [0,' + str(ImageShape[0]-1) + ')'
        if (ROI[2] < 0):
            ROI[2] = int(ImageShape[1] - ROI[0])
        if (ROI[3] < 0):
            ROI[3] = int(ImageShape[0] - ROI[1])
        assert (ROI[2] + ROI[0] <= ImageShape[1]), 'ROI ' + str(ROI) + ' incompatible with image shape ' + str(ImageShape)
        assert (ROI[3] + ROI[1] <= ImageShape[0]), 'ROI ' + str(ROI) + ' incompatible with image shape ' + str(ImageShape)
        return ROI

def Validate_zRange(zRange, zSize, replaceNone=True):
    return sf.ValidateRange(zRange, zSize, MinVal=0, replaceNone=replaceNone)

def ValidateBufferIndex(img_idx, buffer=None, buf_indexes=None):
    """Checks if image index is already in buffer and returns buffer position

    Parameters
    ----------
    img_idx : index of the image, 0-based. If -N, it will get the Nth last image
    buffer  : None or 3D array. Buffer of images
    buf_indexes : None or list of indexes of images in buffer. 
                  If None, buf_indexes = [0, 1, ..., buffer.shape[0]-1]
    """
    if buffer is not None:
        if buf_indexes is None:
            if len(buffer) > img_idx:
                return img_idx
        else:
            if img_idx in buf_indexes:
                return buf_indexes.index(img_idx)
    return None

def ReadBinary(fname, shape, px_format, offset=0, endian=''):
    """ Read binary file to np.ndarray
    
    Parameters
    ----------
    fname : str, full path to file
    shape : tuple with shape of output array (int). It will dictate how many values to read
    px_format : char with pixel format (see _data_types)
    offset : int, start reading from offset, in bytes
    endian : {'>','<',''}. Use '>' to read big-endian, '<' to read little-endian, '' to use system default
    
    Returns
    -------
    res : ndarray with binary content, None if file is not found
    """
    if fname is None:
        return None
    elif os.path.isfile(fname):
        with open(fname, 'rb') as fraw:
            num_read_vals = int(np.prod(shape))
            fraw.seek(offset)
            res = np.asarray(struct.unpack((endian + '%s' + px_format) % num_read_vals,\
                                           fraw.read(num_read_vals*_data_depth[px_format]))).reshape(shape)
            return res
    else:
        logging.warn('DSH.MIfile.ReadBinary WARNING: binary file {0} not found. None returned'.format(fname))
        return None

def WriteBinary(fname, data, data_format, hdr_list=None):
    """Write data to file
    
    Parameters
    ----------
    fname : str, full path to file
    data : numpy ndarray with data to be written
    px_format : char with pixel format (see _data_types)
    hdr_list : list with data to be written at the beginning of the binary file
    """
    fwrite = open(fname, 'wb')
    if (hdr_list is not None):
        buf = bytes()
        logging.debug('writing {0} elements to binary header'.format(len(hdr_list)))
        for elem in hdr_list:
            buf += struct.pack(elem)
        fwrite.write(buf)
    buf = bytes()
    data = data.flatten().astype(_data_types[data_format])
    fwrite.write(struct.pack(('%s' + data_format) % len(data), *data))
    fwrite.close()
    
class MIfile():
    """ Class to read/write multi image file (MIfile) """
    
    def __init__(self, FileName, MetaData=None):
        """Initialize MIfile
        
        Parameters
        ----------
        FileName : filename of multi image file (full path, including folder)
                    it can be None: in this case Metadata will still be loaded
                                    in this case, if the option 'filename' is found in the Metadata, Filename will be updated
        MetaData : string, dict or none. 
                    if string: filename of metadata file
                    if dict: dictionary with metadata. 
                             dict keys will become options of the 'MIfile' section of the metadata
                    if None: metadata filename will be generated starting from FileName,
                             by removing the extension and adding _medatada.ini
                             ex: MIfile.dat > MIfile_metadata.ini
        """
        self.FileName = FileName
        self.ReadFileHandle = None
        self.WriteFileHandle = None
        self.debugMode = True
        logging.debug('MIfile object created with filename ' + str(FileName))
        if (MetaData is None and FileName is not None):
            MetaData = os.path.splitext(FileName)[0] + '_metadata.ini'
            logging.debug('MIfile - Metadata filename automatically generated: ' + str(MetaData))
        self._load_metadata(MetaData)
    
    def __repr__(self):
        return '<MIfile: %s+%sx%sx%sx%s bytes>' % (self.hdrSize, self.ImgNumber, self.ImgHeight, self.ImgWidth, self.PixelDepth)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| MIfile class: |'
        str_res += '\n|---------------+---------------'
        str_res += '\n| Filename      : ' + str(self.FileName)
        str_res += '\n| Header        : ' + str(self.hdrSize) + ' bytes'
        str_res += '\n| Shape         : ' + str(self.Shape) + ' px'
        str_res += '\n| Pixel format  : ' + str(self.Endianness) + str(self.PixelFormat) + ' (' + str(self.PixelDepth) + ' bytes/px)'
        str_res += '\n| Status        : '
        str_status = ''
        if (self.WriteFileHandle is not None):
            str_status = 'Open for writing'
        if (self.ReadFileHandle is not None):
            if (len(str_status)>0):
                str_status += ' and for reading'
            else:
                str_status = 'Open for reading'
        if (len(str_status)<=0):
            str_status = 'Closed'
        str_res += str_status
        str_res += '\n|---------------+---------------'
        return str_res

    def __del__(self):
        self.Close()

    def OpenForReading(self, fName=None):
        if (self.ReadFileHandle is None):
            if (fName is None):
                fName = self.FileName
            self.ReadFileHandle = open(fName, 'rb')
            self.ReadingFileName = fName
        
    def Read(self, zRange=None, cropROI=None, closeAfter=False):
        """Read data from MIfile
        
        Parameters
        ----------
        zRange : image index range, 0-based [minimum, maximum, step]
                if None, all images will be read
        cropROI : Region Of Interest to be read
                if None, full images will be read
        
        Returns
        -------
        res_3D : 3D numpy array
        """
        self.OpenForReading()
        zRange = self.Validate_zRange(zRange)
        if (zRange[2] == 1 and cropROI is None and self.gapBytes==0):
            res_3D = self.GetStack(start_idx=zRange[0], imgs_num=zRange[1]-zRange[0])
        else:
            res_3D = []
            for img_idx in range(zRange[0], zRange[1], zRange[2]):
                res_3D.append(self.GetImage(img_idx=img_idx, cropROI=cropROI))
            res_3D = np.asarray(res_3D)
        if (closeAfter):
            self.Close()
        return res_3D
    
    def GetImage(self, img_idx, cropROI=None, buffer=None, buf_indexes=None, buffer_crop=True):
        """Read single image from MIfile
        
        Parameters
        ----------
        img_idx : index of the image, 0-based. If -N, it will get the Nth last image
        cropROI : if None, full image is returned
                  otherwise, [topleftx (0-based), toplefty (0-based), width, height]
                  width and/or height can be -1 to signify till the end of the image
        buffer  : None or 3D array. Buffer of images already read
        buf_indexes : None or list of indexes of images in buffer. 
                      If None, buf_indexes = [0, 1, ..., buffer.shape[0]-1]
        buffer_crop : if True, assume that images in buffer need to be cropped to specific ROI
                      otherwise, assume that images in buffer are already cropped to specific ROI
        """
        img_idx = int(img_idx)
        if (img_idx<0):
            img_idx += self.ImgNumber
        idx_in_buffer = ValidateBufferIndex(img_idx, buffer, buf_indexes)
        if (cropROI is None):
            if (idx_in_buffer is None):
                return self.GetStack(start_idx=img_idx, imgs_num=1).reshape(self.ImgHeight, self.ImgWidth)
            else:
                return buffer[idx_in_buffer]
        else:
            cropROI = self.ValidateROI(cropROI)
            if (idx_in_buffer is None):
                if (cropROI[0]==0 and cropROI[2]==self.ImgWidth):
                    res_arr = self._read_pixels(px_num=cropROI[2]*cropROI[3], seek_pos=self._get_offset(img_idx=img_idx, row_idx=cropROI[1], col_idx=cropROI[0]))
                    return res_arr.reshape(cropROI[3], cropROI[2])
                else:
                    res = []
                    for row_idx in range(cropROI[1],cropROI[1]+cropROI[3]):
                        res.append(self._read_pixels(px_num=cropROI[2], seek_pos=self._get_offset(img_idx=img_idx, row_idx=row_idx, col_idx=cropROI[0])))
                    return np.asarray(res)
            else:
                if (buffer_crop):
                    return buffer[idx_in_buffer][cropROI[1]:cropROI[1]+cropROI[3],cropROI[0]:cropROI[0]+cropROI[2]]
                else:
                    return buffer[idx_in_buffer]
    
    def GetStack(self, start_idx=0, imgs_num=-1):
        """Read contiguous image stack from MIfile
        WARNING: there is no control on eventual gap between images.
        unless you are sure that there is no gap, use MIfile.Read()
        
        Parameters
        ----------
        start_idx : index of the first image, 0-based
        imgs_num : number of images to read. If -1, read until the end of the file
        
        Returns
        -------
        3D numpy array with shape [num_images , image_height (row number) , image_width (column number)]
        """
        start_idx = int(start_idx)
        if (imgs_num < 0):
            imgs_num = self.ImgNumber - start_idx
        res_arr = self._read_pixels(px_num=imgs_num * self.PxPerImg, seek_pos=self._get_offset(img_idx=start_idx))
        return res_arr.reshape(int(imgs_num), self.ImgHeight, self.ImgWidth)
    
    def zAverage(self, zRange=None, cropROI=None, memSave=False):
        """Calculate z average from MIfile stack
        
        Parameters
        ----------
        zRange : image index range, 0-based [minimum, maximum, step]
                if None, all images will be read
        cropROI : Region Of Interest to be read
                if None, full images will be read
        memSave : bool flag. If true, images will be read one by one to save memory
        
        Returns
        -------
        res_2D : 2D numpy array (float)
        """
        if memSave:
            self.OpenForReading()
            zRange = self.Validate_zRange(zRange)
            cropROI = self.ValidateROI(cropROI, replaceNone=True)
            z_list = list(range(*zRange))
            res2D = np.zeros([cropROI[3], cropROI[2]], dtype=float)
            for zidx in z_list:
                res2D = np.add(res2D, self.GetImage(zidx, cropROI))
            res2D = np.divide(res2D, len(z_list))
            return res2D
        else:
            return np.mean(self.Read(zRange, cropROI), axis=0)

    def GetTimetraces(self, pxLocs, zRange=None):
        """Returns z axis profile for a given set of pixels in the image
        
        Parameters
        ----------
        pxLocs : list of pixel locations, each location being a tuple (row, col)
        zRange : range of time (or z) slices to sample
        
        Returns
        -------
        If only one pixel was asked, single 1D array
        Otherwise, 2D array, one row per pixel
        """
        list_z = list(range(*self.Validate_zRange(zRange)))
        if (type(pxLocs[0]) not in [list, tuple, np.ndarray]):
            pxLocs = [pxLocs]
        res = np.empty((len(pxLocs), len(list_z)))
        for zidx in range(len(list_z)):
            for pidx in range(len(pxLocs)):
                res[pidx,zidx] = self._read_pixels(px_num=1, seek_pos=self._get_offset(img_idx=list_z[zidx],\
                   row_idx=pxLocs[pidx][0], col_idx=pxLocs[pidx][1]))
        if (len(pxLocs) == 1):
            return res.flatten()
        else:
            return res
    
    def Export(self, mi_filename, metadata_filename, zRange=None, cropROI=None):
        """Export a chunk of MIfile to a second file
        
        Parameters
        ----------
        mi_filename : filename of the exported MIfile
        metadata_filename : filename of the exported metadata
        zRange : range of images to be exported. if None, all images will be exported
        cropROI : ROI to be exported. if None, full images will be exported
        """
        self.OpenForWriting(mi_filename)
        mi_chunk = self.Read(zRange, cropROI)
        exp_meta = self.GetMetadata().copy()
        exp_meta['hdr_len'] = 0
        exp_meta['gap_bytes'] = 0
        exp_meta['shape'] = list(mi_chunk.shape)
        if ('fps' in exp_meta):
            val_zRange = self.Validate_zRange(zRange)
            exp_meta['fps'] = float(exp_meta['fps']) * 1.0 / val_zRange[2]
        exp_config = cf.Config()
        exp_config.Import(exp_meta, section_name='MIfile')
        exp_config.Export(metadata_filename)
        self.WriteData(mi_chunk)
    
    def OpenForWriting(self, fName=None, WriteHeader=None, appendMode=False):
        """Open image for writing
        
        Parameters
        ----------
        fName : filename. If None, self.FileName will be used
        WriteHeader : list of dictionnaries each one with two entries: format and value
            if None, no header will be written (obsolete, for backward compatibility)
        """
        if (fName is None):
            fName = self.FileName
        if (not self.IsOpenWriting()):
            if appendMode:
                self.WriteFileHandle = open(fName, 'ab')
                logging.debug('MIfile ' + str(fName) + ' opened for appending')
            else:
                self.WriteFileHandle = open(fName, 'wb')
                logging.debug('MIfile ' + str(fName) + ' opened for writing')
        else:
            logging.debug('MIfile ' + str(fName) + ' was already open')
        if (WriteHeader is not None):
            buf = bytes()
            for elem_hdr in WriteHeader:
                buf += struct.pack(elem_hdr['format'], elem_hdr['value'])
            self.WriteFileHandle.write(buf)
    
    def WriteData(self, data_arr, closeAfter=True, appendMode=False):
        """Write data to file
        """
        self.OpenForWriting(appendMode=appendMode)
        nelems = sf.IterableShape(data_arr)
        if (nelems > self.MaxBufferSize):
            logging.info('MIfile.WriteData: writing large array ({0} elements) in bunches of {1}'.format(nelems, self.MaxBufferSize))
            for i in range(0, nelems, np.uint64(self.MaxBufferSize)):
                j = np.uint64(min(i+self.MaxBufferSize, nelems))
                self.WriteFileHandle.write(self._imgs_to_bytes(data_arr.flat[i:j], self.PixelFormat, do_flatten=True))
                logging.debug('MIfile.WriteData: x-sections {0}-{1} out of {2} written to file'.format(i, j, nelems))
        else:
            self.WriteFileHandle.write(self._imgs_to_bytes(data_arr, self.PixelFormat, do_flatten=True))
        if isinstance(data_arr, np.ndarray):
            logging.debug('ndarray of shape ' + str(data_arr.shape) + ' successfully written to MIfile ' + str(self.FileName))
        else:
            logging.debug('non-ndarray of size ' + str(sys.getsizeof(data_arr)) + ' successfully written to MIfile ' + str(self.FileName))
        if (closeAfter):
            self.Close(write=True, read=False)
            logging.debug('MIfile ' + str(self.FileName) + ' closing after writing')
        else:
            self.WriteFileHandle.flush()
            logging.debug('MIfile ' + str(self.FileName) + ' flushing without closing after writing')

    def Close(self, read=True, write=True):
        if (write and self.WriteFileHandle is not None):
            logging.debug('MIfile closed writing file handle')
            self.WriteFileHandle.close()
            self.WriteFileHandle = None
        if (read and self.ReadFileHandle is not None):
            logging.debug('MIfile closed reading file handle')
            self.ReadFileHandle.close()
            self.ReadFileHandle = None
    
    def GetMetadata(self, section='MIfile'):
        """Returns dictionary with metadata
        """
        return self.MetaData.ToDict(section=section)
    
    def MetadataToDict(self, section=None):
        """Returns dictionary with metadata
        """
        return self.MetaData.ToDict(section=section)
    
    def IsOpenWriting(self):
        if (self.WriteFileHandle is None):
            return False
        else:
            return (not self.WriteFileHandle.closed)
    def IsOpenReading(self):
        if (self.ReadFileHandle is None):
            return False
        else:
            return (not self.ReadFileHandle.closed)
    def GetFilename(self, absPath=False):
        if absPath:
            return os.path.abspath(self.FileName)
        else:
            return self.FileName
    def ImageNumber(self):
        return int(self.ImgNumber)
    def ImageShape(self, indexing='ij'):
        if (indexing=='xy'):
            return [self.ImgWidth, self.ImgHeight]
        else:
            return [self.ImgHeight, self.ImgWidth]
    def ImageHeight(self):
        return self.ImgHeight
    def ImageWidth(self):
        return self.ImgWidth
    def CountBytes(self):
        return np.uint64(self.hdrSize + self.ImgNumber * self.PxPerImg * self.PixelDepth + (self.ImgNumber - 1) * self.gapBytes)
    def GetShape(self):
        return np.asarray(self.Shape.copy())
    def HeaderSize(self):
        return self.hdrSize
    def GetFPS(self):
        return float(self.FPS)
    def GetPixelSize(self):
        return float(self.PixelSize)
    def DataType(self):
        return self.PixelDataType
    def DataFormat(self):
        return self.PixelFormat
    def ValidateROI(self, ROI, replaceNone=False):
        return ValidateROI(ROI, self.ImageShape(), replaceNone)
    def Validate_zRange(self, zRange, replaceNone=True):
        return Validate_zRange(zRange, self.ImgNumber, replaceNone)
    def IsStack(self):
        return False
    
    def _load_metadata(self, meta_data):
        """Reads metadata file
        it also reads the default configuration file
        in case of duplicates, information from MetaDataFile is used
        
        Parameters
        ----------
        meta_data : dict or filename. If dict, its keys should be directly the metadata parameters
                    (ex: hdr_len, gap_bytes, shape, px_format, endian, ...)
        """
        if (type(meta_data) in [dict, collections.OrderedDict]):
            logging.debug('Now loading MIfile metadata (dict with ' + str(len(meta_data)) + ' keys)')
        else:
            logging.debug('Now loading MIfile metadata (from filename: ' + str(meta_data) + ')')
        default_settings = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', 'config_MIfile.ini')
        if os.path.isfile(default_settings):
            default_settings = [default_settings]
        else:
            logging.info('MIfile._load_metadata() : Default MI settings file ({0}) not found. Loading metadata with hard-coded defaults'.format(default_settings))
            default_settings = []
        self.MetaData = cf.LoadMetadata(meta_data, SectionName='MIfile', DefaultFiles=default_settings)
        if 'MIfile' not in self.MetaData.GetSections():
            logging.warn('No MIfile section found in MIfile metadata (available sections: ' + str(self.MetaData.GetSections()) + ')')
        else:
            logging.debug('Now loading MIfile.MetaData from Config object. Available sections: ' + str(self.MetaData.GetSections()))
        self.MaxBufferSize = self.MetaData.Get('settings', 'max_buffer_size', 100000000, np.uint64)
        if (self.FileName is None):
            self.FileName = self.MetaData.Get('MIfile', 'filename', None)
            logging.info('MIfile.FileName updated to ' + str(self.FileName) + ' from metadata')
        self.hdrSize = self.MetaData.Get('MIfile', 'hdr_len', 0, np.uint64)
        self.gapBytes = self.MetaData.Get('MIfile', 'gap_bytes', 0, np.uint64)
        self.Shape = self.MetaData.Get('MIfile', 'shape', [0,0,0], np.uint64)
        self.ImgNumber = self.Shape[0]
        self.ImgHeight = self.Shape[1]
        self.ImgWidth = self.Shape[2]
        self.PxPerImg = np.uint64(self.ImgHeight * self.ImgWidth)
        self.PixelFormat = self.MetaData.Get('MIfile', 'px_format', 'B', str)
        self.Endianness = self.MetaData.Get('MIfile', 'endian', '', str)
        if self.Endianness not in ['', '>', '<']:
            logging.warn('MIfile endianness "' + str(self.Endianness) + '" not recognized: set to default')
            self.Endianness = ''
        self.PixelDepth = _data_depth[self.PixelFormat]
        self.PixelDataType = _data_types[self.PixelFormat]
        self.FPS = self.MetaData.Get('MIfile', 'fps', 1.0, float)
        self.PixelSize = self.MetaData.Get('MIfile', 'px_size', 1.0, float)
            

    def _get_offset(self, img_idx=0, row_idx=0, col_idx=0):
        """Get byte offset for a given pixel in a given image
        
        Parameters
        ----------
        img_idx : image index (first image is #0)
        row_idx : row index (top row is #0)
        col_idx : column index (left column is #0)
        """
        return self.hdrSize + (img_idx * self.PxPerImg + row_idx * self.ImgWidth + col_idx) * self.PixelDepth + img_idx * self.gapBytes
    
    def _read_pixels(self, px_num=1, seek_pos=None):
        """Read given number of contiguous pixels from MIfile
        
        Parameters
        ----------
        px_num :    number of pixels to read
        seek_pos : if None, start reading from current handle position
                   otherwise, offset position (in bytes) from beginning of file
        
        Returns
        -------
        1D numpy array of pixel values
        """
        if (seek_pos is not None):
            if self.debugMode:
                if (seek_pos >= 0 and seek_pos < self.CountBytes()):
                    seek_res = self.ReadFileHandle.seek(np.uint64(seek_pos))
                else:
                    raise IOError('Invalid seek posision: {0}'.format(seek_pos))
            else:
                self.ReadFileHandle.seek(seek_pos)
        px_num = int(px_num)
        bytes_to_read = px_num * self.PixelDepth
        fileContent = self.ReadFileHandle.read(bytes_to_read)
        if len(fileContent) < bytes_to_read:
            raise IOError('MI file read error: EOF encountered when reading image stack starting from seek offset ' + str(seek_pos) +\
                          ': ' + str(len(fileContent)) + ' instead of ' + str(bytes_to_read) + ' bytes (' + str(px_num) +\
                          ' pixels) returned from file ' + str(self.ReadingFileName))
        # get data type from the depth in bytes
        struct_format = (self.Endianness + '%s' + self.PixelFormat) % px_num
        # unpack data structure in a tuple (than converted into 1D array) of numbers
        return np.asarray(struct.unpack(struct_format, fileContent))
    
    def _imgs_to_bytes(self, data, data_format, do_flatten=True):
        res = bytes()
        if do_flatten:
            data = data.flatten().astype(_data_types[data_format])
        else:
            data = data.astype(_data_types[data_format])
        res += struct.pack(('%s' + data_format) % len(data), *data)
        return res