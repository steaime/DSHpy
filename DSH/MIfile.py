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
    """Crops a MIfile and saves it in a different file
    
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
            return [0, 0, ImageShape[1], ImageShape[0]]            
        else:
            return None
    else:
        assert (ROI[0] >= 0 and ROI[0] < ImageShape[1]), 'Top left coordinate (' + str(ROI[0]) + ') must be in the range [0,' + str(ImageShape[1]-1) + ')'
        assert (ROI[1] >= 0 and ROI[1] < ImageShape[0]), 'Top left coordinate (' + str(ROI[1]) + ') must be in the range [0,' + str(ImageShape[0]-1) + ')'
        if (ROI[2] < 0):
            ROI[2] = ImageShape[1] - ROI[0]
        if (ROI[3] < 0):
            ROI[3] = ImageShape[0] - ROI[1]
        assert (ROI[2] + ROI[0] <= ImageShape[1]), 'ROI ' + str(ROI) + ' incompatible with image shape ' + str(ImageShape)
        assert (ROI[3] + ROI[1] <= ImageShape[0]), 'ROI ' + str(ROI) + ' incompatible with image shape ' + str(ImageShape)
        return ROI

def Validate_zRange(zRange, zSize, replaceNone=True):
    if zRange is None:
        if replaceNone:
            return [0, zSize, 1]
        else:
            return None
    if (zRange[1] < 0):
        zRange[1] = zSize
    if (len(zRange) < 3):
        zRange.append(1)
    return zRange
        
class MIfile():
    """ Class to read/write multi image file (MIfile) """
    
    def __init__(self, FileName, MetaData):
        """Initialize MIfile
        
        Parameters
        ----------
        FileName : filename of multi image file (full path, including folder)
                    it can be None: in this case Metadata will still be loaded
                    if the option 'filename' is found in the Metadata, Filename will be updated
        MetaData : string or dict. 
                    if string: filename of metadata file
                    if dict: dictionary with metadata. 
                             dict keys will become options of the 'MIfile' section of the metadata
        """
        self.FileName = FileName
        self.ReadFileHandle = None
        self.WriteFileHandle = None
        logging.debug('MIfile object created with filename ' + str(FileName))
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
        str_res += '\n| Pixel format  : ' + str(self.PixelFormat) + ' (' + str(self.PixelDepth) + ' bytes/px)'
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
    
    def GetImage(self, img_idx, cropROI=None):
        """Read single image from MIfile
        
        Parameters
        ----------
        img_idx : index of the image, 0-based. If -N, it will get the Nth last image
        cropROI : if None, full image is returned
                  otherwise, [topleftx (0-based), toplefty (0-based), width, height]
                  width and/or height can be -1 to signify till the end of the image
        """
        if (img_idx<0):
            img_idx += self.ImgNumber
        if (cropROI is None):
            return self.GetStack(start_idx=img_idx, imgs_num=1).reshape(self.ImgHeight, self.ImgWidth)
        else:
            cropROI = self.ValidateROI(cropROI)
            if (cropROI[0]==0 and cropROI[2]==self.ImgWidth):
                res_arr = self._read_pixels(px_num=cropROI[2]*cropROI[3], seek_pos=self._get_offset(img_idx=img_idx, row_idx=cropROI[1], col_idx=cropROI[0]))
                return res_arr.reshape(cropROI[3], cropROI[2])
            else:
                res = []
                for row_idx in range(cropROI[1],cropROI[1]+cropROI[3]):
                    res.append(self._read_pixels(px_num=cropROI[2], seek_pos=self._get_offset(img_idx=img_idx, row_idx=row_idx, col_idx=cropROI[0])))
                return np.asarray(res)
        
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
        if (imgs_num < 0):
            imgs_num = self.ImgNumber - start_idx
        res_arr = self._read_pixels(px_num=imgs_num * self.PxPerImg, seek_pos=self._get_offset(img_idx=start_idx))
        return res_arr.reshape(imgs_num, self.ImgHeight, self.ImgWidth)

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
        if (sys.getsizeof(data_arr) > self.MaxBufferSize):
            if (sys.getsizeof(data_arr[0]) > self.MaxBufferSize):
                raise IOError('WriteMIfile is trying to write a very large array. Enhanced memory control is still under development')
            else:
                n_elem_xsec = len(data_arr[0].flatten())
                xsec_per_buffer = max(1, self.MaxBufferSize//n_elem_xsec)
                for i in range(0, len(data_arr), xsec_per_buffer):
                    self.WriteFileHandle.write(self._imgs_to_bytes(data_arr[i:min(i+xsec_per_buffer, len(data_arr))], self.PixelFormat, do_flatten=True))
        else:
            self.WriteFileHandle.write(self._imgs_to_bytes(data_arr, self.PixelFormat, do_flatten=True))
        if isinstance(data_arr, np.ndarray):
            logging.debug('ndarray of shape ' + str(data_arr.shape) + ' successfully written to MIfile ' + str(self.FileName))
        else:
            logging.debug('non-ndarray of size ' + str(sys.getsizeof(data_arr)) + ' successfully written to MIfile ' + str(self.FileName))
        if (closeAfter):
            self.Close()
            logging.debug('MIfile ' + str(self.FileName) + ' closing after writing')
        else:
            self.WriteFileHandle.flush()
            logging.debug('MIfile ' + str(self.FileName) + ' flushing without closing after writing')

    def Close(self, read=True, write=True):
        if (read and self.WriteFileHandle is not None):
            self.WriteFileHandle.close()
            self.WriteFileHandle = None
        if (write and self.ReadFileHandle is not None):
            self.ReadFileHandle.close()
            self.ReadFileHandle = None
    
    def GetMetadata(self):
        """Returns dictionary with metadata
        """
        return self.MetaData.ToDict(section='MIfile')
    
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
    def GetFilename(self):
        return self.FileName
    def ImageNumber(self):
        return int(self.ImgNumber)
    def ImageShape(self):
        return [int(self.ImgHeight), int(self.ImgWidth)]
    def ImageHeight(self):
        return int(self.ImgHeight)
    def ImageWidth(self):
        return int(self.ImgWidth)
    def GetShape(self):
        return np.asarray(self.Shape.copy())
    def HeaderSize(self):
        return int(self.hdrSize)
    def GetFPS(self):
        return float(self.FPS)
    def GetPixelSize(self):
        return float(self.PixelSize)
    def DataType(self):
        return self.PixelDataType
    def DataFormat(self):
        return self.PixelFormat
    def ValidateROI(self, ROI):
        return ValidateROI(ROI, self.ImageShape())
    def Validate_zRange(self, zRange):
        return Validate_zRange(zRange, self.ImgNumber)
    
    def _load_metadata(self, MetaData):
        """Reads metadata file
        it also reads the default configuration file
        in case of duplicates, information from MetaDataFile is used
        
        Parameters
        ----------
        MetaData : dict or filename
        """
        default_settings = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_MIfile.ini')
        if (type(MetaData) in [dict, collections.OrderedDict]):
            self.MetaData = cf.Config(None, defaultConfigFiles=[default_settings])
            self.MetaData.Import(MetaData, section_name='MIfile')
        else:
            self.MetaData = cf.Config(MetaData, defaultConfigFiles=[default_settings])
        self.MaxBufferSize = self.MetaData.Get('settings', 'max_buffer_size', 100000000, int)
        if (self.FileName is None):
            self.FileName = self.MetaData.Get('MIfile', 'filename', None)
        self.hdrSize = self.MetaData.Get('MIfile', 'hdr_len', 0, int)
        self.gapBytes = self.MetaData.Get('MIfile', 'gap_bytes', 0, int)
        self.Shape = self.MetaData.Get('MIfile', 'shape', [0,0,0], int)
        self.ImgNumber = self.Shape[0]
        self.ImgHeight = self.Shape[1]
        self.ImgWidth = self.Shape[2]
        self.PxPerImg = self.ImgHeight * self.ImgWidth
        self.PixelFormat = self.MetaData.Get('MIfile', 'px_format', 'B', str)
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
            self.ReadFileHandle.seek(seek_pos)
        bytes_to_read = px_num * self.PixelDepth
        fileContent = self.ReadFileHandle.read(bytes_to_read)
        if len(fileContent) < bytes_to_read:
            raise IOError('MI file read error: EOF encountered when reading image stack starting from seek offset ' + str(seek_pos) +\
                          ': ' + str(len(fileContent)) + ' instead of ' + str(bytes_to_read) + ' bytes (' + str(px_num) +\
                          ' pixels) returned from file ' + str(self.ReadingFileName))
        # get data type from the depth in bytes
        struct_format = ('%s' + self.PixelFormat) % px_num
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