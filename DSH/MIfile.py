import os
import numpy as np
import sys
import struct
from DSH import Config as cf

_data_depth = {'b':1, 'B':1, '?':1, 'h':2, 'H':2, 'i':4, 'I':4, 'f':4, 'd':8}
_data_types = {'b':np.int8, 'B':np.uint8, '?':bool, 'h':np.int16, 'H':np.uint16, 'i':np.int32, 'I':np.uint32, 'f':np.float32, 'd':np.float64}

class MIfile():
    """ Class to read/write multi image file (MIfile) """
    
    def __init__(self, FileName, MetaData):
        """Initialize MIfile
        
        Parameters
        ----------
        FileName : filename of multi image file (full path, including folder)
        MetaData : string or dict. 
                    if string: filename of metadata file
                    if dict: dictionary with metadata
        """
        self.FileName = FileName
        self.ReadFileHandle = None
        self.WriteFileHandle = None
        self._load_metadata(MetaData)
    
    def __repr__(self):
        return '<MIfile: %s+%sx%sx%sx%s bytes>' % (self.hdrSize, self.ImgNumber, self.ImgHeight, self.ImgWidth, self.PixelDepth)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| MIfile class: |'
        str_res += '\n|---------------+------------'
        str_res += '\n| Filename      : ' + str(self.FileName)
        str_res += '\n| Header        : ' + str(self.hdrSize) + 'bytes'
        str_res += '\n| Shape         : ' + str(self.Shape) + 'px'
        str_res += '\n| Pixel format  : ' + str(self.PixelFormat) + ' (' + str(self.PixelDepth) + ' bytes/px)'
        str_res += '\n|----------------------------'
        return str_res

    def __del__(self):
        self.Close()

    def OpenForReading(self, fName=None):
        if (self.ReadFileHandle is None):
            if (fName is None):
                fName = self.FileName
            self.ReadFileHandle = open(fName, 'rb')
        
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
        if zRange==None:
            zRange = [0, self.ImgNumber, 1]
        if (zRange[2] == 1 and cropROI is None):
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
        """Read image from MIfile
        
        Parameters
        ----------
        img_idx : index of the image, 0-based
        cropROI : if None, full image is returned
                  otherwise, [topleftx (0-based), toplefty (0-based), width, height]
                  width and/or height can be -1 to signify till the end of the image
        """
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
        exp_meta = self.GetMetadata()
        exp_meta['hdr_len'] = 0
        exp_meta['shape'] = list(mi_chunk.shape)
        exp_config = cf.Config()
        exp_config.Import(exp_meta, section_name='MIfile')
        exp_config.Export(metadata_filename)
        self.WriteData(mi_chunk)
    
    def OpenForWriting(self, fName=None, WriteHeader=None):
        """Open image for writing
        
        Parameters
        ----------
        fName : filename. If None, self.FileName will be used
        WriteHeader : list of dictionnaries each one with two entries: format and value
            if None, no header will be written (obsolete, for backward compatibility)
        """
        if (self.WriteFileHandle is None):
            if (fName is None):
                fName = self.FileName
            self.WriteFileHandle = open(fName, 'wb')
        if (WriteHeader is not None):
            buf = bytes()
            for elem_hdr in WriteHeader:
                buf += struct.pack(elem_hdr['format'], elem_hdr['value'])
            self.WriteFileHandle.write(buf)
    
    def WriteData(self, data_arr, closeAfter=True):
        self.OpenForWriting()        
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
        if (closeAfter):
            self.Close()

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
    def Shape(self):
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
        """Validates a Region Of Interest (ROI)
        
        Parameters
        ----------
        ROI : [topleftx (0-based), toplefty (0-based), width, height]
                  width and/or height can be -1 to signify till the end of the image
        """
        if (ROI is None):
            return None
        else:
            if (ROI[0] < 0 or ROI[0] >= self.ImgWidth):
                raise ValueError('Top left coordinate (' + str(ROI[0]) + ') must be in the range [0,' + str(self.ImgWidth-1) + ')')
            if (ROI[1] < 0 or ROI[1] >= self.ImgHeight):
                raise ValueError('Top left coordinate (' + str(ROI[1]) + ') must be in the range [0,' + str(self.ImgHeight-1) + ')')
            if (ROI[2] < 0):
                ROI[2] = self.ImgWidth - ROI[0]
            elif (ROI[2] + ROI[0] > self.ImgWidth):
                raise ValueError('ROI ' + str(ROI) + ' incompatible with image shape ' + str([self.ImgWidth, self.ImgHeight]))
            if (ROI[3] < 0):
                ROI[3] = self.ImgHeight - ROI[1]
            elif (ROI[3] + ROI[1] > self.ImgHeight):
                raise ValueError('ROI ' + str(ROI) + ' incompatible with image shape ' + str([self.ImgWidth, self.ImgHeight]))
            return ROI
    
    def _load_metadata(self, MetaData):
        """Reads metadata file
        it also reads the default configuration file
        in case of duplicates, information from MetaDataFile is used
        """
        default_settings = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_MIfile.ini')
        if (type(MetaData) is dict):
            self.MetaData = cf.Config(None, defaultConfigFiles=[default_settings])
            self.MetaData.Import(MetaData, section_name='MIfile')
        else:
            self.MetaData = cf.Config(MetaData, defaultConfigFiles=[default_settings])
        self.MaxBufferSize = self.MetaData.Get('settings', 'max_buffer_size', 100000000, int)
        if (self.FileName is None):
            self.FileName = self.MetaData.Get('MIfile', 'filename', None)
        self.hdrSize = self.MetaData.Get('MIfile', 'hdr_len', 0, int)
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
        return self.hdrSize + (img_idx * self.PxPerImg + row_idx * self.ImgWidth + col_idx) * self.PixelDepth
    
    def _read_pixels(self, px_num=1, seek_pos=None):
        """Read given number of contiguous pixels from MIfile
        
        Parameters
        ----------
        px_num : number of pixels to read
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
                          ': ' + str(len(fileContent)) + ' instead of ' + str(bytes_to_read) + ' bytes (' + str(px_num) + ' pixels) returned')
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