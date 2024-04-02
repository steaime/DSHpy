import numpy as np
from scipy import signal
from DSH import SharedFunctions as sf

class Kernel():
    """ Class defining convolution kernels """
    
    class GaussParams:
        """ Parameters specific to Gaussian kernels
        """
        def __init__(self, sigma):
            """
            Parameters
            ----------
            sigma : tuple, std deviation of the Gaussian in each direction
                    (dimension must match with the kernel dimension)
            """
            self.sigma = sigma
            
        def ToDict(self):
            return {'sigma' : self.sigma}
    
    def __init__(self, kernel_specs=None):
        """ Initialize kernel

        Parameters
        ----------
        kernel_specs : dict with kernel specs
        """
        
        self.Type = None
        self.Shape = None
        self.Dimensions = 0
        self.KernelParams = None
        self.Padding = False
        self.convolveMode = 'valid'
        self.convolve_kwargs = {}
        if (kernel_specs is not None):
            self.FromDict(kernel_specs)
            
    def __repr__(self):
        str_res = '<Kernel [' + str(self.Dimensions) + 'D ' + str(self.Type) + ']'
        if self.Shape is not None:
            str_res += ', ' + 'x'.join(map(str, self.Shape))
        if self.Padding:
            str_res += ', pad'
        else:
            str_res += ', no pad'
        str_res += '>'
        return str_res
    
    def __str__(self):
        str_res = '<Kernel [' + str(self.Dimensions) + 'D ' + str(self.Type) + ']'
        if self.Shape is not None:
            str_res += ', ' + 'x'.join(map(str, self.Shape))
            if self.Type=='Gauss':
                str_res += ', s=(' + ';'.join(map(str, self.KernelParams.sigma)) + ')'
        if self.Padding:
            str_res += ', pad'
        else:
            str_res += ', no pad'
        str_res += '>'
        return str_res
    
    def FromDict(self, dict_source):
        """ Initialize kernel from dictionnary        

        Parameters
        ----------
        dict_source : dict. Must have the following keys: ['shape', 'type', 'padding']
        """
        assert ('shape' in dict_source and 'type' in dict_source and 'padding' in dict_source), 'Source dict missing required keys'
        
        _shape   = sf.StrParse(dict_source['shape'], int)
        _type    = str(dict_source['type'])
        _padding = sf.StrParse(dict_source['padding'], bool)
        if (_type=='Gauss'):
            assert 'sigma' in dict_source, 'Gaussian kernel dict missing sigma key'
            _params = {'sigma':sf.StrParse(dict_source['sigma'], float)}
        else:
            _params = {}
        if 'n_dim' in dict_source:
            n_dim = int(dict_source['n_dim'])
        else:
            n_dim = len(_shape)
        _conv_kwargs = {}
        if 'convolve_kwargs' in dict_source:
            _conv_kwargs.update(sf.StrParse(dict_source['convolve_kwargs']))
        
        self.Initialize(_shape, _type, _params, n_dim, _padding, _conv_kwargs)
    
    def Initialize(self, shape, kernel_type, params={}, n_dim=2, padding=False, convolve_kwargs={}):
        """ Initialize kernel

        Parameters
        ----------
        shape       : integer or tuple of integers, shape of the output kernel matrix.
                      entries must be odd integers
                      if tuple, number of entries must match dimensions (n_dim)
                      if single integer, a tuple of size n_dim with repeated value will be generated
        kernel_type : string, type of kernel. Supported types: ['Gauss', 'flat']
        params      : dict with type-dependent parameters 
        n_dim       : dimensions. Currently only n_dim=2 is supported
        padding     : boolean. If True convolution will be performed using 'same' mode
                      otherwise, only 'valid' convolutions will be accepted (smaller output)
        convolve_kwargs : dict with additional parameters to be passed to scipy.signal.convolve2d
        """
        
        assert kernel_type in ['Gauss', 'flat'], 'Kernel type "' + str() + '" not supported'
        
        self.Type = kernel_type
        self.Shape = sf.CheckIterableVariable(shape, n_dim, cast_type=int)
        self.Dimensions = n_dim
        self.Center = np.true_divide(np.subtract(self.Shape, 1), 2)
        
        if (kernel_type=='Gauss'):
            self.KernelParams = self.GaussParams(sf.CheckIterableVariable(params['sigma'], n_dim))
        elif (kernel_type=='flat'):
            self.KernelParams = None
        
        self.SetPadding(padding)
        
        if 'boundary' not in convolve_kwargs:
            convolve_kwargs['boundary'] = 'fill'
        if 'fillvalue' not in convolve_kwargs:
            convolve_kwargs['fillvalue'] = 0
        self.convolve_kwargs = convolve_kwargs
        
    def SetPadding(self, new_padding):
        self.Padding = new_padding
        if new_padding:
            self.convolveMode = 'same'
        else:
            self.convolveMode = 'valid'
    
    def ToDict(self):
        res = {'shape'  : self.Shape,
               'type'   : self.Type,
               'n_dim'  : self.Dimensions,
               }
        if self.KernelParams is not None:
            res.update(self.KernelParams.ToDict())
        res.update({'padding': self.Padding,
                   'convolveMode': self.convolveMode,
                   'convolve_kwargs': self.convolve_kwargs})
        return res
    
    def ToMatrix(self):
        """Computes the convolution kernel for ROI computation
        """
        
        assert self.Type is not None, 'Kernel not initialized'
        
        if (self.Type=='Flat'):
            kerND = np.ones(self.Shape)
        else:
            coords = []
            for i in range(self.Dimensions):
                coords.append(np.subtract(np.asarray(range(self.Shape[i])), self.Center[i]))
            grid = np.meshgrid(*coords, indexing='ij')
            if (self.Type=='Gauss'):
                kerND = np.zeros_like(grid[0])
                for i in range(self.Dimensions):
                    kerND -= np.divide(np.square(grid[i]), 2*np.square(self.KernelParams.sigma[i]))
                kerND = np.exp(kerND)
            else:
                raise ValueError('Kernel type "' + str(self.Type) + '" not supported')
        # Whatever kernel we are using, let's normalize so that weights has unitary integral
        kerND = np.true_divide(kerND, np.sum(kerND))
        return kerND
            
    def ConvolveImage(self, imgInput):
        """Convolves an image with the kernel
        
        Parameters
        ----------
        imgInput : 2D array to be convolved

        Returns
        -------
        a 2D array with the convolution result
        """

        imgRes = signal.convolve2d(imgInput, self.ToMatrix(), mode=self.convolveMode, **self.convolve_kwargs)
        if (self.convolveMode=='same'):
            imgRes = np.true_divide(imgRes, signal.convolve2d(np.ones_like(imgInput), self.ToMatrix(), mode=self.convolveMode, **self.convolve_kwargs))
        return imgRes
    
    def Copy(self):
        return Kernel(kernel_specs=self.ToDict())