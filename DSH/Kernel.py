import numpy as np
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
            self.Initialize(**kernel_specs)
            
    def __repr__(self):
        str_res = '<Kernel [' + str(self.Type) + ', ' + str(self.Dimensions) + 'D]'
        if self.Shape is not None:
            str_res += ', ' + 'x'.join(self.Shape)
        str_res += '>'
        return str_res
    
    def __str__(self):
        str_res = '<Kernel [' + str(self.Type) + ', ' + str(self.Dimensions) + 'D]'
        if self.Shape is not None:
            str_res += ', ' + 'x'.join(self.Shape)
            if self.Type=='Gauss':
                str_res += ', s=(' + ';'.join(self.Shape) + ')'
        str_res += '>'
        return str_res
    
    def Initialize(self, shape, kernel_type, params={}, n_dim=2, padding=False, convolve_kwargs={}):
        """ Initialize kernel

        Parameters
        ----------
        shape       : tuple of integers, shape of the output kernel matrix.
                      entries must be odd integers
                      number of entries must match dimensions (n_dim)
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
            
        self.Padding = padding
        if padding:
            self.convolveMode = 'same'
            if 'boundary' not in convolve_kwargs:
                convolve_kwargs['boundary'] = 'fill'
            if 'fillvalue' not in convolve_kwargs:
                convolve_kwargs['fillvalue'] = 0
        else:
            self.convolveMode = 'valid'
        
        self.convolve_kwargs = convolve_kwargs
    
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
            grid = np.meshgrid(*coords)
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

        
