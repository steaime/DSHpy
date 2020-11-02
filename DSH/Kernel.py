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
        if (kernel_specs is not None):
            self.Initialize(**kernel_specs)
            
    def __repr__(self):
        str_res = '<Kernel [' + str(self.Type) + ', ' + str(self.Dimensions) + 'D]'
        if self.Shape is not None:
            str_res += ', '
            for i in range(len(self.Shape)):
                if i>0:
                    str_res += 'x'
                str_res += str(self.Shape[i])
        str_res += '>'
        return str_res
    
    def __str__(self):
        return str(self.ToDict())
            
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
        self.Shape = sf.CheckIterableVariable(shape, n_dim)
        self.Dimensions = n_dim
        self.Center = np.true_divide(np.subtract(self.Shape, 1), 2)
        
        if (kernel_type=='Gauss'):
            self.params = self.GaussParams(sf.CheckIterableVariable(self.params['sigma'], n_dim))
        elif (kernel_type=='flat'):
            self.params = None
            
        self.Padding = padding
        if padding:
            self.convolveMode = 'same'
        else:
            self.convolveMode = 'valid'
        
        self.convolve_kwargs = convolve_kwargs
    
    def ToDict(self):
        res = {'shape'  : self.Shape,
               'type'   : self.Type,
               'n_dim'  : self.Dimensions,
               }
        if self.params is not None:
            res.update(self.params.ToDict())
        res.update({'padding': self.Padding,
                   'convolveMode': self.convolveMode,
                   'convolve_kwargs': self.convolve_kwargs})
        return res
    
    def ToMatrix(self):
        """Computes the convolution kernel for ROI computation
        """
        if (self.Type=='Flat'):
            kerND = np.ones(self.Shape)
        else:
            coords = []
            for i in range(self.Dimensions):
                coords.append(np.asarray(range(-self.Center[i], self.Center[i]+self.Shape[i])))
            grid = np.meshgrid(*coords)
            if (self.Type=='Gauss'):
                kerND = np.zeros_like(grid[0])
                for i in range(self.Dimensions):
                    kerND -= np.divide(np.square(grid[i]), 2*np.square(self.params.sigma[i]))
                kerND = np.exp(kerND)
            else:
                raise ValueError('Kernel type "' + str(self.Type) + '" not supported')
        # Whatever kernel we are using, let's normalize so that weights has unitary integral
        kerND = np.true_divide(kerND, np.sum(kerND))
        return kerND

        
