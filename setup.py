"""Setuptools setup script."""
from setuptools import setup

setup(name='DSH',
      version='0.6',
      description='Analyze speckle fields, compute correlations, derive motion maps',
      url='https://github.com/steaime/DSHpy',
      author='Stefano Aime',
      author_email='stefano.aime@espci.fr',
      license='GNU GPL',
      packages=['DSH'],
      install_requires=[
            'numpy',
            'scipy',
            'configparser'
      ], 
      #NOTE: other modules optionally used: emcee
      #      - emcee (VelMaps)
      #      - pandas (VelMaps)
      #      - astropy (PostProcFunctions)
      #test_suite='nose.collector',
      #tests_require=['nose'],
      zip_safe=False)