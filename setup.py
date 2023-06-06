"""Setuptools setup script."""
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='DSH',
      version='0.8',
      description='Analyze speckle fields, compute correlations, derive motion maps',
      long_description=readme(),
      url='https://github.com/steaime/DSHpy',
      author='Stefano Aime',
      author_email='stefano.aime@espci.fr',
      license='GNU GPL',
      packages=['DSH'],
      package_data={'src': ['config/*',]},
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
      include_package_data=True,
      zip_safe=False)