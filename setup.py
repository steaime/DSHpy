"""Setuptools setup script."""
from setuptools import setup

setup(name='DSH',
      version='0.0.1',
      description='Analyze DSH videos',
      url='https://github.com/steaime/DSHpy',
      author='Stefano Aime, Matteo Sabato',
      author_email='aime@seas.harvard.edu',
      license='GNU GPL',
      packages=['DSH'],
      install_requires=[
            'numpy',
            'scipy',
            'configparser'
      ],
      #test_suite='nose.collector',
      #tests_require=['nose'],
      zip_safe=False)