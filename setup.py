from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="jae",
    version="0.0.1",
    author='Jaeden Bardati',
    description='A package with a bunch of code I regularly use. Primarily focused on analysis of MHD simulation data.',
    long_description=long_description,
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
                      'numpy',
                      'matplotlib',
                      'scipy', 
                      'astropy', 
                      'h5py'
                     ]
)
