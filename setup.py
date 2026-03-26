from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="jaba",
    version="0.0.1",
    author='Jaeden Bardati',
    description='A package with a bunch of code I regularly use. Primarily focused on analysis of MHD simulation data.',
    long_description=long_description,
    package_dir={'jaba': 'jabapy'},
    packages=['jaba'],
    python_requires='>=3.7',
    install_requires=[
                      'numpy', ##==1.21.6
                      'matplotlib', #==3.5.3
                      'scipy', ##==1.7.3 
                      'astropy', #==4.3.1
                      'h5py', #==3.1.0
                      'pandas', #==1.1.5
                      'scikit-learn', #==1.0.2
                      'yt', #==4.1.4
                      'pynbody', #==1.3.0
                      'photutils', # ==1.3.0
                      'opencv-python',
                      'notebook',
                     ]
)
