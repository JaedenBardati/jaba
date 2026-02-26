## Summary

This is a repository where I keep a lot my often used code, mostly for GIZMO or SKIRT simulations and analysis. Mostly this is python (see the python `jaba` submodule), but I also include some useful bash scripts. The goal is for it to be portable, expandable, and easy to use.

## Requirements

- Python 3
- Various modules require different python packages, but generally you commonly need `numpy`, `matplotlib`, `scipy`, `astropy`, and `h5py`. See `setup.py` for a full list of everything you could possibly need.

## Install

1) Clone the repo with `git clone git@github.com:JaedenBardati/jaba.git`
2) Set up the package by running `jaba/setup.sh`
3) Now, `import jaba` as needed.

Note that when using a new system, you may need to adjust the setup bash file for jaba and some of the submodules - particularly, `giz` if you plan on using it to manage running GIZMO simulations.
