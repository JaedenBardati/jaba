## Summary

In this repository, I keep a lot my often used code, mostly for GIZMO or SKIRT simulations and analysis. This is primarily python (see the `jaba` submodule), but I also include some useful bash scripts. The goal is for it to be portable, expandable, and easy to use. 

## Requirements

- Python 3
- Various modules require different python packages, but generally you commonly need `numpy`, `matplotlib`, `scipy`, `astropy`, and `h5py`. See `setup.py` for a full list.

## Install

1) Clone the repo with `git clone git@github.com:JaedenBardati/jaba.git`
2) If you have a a unix system (mac or linux), run `bash jaba/setup.sh`, which will create a python environment for you for ease of use. Alternatively (or if you have a windows machine), just `pip install jaba` into your own environment.
3) Then, `import jaba` as you need. And if you ran `setup.sh`, you can call `py`, `pyq`, `qcs`, etc.

