## Summary

Some commonly used code, mostly for [GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) or [SKIRT](https://skirt.ugent.be/root/_home.html) simulations and analysis. This is primarily composed of my python 3 `jabapy` package, but I also include some useful scripts. The goal is for it to be *portable, expandable, and easy to use*.

All you need is bash (so some unix-based system like mac or linux). The `install.sh` file will set up everything else you may need, including python, conda, simulation codes, etc. I have only tested it on the computers and clusters that I have access to, so use with caution.

## Install

1) Run `git clone https://github.com/JaedenBardati/jaba.git` or `git clone git@github.com:JaedenBardati/jaba.git` in a terminal to clone the repo. 
2) Run `bash jaba/install.sh` to setup the package.
3) Run `source ~/.bashrc` to get access to the jaba commands and variables.

That's it. If you run into a problem on your system, just update the `install.sh` accordingly, but try to maintain functionality on existing systems if you want to push your changes to the main branch.

### Optional Submodules

There are a number of optional submodules, including:

- GIZ: A script to handle running/managing [GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) simulations. Call it with `giz` in the terminal.
- (TO DO) SKI: A script to handle running/managing [SKIRT](https://skirt.ugent.be/root/_home.html) simulations. Call it with `ski` in the terminal.

You will have the option to install these when you run `install.sh`.

### Updating and Uninstalling

Once installed, just run `jaba-uninstall` from anywhere to uninstall. You can also call `jaba-update` to update (uninstall, pull from github, and reinstall) or `jaba-reinstall` to reinstall without updating.

## Common Usage

### Analysis

Once you have 

- `py`: launch the newly created python environment (incl. jaba) interactively on the local machine.
- `py file.py`: run your python script (replace `file.py` to your script's file name) on the local machine.
- `pyq file.py`: queue your python script to run as a job on a cluster.
- `jupy`: launch jupyter notebook in the jaba environment (currently only has support for local machines).
- `qcs snapshot.hdf5`: dump diagnostic plots for quick simulation analysis ("quick check simulation")
- `qcsq snapshot.hdf5`: same as above, but queued as a job. 

See the `jabapy` python package code and `examples` directory for details on that. Use `import jaba` in your python files to access `jabapy`.

### Running Simulations

- Run `giz -N 36 -n 1008 -T 2` to start a GIZMO simulation in the working directory that runs on 1008 processes across 36 nodes, each with 2 threads.
- Run `giz -r` to restart a GIZMO simulation.
- (TO DO) Run `ski -N 10 -n 70 -T 8` to run a SKIRT simulation in the working directory that runs on 70 processes across 10 nodes, each with 8 threads.

The rest you can likely gather from running commands with the `--help` argument or by reading the code.
