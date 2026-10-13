## Summary

Some commonly used code, mostly for [GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) or [SKIRT](https://skirt.ugent.be/root/_home.html) simulations and analysis. This is primarily composed of my python 3 `jabapy` package, but I also include some useful scripts. The goal is for it to be *portable, expandable, and easy to use*.

All you need is bash (e.g., some unix-based system like mac or linux). The `install.sh` file will set up everything else you may need, including python, conda, simulation codes, etc. Note that I have only tested it on the computers and clusters that I have access to, so use with caution. This comes with no guarantee that it will work as intended or even work at all.

## Install

1) Run `git clone https://github.com/JaedenBardati/jaba.git` or `git clone git@github.com:JaedenBardati/jaba.git` in a terminal to clone the repo. 
2) Run `bash jaba/install.sh` to setup the package.
3) Run `source ~/.bashrc` to get access to the jaba commands and variables.

That's it. 

If you run into an issue during the `install.sh` process, just update the file accordingly to add support for your specific system at the marked location.

### Optional Submodules

There are a number of optional submodules, including:

- GIZ: A script to handle running/managing [GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) simulations. Call it with `giz` in the terminal.
- (TO DO) SKI: A script to handle running/managing [SKIRT](https://skirt.ugent.be/root/_home.html) simulations. Call it with `ski` in the terminal.
- (TO DO) CLD: A script to handle running/managing [CLOUDY](https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home) simulations. Call it with `cld` in the terminal.

You will have the option to install these when you run `install.sh`.

### Updating and Uninstalling

Once installed, just run `jaba-uninstall` from anywhere to uninstall. You can also call `jaba-update` to update (uninstall, pull from github, and reinstall) or `jaba-reinstall` to reinstall without updating.

## Common Usage

### Analysis

Once you have run the install script, you can use the following commands in your terminal.

- `py`: launch the newly created python environment (incl. jaba) interactively on the local machine.
- `py file.py`: run your python script (replace `file.py` to your script's file name) on the local machine.
- `pyq file.py`: queue your python script to run as a job on a cluster.
- `jupy`: launch jupyter notebook in the jaba environment (currently only has support for local machines).
- `qcs snapshot.hdf5`: dump diagnostic plots for quick simulation analysis ("quick check simulation").
- `qcsq snapshot.hdf5`: same as above, but queued as a job.

Use `import jaba` in the python environment to access `jabapy`. For the `jabapy` python package code, see the `examples` directory for jupyter notebook example usages. 

### Multipanel figures

Use `jaba.visual.Figure` to arrange plots with the same settings as the standalone
plotting helpers. Indices are zero-based `(row, column)`, so `[0, 0]` is the top-left
panel. Repeated calls on an explicit panel overlay data there.

```python
from jaba import visual as jv

fig = jv.Figure(2, 2, figsize=(8, 6), xlabel='x')
fig[0, 0].plot([0, 1, 2], [0, 1, 4], title='Lines', label='first')
fig[0, 0].plot([0, 1, 2], [0, 2, 3], label='second')
fig[0, 1].scatter([0, 1, 2], [2, 0, 1], title='Points')
fig[1, 0].hist([0, 0, 1, 2, 2, 2], bins=3, title='Histogram')
fig[1, 1].imshow([[0, 1], [2, 3]], title='Image', show_colorbar=True)
fig.draw(show=True)
fig.draw(out='multipanel.png')
```

Without an explicit panel, `fig.plot(...)` and the other plotting methods choose the
first panel without plotted data, in reading order. If all panels contain data, select
one explicitly. `fig[:, :].plot(...)` plots on every selected panel. Create a spanning
panel with `fig.add_axis(0, slice(None))` before filling those cells; overlapping panels
are rejected. A span is plotted only once even when several selected cells refer to it.

Constructor settings provide defaults; per-panel calls override them. Access the raw
Matplotlib axes with `fig[0, 0].ax`, or wrap an existing shared subplot grid with
`jv.Figure(fig=existing_figure)`. `draw()` retains panels by default; `clear()` and
`close()` act on this figure explicitly. Plotting methods return their usual artists,
or a list of results for a selection containing multiple panels.

### Galaxy images

Render a snapshot with the continuum, bandpass and plotting steps in one call:

```python
from jaba.apps import galaxy

image = galaxy.galaxy_image(snap, Lbox=10, grid_shape=128, npix=512, workers=-1, log=True)
image['figure'].savefig('galaxy_rgb.png')
```

Galaxy images and datacubes default to `supersampling=2`: four uniformly spaced rays
per output pixel, averaged in linear intensity before the RGB stretch. Use
`supersampling=1` for the original single-ray sampling. `npix` remains the final image
size and `grid_shape` is unchanged. Ray work grows approximately as the factor squared;
offsets are accumulated one at a time rather than retaining an enlarged spectral cube.
Optional optical-depth cubes contain mean ray depths, not effective attenuation depths.

Dust defaults to `dust_deposition='kernel'`: gas dust mass and thermal emission are
spread over neighboring cells with a compact cubic kernel using the snapshot's gas
smoothing lengths. The discrete cell-center weights conserve each selected particle's
mass and luminosity. Kernels too small to reach any cell center use the containing cell;
no arbitrary minimum smoothing radius is added. Stars retain direct deposition.
Particle centers select the box and central exclusion; kernels clipped by the box are
renormalized inside it. Use `dust_deposition='direct'` to reproduce the former behavior.
Kernel support expands compact occupancy and can require more memory and preparation
time; `max_grid_bytes` still applies and `storage_dir` enables disk-backed coefficients.
Rebuild prepared grids and datacubes after changing deposition: recoloring an old cube
does not update its dust distribution.

`Lbox` is in kpc when supplied without units. This uses the approximate visible RGB
filters and the notebook's shared asinh stretch; it is not a calibrated true-color image.
Choose `bands=('SDSS.i', 'SDSS.r', 'SDSS.g')` or Johnson I/V/B for false color.
`component` selects `effective` (default), `absorption`, or `pure_emission`.
`percentile` and `stretch` control the display, while additional keywords go to
`galaxy_datacube`. The image helper defaults to 128 linear wavelengths over
0.28–1.25 micron and omits optical-depth cubes; `lam` or `wavelength_range` can override
that coverage. Standard filters download through the resource cache on first use.

For an existing calculation, use `galaxy.galaxy_image(datacube=galaxy_rt)` to change
filters or stretch without tracing again. The result contains the RGB array, physical
band images, extent, figure, axes and image artist; it does not retain the spectral cube.
Use `plot=False` to return the RGB and band images without creating a figure (the
`figure`, `ax`, and `image` entries are then `None`). Use `plot='galaxy_rgb.png'`
to save without showing; the saved figure is closed, including a supplied `ax`'s figure.
`plot=True` is the default and keeps the figure open for notebook display.
Pass `ax=` to draw into an existing subplot. When `npix` is supplied without
`grid_shape`, the grid uses `npix` cells along each axis; an explicit `grid_shape` wins.
The default `inner_radius` is 5% of a pixel width (`0.05 * Lbox / npix`);
pass `inner_radius=0` to disable the central particle cutoff.

### Running Simulations

If you installed the relevant optional submodules, you can run the following command examples.

- Run `giz -N 4 -n 112 -T 2` to start a GIZMO simulation in the working directory that runs on 112 processes across 4 nodes, each with 2 threads.
- Run `giz -r` to restart a GIZMO simulation, or `giz -r2` to restart from an hdf5 snapshot.
- (TO DO) Run `ski -N 10 -n 70 -T 8` to run a SKIRT simulation in the working directory that runs on 70 processes across 10 nodes, each with 8 threads.

The rest you can likely gather from running commands with the `--help` argument or by reading the code.

### Other useful scripts

If you have allowed them to be installed in the installation process, you can run the following other scripts.

- Run `setup-ssh` to setup ssh keys for a new server or github push access, etc.
- Run `sxh servername` to launch a tmux terminal with panels that ssh/sftp into the server nickname requested and one local terminal panel.

You can `cat ~/.bashrc` and look the jaba tag for a list of commands you have available to you.
