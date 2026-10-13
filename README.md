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

Galaxy images and datacubes default to `supersampling=1`. Use `supersampling=2` for four uniformly spaced rays
per output pixel, averaged in linear intensity before the RGB stretch. Use
`supersampling=1` for the original single-ray sampling. `npix` remains the final image
size and `grid_shape` is unchanged. Ray work grows approximately as the factor squared;
offsets are accumulated one at a time rather than retaining an enlarged spectral cube.
Optional optical-depth cubes contain mean ray depths, not effective attenuation depths.

Dust defaults to `dust_deposition='kernel'`: gas dust mass and thermal emission are
spread over neighboring cells with a compact cubic kernel using the snapshot's gas
smoothing lengths. The discrete cell-center weights conserve each selected particle's
mass and luminosity. Kernels too small to reach any cell center use the containing cell;
no arbitrary minimum smoothing radius is added. Stars also default to kernel deposition.
Stellar smoothing is controlled by `stellar_deposition='kernel', stellar_ngb=32`.
Each star's cubic-kernel support radius is its distance to the 32nd other real star;
wind particles are excluded. Neighbours come from the full snapshot, including outside
the image box, while only selected stars deposit luminosity. Fewer neighbours use the
furthest available star; an isolated star deposits directly. Kernel weights conserve
luminosity at each wavelength and renormalize at box boundaries. Radii are computed
once per datacube call and reused across wavelength batches. This can blur compact
clusters, increase occupied grid cells and add preparation time; direct stellar
deposition is available with `stellar_deposition='direct'`.
Particle centers select the box and central exclusion; kernels clipped by the box are
renormalized inside it. Use `dust_deposition='direct'` to reproduce the former behavior.
Kernel support expands compact occupancy and can require more memory and preparation
time. One-call `galaxy_image` and `galaxy_datacube` now default to
`wavelength_batch_size='auto'`: when dense full-spectrum coefficients exceed
`max_grid_bytes`, they prepare, trace and release wavelength batches, assembling only
the final image-sized spectra. A `512**3` grid with the default 4 GiB coefficient budget
uses at most three wavelengths plus dust density at a time (4 GiB), instead of keeping
all 128 image wavelengths in the grid. Snapshot caches, final cubes and processing
temporaries require additional RAM. This repeats particle processing and ray queries,
so it trades runtime for memory without changing deposition or wavelength sampling.
Set an integer batch size >= 2 to control it, or `None` to disable it.
`prepare_galaxy_datacube` still retains all wavelengths for reusable preparations.
An explicit `storage_dir` enables disk-backed coefficients and disables automatic
batching; `max_grid_bytes` continues to apply to arrays kept in RAM.
Rebuild prepared grids and datacubes after changing deposition: recoloring an old cube
does not update its dust distribution.

`Lbox` is in kpc when supplied without units. This uses the approximate visible RGB
filters and the notebook's shared asinh stretch; it is not a calibrated true-color image.
Choose `bands=('SDSS.i', 'SDSS.r', 'SDSS.g')` or Johnson I/V/B for false color.
`component` selects `effective` (default), `absorption`, or `pure_emission`.
`percentile` and `stretch` control the display, while additional keywords go to
`galaxy_datacube`. The image helper chooses 21 evenly spaced samples across each selected filter table
plus its peak, then traces their sorted union. RGB.R/G/B produce 24 wavelengths over
380–780 nm. This is a continuum sampling heuristic; pass explicit `lam` for finer
sampling or spectral features. `wavelength_range` overrides filter sampling; specifying
`n_wavelength` or `wavelength_spacing` alone creates a regular grid over the combined
filter range. Optical-depth cubes are omitted by default. Standard filters download through the resource cache on first use.

For an existing calculation, use `galaxy.galaxy_image(datacube=galaxy_rt)` to change
filters or stretch without tracing again. The result contains the RGB array, physical
band images, extent, figure, axes and image artist; it does not retain the spectral cube.
Use `plot=False` to return the RGB and band images without creating a figure (the
`figure`, `ax`, and `image` entries are then `None`). Use `plot='galaxy_rgb.png'`
to save without showing; the saved figure is closed, including a supplied `ax`'s figure.
`plot=True` is the default and keeps the figure open for notebook display.
Pass `ax=` to draw into an existing subplot. When `npix` is supplied without
`grid_shape`, the grid uses `npix` cells along each axis; an explicit `grid_shape` wins.
The default `inner_radius` is 10% of a pixel width (`0.1 * Lbox / npix`);
pass `inner_radius=0` to disable the central particle cutoff.

### Quick galaxy images

For a quick centered galaxy render, run:

```bash
py tools/galaxy_image.py /path/to/snapshot_334.hdf5 --box-size 20 --npix 1920 --output galaxy_334
```

This loads the snapshot, resets transformations and centers on particle type 3.
`--center-type` changes the centering target; `--faceon-radius-kpc` optionally rotates it.
The default grid is `min(npix, 512)` per axis. With the default 4 GiB coefficient budget,
spectral chunks are capped at 24 at a 128-cubed grid and 3 at 512-cubed; automatic
wavelength batching also bounds live coefficients. Use `--grid-shape`,
`--frequency-chunk-size` and `--max-grid-gib` to override these settings.
The tool uses filter-derived RGB wavelengths unless `--wavelengths-nm` is supplied.
The CLI retains direct stellar deposition by default; use `--stellar-deposition kernel --stellar-ngb 32` for stellar smoothing.

Outputs are `galaxy_334.png` (exactly `npix` by `npix`, without axes or margins) and
`galaxy_334.npy` (unquantized display RGB, shape `(npix, npix, 3)`, values in `[0, 1]`).
The array retains the renderer's lower-origin convention; display it with `origin='lower'`.
It contains stretched RGB, not physical spectral intensities. Existing output files
are refused; choose a new prefix. Run with `--help` for all options.

The script saves restart files in `<output>.restart` by default. After a failure, rerun
the same command with `--resume`; completed wavelength batches are loaded instead of
recomputed. `--restart-dir` selects another directory and `--wavelength-batch-size`
sets the checkpoint interval (default 3 wavelengths, capped by the coefficient budget).
The current unfinished batch must restart; particle loading, centering and preparation
for unfinished batches are repeated. Final RGB is checkpointed before PNG/NPY export,
so an export failure can resume without rendering. Files are published only after a
successful write, and partial temporary files are ignored. On failure, completed checkpoints and outputs are retained. After both outputs are
successfully exported, restart files are removed by default; use `--keep-restart`
to retain them. Existing output files are never replaced. Settings, snapshot size/modification time and
rendering source must match; use a new prefix after changing them. Resource tables
must also remain unchanged. A process lock prevents simultaneous writers. Restart
storage holds image-sized spectra for all three opacity cases, plus final RGB; allow
disk space accordingly. It does not save full 3D coefficient grids.

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
