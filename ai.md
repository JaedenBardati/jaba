# Project Context & AI Instructions

## Purpose

Jaba is a personal research toolkit for astrophysical simulation analysis, primarily GIZMO and SKIRT, with room for other simulation codes. Keep it portable, expandable, and easy to use. Reusable Python code lives in `jabapy/`; shell scripts handle environments, cluster jobs, and simulation management.

`setup.py` installs `jabapy/` under the package name `jaba`, so installed usage is `import jaba`. Existing repository tests also import `jabapy` directly. Preserve the installed API and existing import shortcuts (`jaba.units`, `jaba.constants`, and `jaba.visual`).

## Project map

| Path | Responsibility |
| --- | --- |
| `jabapy/snapshot.py` | Core HDF5/GIZMO snapshot loading, particle datasets, derived quantities, units, and convenience accessors. Start here for simulation data access. |
| `jabapy/physics/` | Reusable physics: radiation, spectra, opacity, and radiative transfer in `radiation.py`; dynamics in `dynamics.py`; cosmology in `cosmology.py`. |
| `jabapy/apps/` | Problem-specific workflows: broad-line-region/H-alpha calculations in `blr.py`, galaxy continuum preparation and ray tracing in `galaxy.py`, and initial-condition/snapshot transformations in `gizmo_ics.py`. |
| `jabapy/io/` | General file handling and HDF5/FITS readers and writers. |
| `jabapy/utils/` | Other general utilities: units/constants, coordinates, data structures, plotting, timing, argument parsing, and reloading. `grid.py` contains gridding, particle binning, interpolation, and numerical kernels; `resources.py` downloads and loads external resource tables. |
| `tests/` | Focused Python `unittest` tests for jabapy calculations. |
| `examples/` | Jupyter notebooks demonstrating package usage and Python basics. |
| `tools/` | Python command-line tools using jaba, such as `quickchecksim.py`. |
| `standalone/` | Python scripts that do not require jaba; they may still require third-party packages. |
| `scripts/` | Shell scripts including entry points for Python/Jupyter, cluster execution, SSH/sync helpers, and simulation wrappers such as `giz` and `cld`. Availability depends on installation. |
| `install.sh`, `setup.py` | System/environment setup and Python package metadata, respectively. See `README.md` for installation and command usage. |

## Where code belongs

- Keep `utils` and `io` general-purpose and independent of other jaba files. Do not import other jaba files from these modules.
- Keep `physics` focused on reusable physics, using shared utilities as needed; avoid dependencies on `snapshot` or `apps`.
- `snapshot.py` may use `physics`, `io`, and `utils`, but must not import `apps`.
- When refering to `snapshot.py`, avoid code-specific data references (e.g. snap[0, 'Coordinates']) and use the general references instead (e.g. snap.pos0).
- Put workflows combining snapshots, physics, and domain-specific assumptions in `apps`; this layer may use the rest of the package.
- Extend existing functions and modules before adding abstractions. Avoid circular imports, duplicated physics, and unrelated refactors.

## Working environment and verification

- The repository is normally located at `$JABA_LOCATION`, configured through `~/.bashrc` by the installer. Inspect the current checkout and relevant code before editing.
- To run Python, source `~/.bashrc`, call `activate_jaba_python_environment` (alias `jaba-activate`), and launch scripts with `py`. Call `deactivate_jaba_python_environment` (alias `jaba-deactivate`) when finished. In noninteractive Bash, enable alias expansion with `shopt -s expand_aliases` before using aliases.
- Run relevant existing tests for code changes. From the repository root, use `py -m unittest discover -s tests -p 'test_radiation.py'` for one module, or `py -m unittest discover -s tests` for the suite. Documentation-only edits need a content check, not simulation runs.
- Prefer existing jaba functions when dependency boundaries and performance allow. Otherwise, use the standard library or dependencies already declared in `setup.py`; ask before adding a new dependency.
- Prefer `jabapy/utils/visual.py` for supported plotting tasks. Extend it only for reusable functionality; keep one-off plotting in the calling script or notebook.
- Prefer small synthetic inputs and temporary outputs for verification. Check physical units, array shapes, numerical limits, and conservation where relevant; avoid expensive full-snapshot runs unless needed.
- Limit analysis terminal output to the first or last 20 lines. Print array shapes or summaries, never raw arrays.

## Coding conventions

- Generally match the surrounding style, formatting, names, and public interfaces. Prefer compact, readable code; do not add unnecessary functions or classes to simple analysis scripts.
- Group related operations without blank lines inside the block. Precede each block with a blank line and a single-line, lowercase comment describing its high-level intent. Avoid comments for single lines unless they are expensive or complex.
- Restrict multiline comments to file, class, or function headers; do not put Markdown in comments. Avoid excessively long comments.
- Preserve existing unit conventions. Functions named `*_cgs` generally operate on numerical CGS values: inspect the function contract and explicitly convert units at boundaries before stripping quantities.
- Generally, preserve array-axis conventions, particle ordering, and return shapes. Treat memory use and performance as part of correctness for large snapshots; retain vectorized/chunked operations and Numba-compatible kernels where used.
- Avoid explicit error handling in analysis code unless requested or testing. In structured code, handle errors only outside performance-critical paths.
- In places where there is signficant computational time (tens of seconds or more for a typical use case), add a `log` option that informs the user of what is taking place.

## Safety and scope

- Preserve existing uncommitted work. Change only what the task requires; do not revert, reformat, or clean up unrelated files.
- Never overwrite raw simulation data or cached execution states. Save derived results with a distinct suffix, or ask for a specific output path.
- Obtain explicit confirmation before deleting files, overwriting critical data, or making irreversible system changes.
- Ask before modifying Python environments or global configuration. Do not run installation, update, reinstall, or uninstall commands as incidental fixes.
- Keep generated data, downloaded resources, environment directories, and build/cache artifacts out of source edits.
- When editing notebooks, preserve unrelated cells, metadata, and saved outputs. Do not rerun expensive cells unless needed for the task.
- When moving or renaming modules, update imports, package exports, `setup.py`, tests, documentation, and affected notebooks together.
- Deliver complete, functional code without placeholders or truncated blocks. Finish tasks with a concise bulleted summary of exactly what changed, and state relevant verification or limitations.

## Git commits

- Commit each completed task after relevant tests pass. Stage only changes belonging to that feature; do not include unrelated pre-existing work. Report the commit hash and a brief summary to the user.
- During ongoing work, after more than five commits since the last push and a few hours of work, ask whether to push to GitHub and summarize the unpushed changes. Do not push without authorization.

## Communication

- Keep the user informed with any changes to `jaba`. Before introducing a new physical assumption or approximation that changes scientific results, briefly explain it and confirm it with the user. Existing documented choices do not need repeated confirmation unless the task changes their applicability.

## Typical workflow

Use this as a rough template for a path of thought when adding a computational time- or memory-intensive feature:

0) Understanding stage: Define what you need (prompt user for any specifics or approximations that you may need to make if it is not clear). Break down what you need to do in concrete steps.
1) Research stage: Look around for any relevant functions, classes or other code that can be re-used in your code. Do not re-invent the wheel.
2) Proto-typing stage: Make a working change to the code that does the desired behaviour that can serve as a baseline.
3) Testing stage: Test the code and iterate until desired behaviour is reached. Add an explicit testing suite in `tests` whenever it is a large enough or relevant enough piece of code to do so.
4) Efficiency stage: Determine whether it is computational time- and/or memory-limited and what is causing the majority of the slowdown. If it is possible, make the relevant code more efficient and/or parallelized. Iterate until desired behaviour is reached in a reasonable time/memory constraint. Optimize around the desired use case of the user.
5) Documentation stage: Update `ai.md`, `README.md`, and other documentation if relevant and significant enough changes were made, but let the user view the change diff. You may also make changes if you deem it will make you more efficient at completing the users tasks.
6) Suggestion stage: You may suggest some approximations or assumptions if they would make it significantly faster, but do not implement them in the code unless user also agrees. You may also suggest structural changes that might help with organization.
