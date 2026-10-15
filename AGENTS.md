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
- Avoid unnecessary or excessively detailed comments. Keep comments high-level and brief.

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
- When you do a tiny correction commit and didn't push to Github yet, just ammend the commit instead to avoid commit bloat.

## Communication

- Keep the user informed with any changes to `jaba`. Before introducing a new physical assumption or approximation that changes scientific results, briefly explain it and confirm it with the user. Existing documented choices do not need repeated confirmation unless the task changes their applicability.

## Typical workflow

Use this as a rough template for the development process when implementing a new feature, particularly one that is computationally time- or memory-intensive. Adapt the workflow to the complexity of the task; not every change requires every stage.

0) **Understanding stage:** Define the desired functionality, expected inputs and outputs, constraints, and acceptable approximations. Ask the user for clarification when essential details are missing or not clear (remember you can infer some details from existing architecture). Break the task down into concrete, manageable steps and identify potential challenges before beginning.

1) **Research stage:** Explore the existing codebase for relevant functions, classes, utilities, and established coding patterns that can be reused. Understand how the proposed change interacts with existing functionality, dependencies, and data structures. Do not reinvent the wheel or introduce unnecessary complexity.

2) **Proto-typing stage:** Implement a simple, working version of the feature that achieves the desired behavior. Prioritize correctness, readability, and compatibility with the existing code over performance at this stage. Establish a functional baseline before attempting significant optimizations.

3) **Testing stage:** Verify that the implementation produces the expected results, including edge cases, numerical accuracy, and compatibility with existing functionality. Compare against analytical solutions, known results, or the previous implementation where possible. Iterate until the desired behavior is achieved. Add appropriate automated tests in tests/ for sufficiently significant changes, and ensure existing tests continue to pass.

4) **Efficiency stage:** Profile or benchmark the working implementation to determine whether it is computationally time- and/or memory-limited, and identify the dominant bottlenecks. Prioritize optimizations with the greatest expected impact, such as improving algorithms, reducing redundant calculations, minimizing allocations, improving memory access patterns, or introducing parallelization. Avoid premature optimization and unnecessary complexity. After each significant optimization, verify that correctness is preserved and measure the actual improvement. Optimize primarily for the user's intended use cases, representative input sizes, and available hardware.

5) **Clean up stage:** Remove temporary files, debugging statements, unused variables, obsolete code, and experimental implementations that are no longer needed. Avoid leaving unnecessary dependencies or configuration changes. Ensure that the final implementation is readable, maintainable, and consistent with the surrounding codebase. Do not delete unrelated files or user data. Remove unnecessary or excessive comments.

6) **Documentation stage:** Update relevant documentation when changes are sufficiently significant. MINIMALLY CHANGE THIS DOCUMENTATION, especially for small changes. To change `README.md` and/or `AGENTS.md`, explicit permission must be obtained from user. Document new functionality, interfaces, dependencies, configuration options, limitations, and important implementation decisions as appropriate. Keep documentation concise and consistent with the actual code. Allow the user to review documentation changes through the diff. You may also improve project instructions or documentation when doing so would meaningfully improve the efficiency or reliability of future development tasks.

7) **Review stage:** Review the final diff for unintended modifications, regressions, unnecessary complexity, and opportunities for simplification. Run relevant tests and benchmarks one final time. Summarize what changed, the tests performed and their results, any measured performance improvements, and any remaining limitations. Clearly distinguish between verified results and untested assumptions. Do not claim that tests passed or performance improved without having verified them. Commit git changes or push to Github when relevant.

8) **Suggestion stage:** Suggest potential approximations, assumptions, or alternative algorithms that could significantly improve performance, memory usage, or maintainability, but do not implement changes that materially affect accuracy or expected behavior without the user's approval. You may also suggest larger structural or organizational improvements, especially when they would make future development easier. Keep these suggestions separate from the completed implementation rather than expanding the scope unnecessarily.


## General principles

- **Correctness before performance:** Establish and verify a working baseline before optimizing. Never sacrifice correctness for speed without explicitly discussing the tradeoff with the user.

- **Measure before optimizing:** Use profiling, benchmarks, and representative workloads to guide performance improvements rather than relying on assumptions.

- **Prefer minimal, targeted changes:** Reuse existing infrastructure and avoid unnecessary refactoring or changes to unrelated functionality.

- **Preserve existing behavior:** Maintain backward compatibility unless a change is explicitly requested or necessary. Identify and communicate any unavoidable breaking changes.

- **Numerical robustness:** For scientific or numerical code, consider floating-point precision, convergence, conservation properties, boundary conditions, and reproducibility where applicable.

- **Scalability:** Consider how computational cost and memory requirements scale with problem size. Prefer algorithms that remain practical for realistic workloads, not just small test cases.

- **Parallelization with care:** Account for synchronization overhead, race conditions, thread safety, load balancing, and memory duplication when introducing parallelism.

- **Avoid unnecessary dependencies:** Prefer existing libraries and standard functionality unless a new dependency provides a clear benefit.

- **Respect the user's scope:** Do not introduce speculative features, approximations, or broad architectural changes without a compelling reason or explicit approval.

- **Communicate meaningful decisions:** Explain important design choices, tradeoffs, limitations, and deviations from the original request without overwhelming the user with routine implementation details.
