"""
This will help download and prepare any resource packages needed for Jaba.
"""

import os
import gzip
import re
from functools import lru_cache
import urllib.request
from urllib.parse import unquote, urlsplit
import tarfile, zipfile
import shutil
from pathlib import Path

import numpy as np
import scipy as sp

# --- resource loaders ---
def _load_BC03_spectra(directory):
    # Reads the binary SSP tables distributed in the BC03 2016 resource archives.

    # locate the compressed or unpacked spectra in the resource directory
    paths = sorted(directory.rglob('*.ised.gz')) or sorted(directory.rglob('*.ised'))
    if not paths:
        raise FileNotFoundError(f'No BC03 .ised tables found in {directory}')
    tables = []

    # read fortran records, including their leading and trailing byte counts
    def read_record(stream, endian):
        marker = stream.read(4)
        if len(marker) != 4:
            raise ValueError('Truncated BC03 record marker.')
        size = int.from_bytes(marker, endian)
        record = stream.read(size)
        if len(record) != size or stream.read(4) != marker:
            raise ValueError('Invalid BC03 record length.')
        return record

    # recover the age grid, metallicity, wavelength grid and spectra from each table
    for path in paths:
        with (gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')) as stream:
            marker = stream.read(4)
            endian = 'little' if 0 < int.from_bytes(marker, 'little') < 100000 else 'big'
            dtype = '<f4' if endian == 'little' else '>f4'
            stream.seek(0)
            header = read_record(stream, endian)
            n_age = int.from_bytes(header[:4], endian)
            ages = np.frombuffer(header, dtype=dtype, count=n_age, offset=4).astype(float)
            metallicity = float(re.search(rb'\bZ\s*=\s*([0-9.Ee+-]+)', header[4+4*n_age:]).group(1))
            record = read_record(stream, endian)
            n_lam = int.from_bytes(record[:4], endian)
            wavelengths = np.frombuffer(record, dtype=dtype, count=n_lam, offset=4).astype(float)
            spectra = np.empty((n_age, n_lam), dtype=np.float32)
            for i in range(n_age):
                record = read_record(stream, endian)
                if int.from_bytes(record[:4], endian) != n_lam:
                    raise ValueError(f'Inconsistent BC03 spectrum length in {path}')
                spectra[i] = np.frombuffer(record, dtype=dtype, count=n_lam, offset=4)
        if tables and (not np.array_equal(ages, tables[0][1]) or not np.array_equal(wavelengths, tables[0][2])):
            raise ValueError('BC03 tables must share age and wavelength grids.')
        tables.append((metallicity, ages, wavelengths, spectra))

    # order the metallicity grid and retain the parsed tables for subsequent calls
    tables.sort(key=lambda table: table[0])
    metallicities = np.array([table[0] for table in tables])
    if np.any(np.diff(metallicities) <= 0) or np.any(np.diff(ages) <= 0) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError('BC03 grids must be strictly increasing.')
    return metallicities, ages, wavelengths, np.stack([table[3] for table in tables])

def _BC03_current_mass_fraction(model, metals, ages):
    """Read living-star plus remnant fractions from the SSP .4color tables."""

    # convert the companion mass tables onto the spectral age grid
    directory = get_resource_path('stellar_population_models', model, '')
    paths = sorted(directory.rglob('*.4color.gz')) or sorted(directory.rglob('*.4color'))
    fractions = []
    for path in paths:
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
            text = stream.read()
        metal = float(re.search(r'\bZ\s*=\s*([0-9.Ee+-]+)', text).group(1))
        data = np.loadtxt(text.splitlines())
        fraction = np.interp(ages, 10**data[:, 0], data[:, 10], left=1.0)
        fractions.append((metal, fraction))
    fractions.sort(key=lambda row: row[0])
    if len(fractions) != len(metals) or not np.allclose([row[0] for row in fractions], metals):
        raise ValueError('BC03 .4color mass tables must match the spectral metallicity grid.')
    return sp.interpolate.RegularGridInterpolator((metals, ages), np.stack([row[1] for row in fractions]))


def _load_bandpass(directory):
    """Read an SVO filter-profile VOTable, preserving response and detector metadata."""
    from astropy.io.votable import parse_single_table
    from astropy import units as u
    table = parse_single_table(directory/'profile.xml', verify='ignore')
    params = {param.name: param.value for param in table.params}
    wavelength = np.asarray(table.array['Wavelength'], dtype=float)*u.Unit(params['WavelengthUnit'])
    transmission = np.asarray(table.array['Transmission'], dtype=float)
    if wavelength.size < 2 or np.any(~np.isfinite(wavelength.value)) or np.any(wavelength.value <= 0) or np.any(np.diff(wavelength.value) <= 0):
        raise ValueError('Filter wavelengths must be finite, positive and strictly increasing.')
    if transmission.shape != wavelength.shape or np.any(~np.isfinite(transmission)) or np.any(transmission < 0) or not np.any(transmission > 0):
        raise ValueError('Filter transmission must be finite, nonnegative and nonzero.')
    return dict(wavelength=wavelength.to(u.AA), transmission=transmission,
                detector='photon' if str(params['DetectorType']) == '1' else 'energy',
                description=params['Description'], filter_id=params['filterID'],
                source='https://svo2.cab.inta-csic.es/theory/fps/index.php?id='+params['filterID'])


# --- configuration ---
CACHE_DIR = Path.home() / ".cache" / "jabapy"

RESOURCE_REGISTRY = {
    # --- stellar population model data ---
    "stellar_population_models": {
        f"BC03_{library}_{imf}": {"url": f"https://www.bruzual.org/bc03/Updated_version_2016/BC03_{library}_{imf}.tgz", "loader": _load_BC03_spectra}
        for library in ("basel", "miles", "stelib")
        for imf in ("chabrier", "kroupa", "salpeter", "TopHeavyIMF")
        # ...
    },
    "bandpass_filters": {
        f"{system}.{band}": {
            "url": f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={prefix}.{band}",
            "filename": "profile.xml", "loader": _load_bandpass,
        }
        for system, bands, prefix in (("Johnson", "UBVRI", "Generic/Johnson"), ("SDSS", "ugriz", "SLOAN/SDSS"))
        for band in bands
    }
    # ...
}

# --- fetch engine ---
def _fetch_and_extract(category: str, package_name: str) -> Path:
    """Resolves nested configs, downloads archives, and extracts them to their sub-key names."""

    # resolve the category and package name
    category_group = RESOURCE_REGISTRY.get(category)
    if not category_group:
        raise ValueError(f"Unknown resource category: '{category}'")
    info = category_group.get(package_name)
    if not info:
        raise ValueError(f"Unknown package '{package_name}' inside category '{category}'")
    
    # lazy load return or create the target directory if needed
    target_dir = CACHE_DIR / category / package_name
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # infer the archive type from the file extension
    url = info["url"]
    filename = info.get("filename", Path(unquote(urlsplit(url).path)).name)
    if not filename or filename in (".", "..") or Path(filename).name != filename:
        raise ValueError(f"Missing resource filename in URL: {url}")
    filename_lower = filename.lower()
    if filename_lower.endswith((".tar.gz", ".tgz", ".tar")):
        archive_type = "tar"
    elif filename_lower.endswith(".zip"):
        archive_type = "zip"
    else:
        archive_type = None
    
    temp_archive = CACHE_DIR / f"{package_name}_temp.download" # temporary file for download
    try:
        # download the archive
        urllib.request.urlretrieve(url, temp_archive)
        
        # extract the archive depending on the type
        if archive_type == "tar":
            with tarfile.open(temp_archive, "r:*") as tar_ref:
                if hasattr(tarfile, 'data_filter'):
                    tar_ref.extractall(target_dir, filter='data')
                else:
                    for member in tar_ref.getmembers():
                        target_path = (target_dir / member.name).resolve()
                        if not target_path.is_relative_to(target_dir.resolve()):
                            raise Exception(f"Malicious path blocked: {member.name}")
                    tar_ref.extractall(target_dir)
        elif archive_type == "zip":
            with zipfile.ZipFile(temp_archive, 'r') as zip_ref:
                if hasattr(zip_ref, 'extraction_policy'):
                    zip_ref.extractall(target_dir, filter='data')
                else:
                    zip_ref.extractall(target_dir)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_archive, target_dir / filename)

        return target_dir

    # clean up partial/broken directories if download fails
    except Exception as e:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        raise RuntimeError(f"Failed to lazy-load resource '{package_name}': {e}")

    # always delete the temporary raw download file
    finally:
        if temp_archive.exists():
            temp_archive.unlink()

# --- public API ---
def get_resource_path(category: str, package_name: str, filename: str) -> Path:
    """General utility method to get a specific resource file path."""
    package_path = _fetch_and_extract(category, package_name)
    return package_path / filename 


@lru_cache(maxsize=None)
def _load_resource_cached(directory, loader):
    return loader(directory)


def load_resource(category: str, package_name: str, cache=True):
    """Downloads a resource and returns its registry loader's output, or its directory if no loader is defined.

    Loaders accept the extracted package directory as a Path. The two most recently
    used loader/directory pairs are cached in memory. cache=False bypasses this cache
    without replacing existing entries; downloaded files remain cached on disk.
    Cached outputs are shared between callers and should be treated as read-only.
    """

    # resolve the local resource and invoke its optional parser
    directory = get_resource_path(category, package_name, '')
    loader = RESOURCE_REGISTRY[category][package_name].get('loader')
    if loader is None:
        return directory
    return _load_resource_cached(directory, loader) if cache else loader(directory)


def clear_resource_cache():
    """Releases all parsed resource cache entries without removing downloaded files."""
    _load_resource_cached.cache_clear()

