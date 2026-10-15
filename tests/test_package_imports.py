import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestPackageImports(unittest.TestCase):
    def test_numba_cache_across_package_names(self):
        """Read caches from either namespace without the repository on sys.path."""
        package = Path(__file__).resolve().parents[1] / 'jabapy' / '__init__.py'
        script = '''
import importlib
import importlib.util
import sys
import numpy as np

# load the package under the requested name without adding its parent to sys.path
name, path, expect_hit = sys.argv[1:]
spec = importlib.util.spec_from_file_location(name, path)
package = importlib.util.module_from_spec(spec)
sys.modules[name] = package
spec.loader.exec_module(package)
other = 'jaba' if name == 'jabapy' else 'jabapy'
assert importlib.import_module(other) is package
grid = importlib.import_module(name + '.utils.grid')
assert importlib.import_module(other + '.utils.grid') is grid
assert importlib.import_module(other + '.visual') is package.utils.visual

# exercise the cached kernel and verify conservation after deserialization
result = grid.bin_particles_kernel([[0., 0., 0.]], [2.], h=[.8],
                                   mins=[-1.]*3, maxs=[1.]*3, dims=[8]*3)
np.testing.assert_allclose(result.sum(), 2.)
if expect_hit == 'yes':
    assert sum(grid._deposit_cubic_cells.stats.cache_hits.values()) > 0
'''

        # use separate caches to check both source-to-installed and installed-to-source
        for names in (('jabapy', 'jaba'), ('jaba', 'jabapy')):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                env = dict(os.environ, NUMBA_CACHE_DIR=directory, MPLBACKEND='Agg')
                for index, name in enumerate(names):
                    result = subprocess.run([sys.executable, '-I', '-c', script, name, str(package),
                                             'yes' if index else 'no'], cwd=directory, env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr[-4000:])


if __name__ == '__main__':
    unittest.main()
