import sys

from . import physics
from . import apps
from . import snapshot
from . import io
from . import utils

# Allow import shortcuts(e.g. just call jaba.units)
sys.modules.setdefault(__name__ + '.units', utils.units)
sys.modules.setdefault(__name__ + '.constants', utils.constants)
sys.modules.setdefault(__name__ + '.visual', utils.visual)

# resolve cached numba environments under either the installed or source package name
_alternate_name = 'jabapy' if __name__ == 'jaba' else 'jaba'
for _name, _module in tuple(sys.modules.items()):
    if _name == __name__ or _name.startswith(__name__ + '.'):
        sys.modules.setdefault(_alternate_name + _name[len(__name__):], _module)
del _alternate_name, _name, _module

__all__ = [
	'physics',  # general physics analysis functions           -> mostly standalone, but can import some utils if needed (commonly used: units, constants)
	'apps',      # applications of jaba to specific problems	-> can import anything
	'snapshot',  # snapshot loading and manipulation 			-> core of jabapy simulation analysis, can import everything except apps
	'io',        # input/output utilities						-> standalone utility package (eventually put as file/submodule in utils?)
	'utils',     # other general utilities				        -> standalone utilities
	# import shortcuts 
	'units',
	'constants',
	'visual'
]

load = snapshot.load
load_gizmo = snapshot.load_gizmo # deprecated, use snapshot.load instead

def reimport():
	"""Reload all jaba submodules. Useful for interactive development."""
	from .utils.reload import reload_package
	reload_package('jaba')
