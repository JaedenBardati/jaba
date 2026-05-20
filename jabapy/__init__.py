import sys

from . import analysis
from . import apps
from . import snapshot
from . import io
from . import utils

# Allow import shortcuts(e.g. just call jaba.units)
sys.modules.setdefault(__name__ + '.units', utils.units)
sys.modules.setdefault(__name__ + '.constants', utils.constants)
sys.modules.setdefault(__name__ + '.visual', utils.visual)

__all__ = [
	'analysis',  # general physics analysis functions           -> mostly standalone, but can import some utils if needed (commonly used: units, constants)
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
