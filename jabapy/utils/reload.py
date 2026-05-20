import sys
import importlib

def reload_package(package_name, verbose=False):
    """Reload a package and all its submodules."""
    for module_name in list(sys.modules.keys()):
        if module_name.startswith(package_name):
            if verbose:
                print(f"Reloading module: {module_name}")
            importlib.reload(sys.modules[module_name])

# Usage: reload_package('jaba')
