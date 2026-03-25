#!/usr/bin/env bash
#SBATCH -p "development"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00

############
# Setup jaba
source "$HOME/.bashrc"
if [ -z "${JABA_LOCATION+x}" ]; then
    echo "It seems that Jaba is not setup yet. Please run the jaba setup.sh script and try again."
    exit
fi
shopt -s expand_aliases
activate_jaba_python_environment
############

# Main code
## Runs a quick check on the inputed GIZMO snapshot (HDF5).
##   The first file argument specifies the snapshot file(s) to run the analysis on. 
##   By default, it will output to "./analysis" directory - change the variable below if you need.
##   If no file is specified, it will run in interactive mode (load all relevant modules for analysis).
##   If "." is specified as the file, it will run on all files in the current directory.
ANALYSIS_DIR="./analysis"
QCSPY_FILE="${JABA_LOCATION}/tools/quickchecksim.py"


if [ "$1" == "" ]; then
    echo "launching python interactively..."
    ${JABA_PYTHON_CMD} -i -c "import numpy as np; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import importlib.util, sys, os; spec=importlib.util.spec_from_file_location('quickchecksim',os.path.expanduser('${QCSPY_FILE}')); qcs=importlib.util.module_from_spec(spec); sys.modules['quickchecksim']=qcs; spec.loader.exec_module(qcs); from astropy import units as u;"
else
    if [[ ! -d "$ANALYSIS_DIR" ]]; then
        mkdir -p "$ANALYSIS_DIR"
        echo "created $ANALYSIS_DIR"
    else
        echo "$ANALYSIS_DIR already exists"
    fi

    if [ "$1" == "." ]; then
        echo "launching default quick check on all files in ${pwd} ..."
        for file in *.hdf5; do  # temp, should be moved to quickchecksim.py ? 
            echo "> FILE: ${file}"

            echo "${ANALYSIS_DIR}/dens_${file%.hdf5}.pdf"
            if [[ -f "${ANALYSIS_DIR}/dens_${file%.hdf5}.pdf" ]]; then
                echo "Skipping ${file} because it already has an output."
            else
                ${JABA_PYTHON_CMD} ${QCSPY_FILE} ${file} ${ANALYSIS_DIR} ${@:2}
            fi
        done
    else
        echo "launching default quick check on file ${1} ..."
        ${JABA_PYTHON_CMD} ${QCSPY_FILE} ${1} ${ANALYSIS_DIR} ${@:2}
    fi
fi

############
deactivate_jaba_python_environment
