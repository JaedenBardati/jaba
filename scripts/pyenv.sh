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
## Launches python 3 in custom environment. Code produced by the jaba setup script.
##    The first file argument specifies a python file to run. Leaving this empty launches python interactively.
if [ "${1}" == "" ]; then
    if [ -n "$SLURM_JOBID" ]; then
        echo "Error: Running pyenv interactively is not supported in a SLURM job. Please specify a python file to run."
        exit 1
    fi
    echo "Launching python interactively ..."
    ${JABA_PYTHON_CMD} -i
else
    echo "Launching python with file ${1} ..."
    ${JABA_PYTHON_CMD} ${1} ${@:2}
fi

############
deactivate_jaba_python_environment
