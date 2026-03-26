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

if [ -n "$SLURM_JOBID" ]; then
    echo "Error: Running jupyter notebook is not supported in a SLURM job. Please use regular python files with pyenv instead."
    exit 1
fi

# Main code
## Launches jupyter notebook in custom environment. Code produced by the jaba setup script.
##    The first file argument specifies a jupyter notebook file to run. Leaving this empty launches jupyter notebook interactively.
if [ "${1}" == "" ]; then
    echo "Launching jupyter notebook ..."
    ${JABA_JUPYTER_CMD}
else
    echo "Launching jupyter notebook with file ${1} ..."
    ${JABA_JUPYTER_CMD} ${1} ${@:2}
fi

############
deactivate_jaba_python_environment
