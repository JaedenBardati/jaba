#!/usr/bin/env bash
REPO_LOCATION="$(cd "$(dirname "$0")" && pwd)" # start in the repo directory (assuming setup.sh is in the root)

##############################
# defaults
BASHRC_FILE="$HOME/.bashrc"  # only use bashrc for simplicity, even if on mac, but then redirect bash_profile to source bashrc
BASHRC_TEMP_FILE="${BASHRC_FILE}.tmp"
PYENVSH_FILE="$REPO_LOCATION/scripts/pyenv.sh"
QCSSH_FILE="$REPO_LOCATION/scripts/qcs/qcs.sh"

PYTHON_EXE="python3"
BREW_EXE="brew"
CONDA_EXE="conda"
ENVIRONMENT_NAME="venv"
ENVIRONMENT_TYPE="pip" 

JABA_VARIABLES_STRING="# >>> Added by Jaba >>>"
JABA_VARIABLES_ENDSTRING="# <<< Added by Jaba <<<"


### linux or mac?
SYSTEM_TYPE="$(uname -s)"
if [[ "$SYSTEM_TYPE" == "Darwin" ]]; then
    # MAC
    printf "I think you are using a Mac.\n"
    PYTHON_INSTALL_METHOD="homebrew"
elif [[ "$SYSTEM_TYPE" == "Linux" ]]; then
    # LINUX
    printf "I think you are using Linux.\n"
    PYTHON_INSTALL_METHOD="apt-get"
else
    printf "I can't tell what system you are using... I get '$SYSTEM_TYPE' from 'uname -s'. Please check and modify jaba's setup.sh accordingly.\n"
    exit 1
fi

## slurm or no scheduler?
SCHEDULER_EXE=""  # leave blank if no scheduler
SRUN_EXE=""
if sinfo --version &> /dev/null; then
    printf "Slurm is available.\n"
    SCHEDULER_EXE="sbatch"
    SRUN_EXE="srun"
    if module --version &> /dev/null; then
        printf "Module system is available.\n"
        PYTHON_INSTALL_METHOD="module"
    else
        printf "There is no module system available, but slurm somehow is? This is unexpected. Please check and modify jaba's setup.sh accordingly.\n"
    fi
else
    printf "Slurm is not available. I will assume that you have don't have a task scheduler.\n"
fi

### infer host, override variables as needed (CHANGE THIS IF YOU HAVE AN UNRECOGNIZED SYSTEM TYPE OR NEED PERSONAL DEFAULTS)
HOSTNAME="$(hostname)"
if [[ "$HOSTNAME" == *"frontera"* && "$SYSTEM_TYPE" == "Linux" && "$SCHEDULER_EXE" == "sbatch" ]]; then
    printf "I think you are on Frontera. Resetting parameters accordingly.\n"
    SRUN_EXE="ibrun"
    MAIN_PACKAGE_MODULES="intel/19.1.1 mvapich2-x/2.3 python3/3.7.0 phdf5/1.10.4"
elif [[ "$HOSTNAME" == "Jaedens-MacBook-Pro.local" && "$SYSTEM_TYPE" == "Darwin" && "$SCHEDULER_EXE" == "" ]]; then
    printf "I think you are on Jaeden's MacBook Pro. Resetting parameters accordingly.\n"
    ENVIRONMENT_TYPE="conda"
else
    printf "I don't recognize your host '$HOSTNAME'. " 
    if [[ $SCHEDULER_EXE == "" ]]; then
        printf "Since you don't appear to be on a cluster, I'll try setting up everything up using the defaults for your machine type.\n"
    else
        printf "Yet you appear to be on a cluster... I'll try using some defaults, but I would highly suggest modifying jaba's setup.sh to add the relevant modules.\n"
        MAIN_PACKAGE_MODULES="intel impi python3"
    fi
fi

### check if bashrc exists and if not, create it
if [[ ! -e "$BASHRC_FILE" ]]; then
    printf "Creating %s (it did not exist).\n" "$BASHRC_FILE"
    touch "$BASHRC_FILE" || { printf "Failed to create %s\n" "$BASHRC_FILE"; exit 1; }
fi

### if mac, add .zshrc and .bash_profile to source .bashrc if it doesn't already (or zshrc if using zsh)
if [[ "$SYSTEM_TYPE" == "Darwin" ]]; then
    ZSHRC_FILE="$HOME/.zshrc"
    if [[ ! -e "$ZSHRC_FILE" ]]; then
        printf "Creating %s (it did not exist).\n" "$ZSHRC_FILE"
        echo "source $BASHRC_FILE" > "$ZSHRC_FILE" || { printf "Failed to create %s\n" "$ZSHRC_FILE"; exit 1; }
    else
        if ! grep -q "source $BASHRC_FILE" "$ZSHRC_FILE"; then
            printf "Adding source bashrc command to %s.\n" "$ZSHRC_FILE"
            echo "source $BASHRC_FILE" >> "$ZSHRC_FILE" || { printf "Failed to update %s\n" "$ZSHRC_FILE"; exit 1; }
        fi
    fi

    BASH_PROFILE_FILE="$HOME/.bash_profile"
    if [[ ! -e "$BASH_PROFILE_FILE" ]]; then
        printf "Creating %s (it did not exist).\n" "$BASH_PROFILE_FILE"
        echo "source $BASHRC_FILE" > "$BASH_PROFILE_FILE" || { printf "Failed to create %s\n" "$BASH_PROFILE_FILE"; exit 1; }
    else
        if ! grep -q "source $BASHRC_FILE" "$BASH_PROFILE_FILE"; then
            printf "Adding source bashrc command to %s.\n" "$BASH_PROFILE_FILE"
            echo "source $BASHRC_FILE" >> "$BASH_PROFILE_FILE" || { printf "Failed to update %s\n" "$BASH_PROFILE_FILE"; exit 1; }
        fi
    fi
fi
printf "\n"


##############################
## main setup script

remove_block_between_markers() {
    local target_file="$1"
    local start_marker="$2"
    local end_marker="$3"

    if [[ ! -f "$target_file" ]]; then
        printf "No file found at %s; nothing to reset.\n" "$target_file"
        return 1
    fi

    local tmp_file
    tmp_file="$(mktemp)"

    awk -v start="$start_marker" -v end="$end_marker" '
        BEGIN { inblock=0; found_start=0; found_end=0 }
        $0==start { inblock=1; found_start=1; next }
        $0==end { if (inblock) { inblock=0; found_end=1; next } }
        !inblock { print }
        END {
            if (found_start && !found_end) exit 2
            if (!found_start) exit 3
        }
    ' "$target_file" > "$tmp_file"
    local awk_status=$?

    if [[ $awk_status -eq 2 ]]; then
        printf "Found start marker '%s' but not end marker '%s' in %s; refusing to modify.\n" "$start_marker" "$end_marker" "$target_file"
        rm -f "$tmp_file"
        return 1
    elif [[ $awk_status -eq 3 ]]; then
        #printf "Did not find start marker '%s' in %s; nothing to remove.\n" "$start_marker" "$target_file"
        rm -f "$tmp_file"
        return 0
    elif [[ $awk_status -ne 0 ]]; then
        printf "Failed to process %s for reset (awk status %s).\n" "$target_file" "$awk_status"
        rm -f "$tmp_file"
        return 1
    fi

    if ! mv "$tmp_file" "$target_file"; then
        printf "Failed to copy over temporary file %s to target file %s.\n" "$tmp_file" "$target_file"
        rm -f "$tmp_file"
        return 1
    fi

    rm -f "$tmp_file"
    return 0
}

### check if jaba block already exists in bashrc, and if so, ask user if they want to reset it, or keep it but skip the main setup
if grep -Fxq "$JABA_VARIABLES_STRING" "$BASHRC_FILE"; then
    printf "Jaba variables already in ${BASHRC_FILE}.\n"
    read -p "Reset jaba? [y/n] " reset_jaba
    if [[ "$reset_jaba" == "y" ]]; then
        DO_MAIN_SETUP="Y"
    else
        printf "Skipping main setup.\n"
        DO_MAIN_SETUP="N"
    fi
else
    printf "Jaba variables not already in ${BASHRC_FILE}.\n"
    DO_MAIN_SETUP="Y"
fi

if [[ $DO_MAIN_SETUP == "Y" ]]; then
    SETUP_PYTHON_ENVIRONMENT_COMMANDS=""

    ## > install/load in python
    if [[ $PYTHON_INSTALL_METHOD == "homebrew" ]]; then
        # >> install homebrew if needed
        if ! command -v "${BREW_EXE}" &> /dev/null; then
            read -p "Homebrew is not installed. Attempt to install homebrew? [y/n] " install_brew
            if [[ "$install_brew" == "y" ]]; then
                printf "Installing homebrew...\n"
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
                    printf "Homebrew installation failed.\n"
                    exit 1
                }

                # >>> make brew available in the current shell if installer didn't.
                if ! command -v "${BREW_EXE}" &> /dev/null; then
                    if [[ -x /opt/homebrew/bin/brew ]]; then
                        eval "$(/opt/homebrew/bin/brew shellenv)"
                    elif [[ -x /usr/local/bin/brew ]]; then
                        eval "$(/usr/local/bin/brew shellenv)"
                    fi
                fi
                
                if ! command -v "${BREW_EXE}" &> /dev/null; then
                    printf "Homebrew installed, but '%s' is still not on PATH. Restart your shell or add brew to PATH, then re-run setup.\n" "${BREW_EXE}"
                    exit 1
                fi
            else
                printf "Homebrew is required for this installation method. Please install homebrew or modify setup.sh to specify a different installation method.\n"
                exit 1
            fi
        fi
        # >> install python if needed
        if ! command -v "${PYTHON_EXE}" &> /dev/null; then
            read -p "Python3 is not installed. Attempt to install python3 via homebrew? [y/n] " install_python
            if [[ "$install_python" == "y" ]]; then
                printf "Installing python3 via homebrew...\n"
                $BREW_EXE install python3
            fi
        fi
        # >> install conda if needed
        if [[ "$ENVIRONMENT_TYPE" == "conda" ]] && ! command -v "${CONDA_EXE}" &> /dev/null; then
            read -p "Conda is not installed. Attempt to install conda via homebrew? [y/n] " install_conda
            if [[ "$install_conda" == "y" ]]; then
                printf "Installing conda via homebrew...\n"
                $BREW_EXE install --cask miniconda
                $CONDA_EXE init "$(basename "${SHELL}")"
            fi
        fi  
    elif [[ $PYTHON_INSTALL_METHOD == "apt-get" ]]; then
        # >> install python if needed
        if ! command -v "${PYTHON_EXE}" &> /dev/null; then
            read -p "Python3 is not installed. Attempt to install python3 via apt-get? [y/n] " install_python
            if [[ "$install_python" == "y" ]]; then
                printf "Installing python via apt-get...\n"
                sudo apt-get update
                sudo apt-get install python3 python3-pip
            fi
        fi
        # >> install conda if needed
        if [[ "$ENVIRONMENT_TYPE" == "conda" ]] && ! command -v "${CONDA_EXE}" &> /dev/null; then
            read -p "Conda is not installed. Attempt to install conda via homebrew? [y/n] " install_conda
            if [[ "$install_conda" == "y" ]]; then
                printf "Installing conda via homebrew...\n"
                $BREW_EXE install --cask miniconda
                $CONDA_EXE init "$(basename "${SHELL}")"
            fi
        fi 
    elif [[ $PYTHON_INSTALL_METHOD == "module" ]]; then
        # >> check that conda is not requested
        if [[ "$ENVIRONMENT_TYPE" == "conda" ]]; then
            printf "Conda environment requested, but module-based python loading does not support conda environments. Please modify setup.sh to specify a different environment type.\n"
            exit 1
        fi
        # >> test main package modules
        if ! module avail ${MAIN_PACKAGE_MODULES} &> /dev/null; then
            printf "Failed to load main package modules '%s'. Please check that these are correct for your system and modify setup.sh if needed.\n" "${MAIN_PACKAGE_MODULES}"
            exit 1
        fi
        SETUP_PYTHON_ENVIRONMENT_COMMANDS="${SETUP_PYTHON_ENVIRONMENT_COMMANDS} module purge; module load ${MAIN_PACKAGE_MODULES};"
    else
        printf "Unknown python installation method '%s'. Please modify setup.sh to specify how you want to install or load python.\n" "$PYTHON_INSTALL_METHOD"
        exit 1
    fi

    ### > make/load python environment
    if [[ "$ENVIRONMENT_TYPE" == "conda" ]]; then
        # >> check if there is already a conda environment with the same name, and use it if so, otherwise create a new one
        if $CONDA_EXE env list | grep -qE "^\s*${ENVIRONMENT_NAME}\s"; then
            printf "Conda environment '%s' already exists. Activating it...\n" "$ENVIRONMENT_NAME"
        else
            printf "Creating conda environment named '%s' ...\n" "$ENVIRONMENT_NAME"
            $CONDA_EXE create -n "$ENVIRONMENT_NAME" -y
        fi
        $CONDA_EXE activate "$ENVIRONMENT_NAME"
        SETUP_PYTHON_ENVIRONMENT_COMMANDS="${SETUP_PYTHON_ENVIRONMENT_COMMANDS} eval \\\"$(command conda 'shell.bash' 'hook' 2> /dev/null)\\\"; $CONDA_EXE activate ${ENVIRONMENT_NAME};"
    elif [[ "$ENVIRONMENT_TYPE" == "pip" ]]; then
        # >> check if there is already a pip environment, and use it if so, otherwise create a new one
        if [[ -d ".${ENVIRONMENT_NAME}" ]]; then
            printf "Pip environment '%s' already exists. Activating it...\n" ".${ENVIRONMENT_NAME}"
        else
            printf "Creating pip environment at '%s' ...\n" ".${ENVIRONMENT_NAME}"
            $PYTHON_EXE -m venv ".${ENVIRONMENT_NAME}"
        fi
        source ".${ENVIRONMENT_NAME}/bin/activate"
        SETUP_PYTHON_ENVIRONMENT_COMMANDS="${SETUP_PYTHON_ENVIRONMENT_COMMANDS} source \\\"$REPO_LOCATION/.${ENVIRONMENT_NAME}/bin/activate\\\";"
    else
        printf "Unknown environment type '%s'. Please modify setup.sh to specify a valid environment type.\n" "$ENVIRONMENT_TYPE"
        exit 1
    fi

    ### > install python module
    printf "Installing jaba as a module to python ${ENVIRONMENT_TYPE} environment ${ENVIRONMENT_NAME} ...\n"
    $PYTHON_EXE -m pip install --upgrade pip
    $PYTHON_EXE -m pip install -e . # note that you should install this to an environment you like

    ### > deactivate environment
    if [[ "$ENVIRONMENT_TYPE" == "conda" ]]; then
        $CONDA_EXE deactivate
    elif [[ "$ENVIRONMENT_TYPE" == "pip" ]]; then
        deactivate
    fi

    ### > setup jaba settings in bashrc 
    printf "\nSetting up jaba variables and aliases in %s ...\n" "$BASHRC_FILE"
    if [[ -f "$BASHRC_TEMP_FILE" ]]; then
        printf "Temp file %s already exists. Please check and remove it before running setup.sh again.\n" "$BASHRC_TEMP_FILE"
        exit 1
    fi
    
    rsync -ac "$BASHRC_FILE" "$BASHRC_TEMP_FILE" > /dev/null || { printf "Failed to create temporary copy of bashrc file.\n"; exit 1; }
    remove_block_between_markers "$BASHRC_TEMP_FILE" "$JABA_VARIABLES_STRING" "$JABA_VARIABLES_ENDSTRING" || exit 1
    {
        printf "\n${JABA_VARIABLES_STRING}\n"
        printf "export JABA_LOCATION=\"%s\"\n" "$REPO_LOCATION"

        printf "\n#python environment\n"
        printf "alias setup_jaba_python_environment=\"${SETUP_PYTHON_ENVIRONMENT_COMMANDS}\"\n"
        printf "alias pyenv='$PYENVSH_FILE'\n"
        printf "alias py='pyenv'\n"
        if [[ ! "$SCHEDULER_EXE" == "" ]]; then
            printf "alias pyq='sbatch $PYENVSH_FILE'\n"
        fi
        printf "alias qcs='${QCSSH_FILE}'\n"

        read -p "Also add Jaeden's other (non-jaba) general aliases? [y/n] " add_alias
        if [[ "$add_alias" == "y" ]]; then
            printf "\n#non-jaba general aliases\n"
            printf "alias tailf='tail -f'\n"
            if [[ ! "$SCHEDULER_EXE" == "" ]]; then
                printf "alias sq='squeue -u jbardati'\n"
            fi
        fi

        printf "${JABA_VARIABLES_ENDSTRING}\n\n"
    } >> "${BASHRC_TEMP_FILE}"

    printf "\nProposed changes to %s:\n" "$BASHRC_FILE"
    diff -u "$BASHRC_FILE" "$BASHRC_TEMP_FILE"
    read -p "Should I make the above changes to your .bashrc? [y/n] " confirm_bashrc
    if [[ "$confirm_bashrc" == "y" ]]; then
        mv -v "$BASHRC_TEMP_FILE" "$BASHRC_FILE" > /dev/null 
    else
        printf "Aborting bashrc changes and ending program. Temp file is at %s if you want to make the changes manually.\n" "$BASHRC_TEMP_FILE"
        exit 1
    fi
fi

##############################
### setup git submodules
read -p "Set up submodules? [y/n] " setup_submodules
if [[ "$setup_submodules" == "y" ]]; then
    printf "Setting up git submodules...\n"
    git submodule update --init --recursive
    git config --global push.recurseSubmodules on-demand
    git config --global submodule.recurse true

    ### > setup GIZ
    read -p "Set up GIZ? [y/n] " setup_giz
    if [[ "$setup_giz" == "y" ]]; then
        printf "Setting up GIZ submodule...\n"
        cd scripts/giz
        setup.sh
        cd ../..
    else
        printf "Skipping GIZ setup.\n"
    fi

    ### > setup SKI
    ### ...

    ### > setup ...
    ### ...

else
    printf "Skipping git submodule setup.\n"
fi
##############################

printf "Setup complete.\n"
exit 0