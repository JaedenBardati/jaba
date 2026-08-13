#!/usr/bin/env bash
REPO_LOCATION="$(cd "$(dirname "$0")" && pwd)" # start in the repo directory (assuming setup.sh is in the root)
cd $REPO_LOCATION
printf "....JABA INSTALLATION....\n"

##############################
# defaults
BASHRC_FILE="$HOME/.bashrc"  # only use bashrc for simplicity, even if on mac, but then redirect bash_profile/zshrc to source bashrc
BASHRC_TEMP_FILE="${BASHRC_FILE}.tmp"
PYENVSH_FILE="$REPO_LOCATION/scripts/pyenv.sh"
QCSSH_FILE="$REPO_LOCATION/scripts/qcs.sh"

PYTHON_CMD="python3"
BREW_CMD="brew"
CONDA_CMD="conda"  # do NOT name this CONDA_EXE; conda init overwrites that variable
JUPYTER_CMD="jupyter notebook" # or jupyter lab if you prefer (change here)
PYTHON_ENVIRONMENT_NAME="jaba_env"
PYTHON_ENVIRONMENT_TYPE="pip" 

JABA_VARIABLES_STRING="# >>> Added by Jaba >>>"
JABA_VARIABLES_ENDSTRING="# <<< Added by Jaba <<<"

INFERRED_SYSTEM="Unknown" # to be set later


source "${BASHRC_FILE}"
JABA_LOCATION=REPO_LOCATION # safeguard for code errors

### linux or mac?
SYSTEM_TYPE="$(uname -s)"
if [[ "$SYSTEM_TYPE" == "Darwin" ]]; then
    # MAC
    printf "I think you are using a Mac.\n"
    PYTHON_INSTALL_METHOD="homebrew"
    INFERRED_SYSTEM="General Mac"
elif [[ "$SYSTEM_TYPE" == "Linux" ]]; then
    # LINUX
    printf "I think you are using Linux.\n"
    PYTHON_INSTALL_METHOD="apt-get"
    INFERRED_SYSTEM="General Linux Local"
else
    printf "I can't tell what system you are using... I get '$SYSTEM_TYPE' from 'uname -s'. Please check and modify jaba's setup.sh accordingly.\n"
    exit 1
fi
SYSTEM_SUBTYPE="$(uname -o)"

## slurm or no scheduler?
SCHEDULER_CMD=""  # leave blank if no scheduler
SRUN_CMD=""
if sinfo --version &> /dev/null; then
    printf "Slurm is available.\n"
    SCHEDULER_CMD="sbatch"
    SRUN_CMD="srun"
    if module --version &> /dev/null; then
        printf "Module system is available.\n"
        PYTHON_INSTALL_METHOD="module"
    else
        printf "There is no module system available, but slurm somehow is? This is unexpected. Please check and modify jaba's setup.sh accordingly.\n"
    fi
else
    printf "Slurm is not available. I will assume that you have don't have a task scheduler.\n"
fi
if [[ "$SYSTEM_TYPE" == "Linux" && "$SCHEDULER_CMD" != "" ]]; then
    INFERRED_SYSTEM="General Linux Server"
fi

### infer host, override variables as needed (CHANGE THIS IF YOU HAVE AN UNRECOGNIZED SYSTEM TYPE OR NEED PERSONAL DEFAULTS)
HOSTNAME="$(hostname)"
if [[ "$HOSTNAME" == *"frontera"* && "$SYSTEM_TYPE" == "Linux" && "$SCHEDULER_CMD" == "sbatch" ]]; then
    printf "I think you are on Frontera. Resetting parameters accordingly.\n"
    INFERRED_SYSTEM="Frontera"
    SRUN_CMD="ibrun"
    MAIN_PACKAGE_MODULES="intel/19.1.1 mvapich2-x/2.3 python3/3.7.0 phdf5/1.10.4"
elif [[ "$HOSTNAME" == "Jaedens-MacBook-Pro.local" && "$SYSTEM_TYPE" == "Darwin" && "$SCHEDULER_CMD" == "" ]]; then
    printf "I think you are on Jaeden's MacBook Pro. Resetting parameters accordingly.\n"
    INFERRED_SYSTEM="Jaedens MacBook"
    PYTHON_ENVIRONMENT_TYPE="conda"
#...
else
    printf "I don't recognize your host '$HOSTNAME'. " 
    if [[ $SCHEDULER_CMD == "" ]]; then
        printf "Since you don't appear to be on a cluster, I'll try setting up everything up using the defaults for your machine type.\n"
    else
        printf "Yet you appear to be on a cluster...\n" 
        printf "[warning] I'll try using some defaults, but I would highly suggest modifying jaba's setup.sh to add the relevant modules.\n"
        MAIN_PACKAGE_MODULES="intel impi ${PYTHON_CMD}"
    fi
fi
printf "\n"

### check if bashrc exists and if not, create it
if [[ ! -e "$BASHRC_FILE" ]]; then
    printf "Creating %s (it did not exist).\n" "$BASHRC_FILE"
    touch "$BASHRC_FILE" || { printf "Failed to create %s\n" "$BASHRC_FILE"; exit 1; }
fi

### if local, add .zshrc and .bash_profile to source .bashrc if it doesn't already (or zshrc if using zsh)
if [[ "$SCHEDULER_CMD" == "" ]]; then
    ZSHRC_FILE="$HOME/.zshrc"
    if [[ ! -e "$ZSHRC_FILE" ]]; then
	if [[ "${SYSTEM_TYPE}" == "Darwin" ]]; then 
            read -p "Do want to create a .zshrc file that sources .bashrc? (do this if you regularly use zsh, recommended for mac) [y/n] " create_zshrc
            if [[ "$create_zshrc" == "y" ]]; then
                printf "Creating %s (it did not exist).\n" "$ZSHRC_FILE"
                echo "source $BASHRC_FILE" > "$ZSHRC_FILE" || { printf "Failed to create %s\n" "$ZSHRC_FILE"; exit 1; }
            fi
	else
            printf "skipping .zshrc redirect to .bashrc since you are not on a mac\n"
	fi
    else
        source "$ZSHRC_FILE"
        if ! grep -q "source $BASHRC_FILE" "$ZSHRC_FILE"; then
            printf "Adding source bashrc command to %s.\n" "$ZSHRC_FILE"
            echo "source $BASHRC_FILE" >> "$ZSHRC_FILE" || { printf "Failed to update %s\n" "$ZSHRC_FILE"; exit 1; }
        else
	    printf "your .zshrc already redirects to .bashrc\n"
	fi
    fi

    BASH_PROFILE_FILE="$HOME/.bash_profile"
    if [[ ! -e "$BASH_PROFILE_FILE" ]]; then
        read -p "Do want to create a .bash_profile file that sources .bashrc? (do this if you regularly use bash) [y/n] " create_bash_profile
        if [[ "$create_bash_profile" == "y" ]]; then
            printf "Creating %s (it did not exist).\n" "$BASH_PROFILE_FILE"
            echo "source $BASHRC_FILE" > "$BASH_PROFILE_FILE" || { printf "Failed to create %s\n" "$BASH_PROFILE_FILE"; exit 1; }
        fi
    else
        source "$BASH_PROFILE_FILE"
        if ! grep -q "source $BASHRC_FILE" "$BASH_PROFILE_FILE"; then
            printf "Adding source bashrc command to %s.\n" "$BASH_PROFILE_FILE"
            echo "source $BASHRC_FILE" >> "$BASH_PROFILE_FILE" || { printf "Failed to update %s\n" "$BASH_PROFILE_FILE"; exit 1; }
        else
	    printf "your .bash_profile already redirects to .bashrc\n"
	fi
    fi

    SH_PROFILE_FILE="$HOME/.profile"
    if [[ ! -e "$SH_PROFILE_FILE" ]]; then
	if [[ "${SYSTEM_TYPE}" != "Darwin" ]]; then
            read -p "Do want to create a .profile file that sources .bashrc? (do this if you regularly use sh) [y/n] " create_sh_profile
            if [[ "$create_sh_profile" == "y" ]]; then
                printf "Creating %s (it did not exist).\n" "$SH_PROFILE_FILE"
                echo "source $BASHRC_FILE" > "$SH_PROFILE_FILE" || { printf "Failed to create %s\n" "$SH_PROFILE_FILE"; exit 1; }
	    fi
	else
            printf "skipping .profile redirect to .bashrc since you are on a mac\n"
	fi
    else
        source "$SH_PROFILE_FILE"
        if ! grep -q "source $BASHRC_FILE" "$SH_PROFILE_FILE"; then
            printf "Adding source bashrc command to %s.\n" "$SH_PROFILE_FILE"
            echo "source $BASHRC_FILE" >> "$SH_PROFILE_FILE" || { printf "Failed to update %s\n" "$SH_PROFILE_FILE"; exit 1; }
        else
	    printf "your .profile already redirects to .bashrc\n"
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
DO_MAIN_SETUP="N"
if grep -Fxq "$JABA_VARIABLES_STRING" "$BASHRC_FILE"; then
    printf "Jaba variables already in ${BASHRC_FILE}.\n"
    read -p "Reset jaba? [y/n] " reset_jaba
    if [[ "$reset_jaba" == "y" ]]; then
        DO_MAIN_SETUP="Y"
    else
        printf "Skipping main setup.\n"
    fi
else
    printf "Jaba variables not already in ${BASHRC_FILE}.\n"
    DO_MAIN_SETUP="Y"
fi

if [[ $DO_MAIN_SETUP == "Y" ]]; then
    ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS=""
    DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS=""

    ## > install/load in python
    if [[ $PYTHON_INSTALL_METHOD == "homebrew" ]]; then
        # >> install homebrew if needed
        if ! command -v "${BREW_CMD}" &> /dev/null; then
            read -p "Homebrew is not installed. Attempt to install homebrew? [y/n] " install_brew
            if [[ "$install_brew" == "y" ]]; then
                printf "Installing homebrew...\n"
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
                    printf "Homebrew installation failed.\n"
                    exit 1
                }

                # >>> make brew available in the current shell if installer didn't.
                if ! command -v "${BREW_CMD}" &> /dev/null; then
                    if [[ -x /opt/homebrew/bin/brew ]]; then
                        eval "$(/opt/homebrew/bin/brew shellenv)"
                    elif [[ -x /usr/local/bin/brew ]]; then
                        eval "$(/usr/local/bin/brew shellenv)"
                    fi
                fi
                
                if ! command -v "${BREW_CMD}" &> /dev/null; then
                    printf "Homebrew installed, but '%s' is still not on PATH. Restart your shell or add brew to PATH, then re-run setup.\n" "${BREW_CMD}"
                    exit 1
                fi
            else
                printf "Homebrew is required for this installation method. Please install homebrew or modify setup.sh to specify a different installation method.\n"
                exit 1
            fi
        fi
        # >> install python if needed
        if ! command -v "${PYTHON_CMD}" &> /dev/null; then
            read -p "Python3 is not installed. Attempt to install ${PYTHON_CMD} via homebrew? [y/n] " install_python
            if [[ "$install_python" == "y" ]]; then
                printf "Installing ${PYTHON_CMD} via homebrew...\n"
                $BREW_CMD install ${PYTHON_CMD}
            fi
        fi
        # >> install conda if needed
        if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]] && ! command -v "${CONDA_CMD}" &> /dev/null; then
            read -p "Conda is not installed. Attempt to install conda via homebrew? [y/n] " install_conda
            if [[ "$install_conda" == "y" ]]; then
                printf "Installing conda via homebrew...\n"
                $BREW_CMD install --cask miniconda
            fi
        fi  
    elif [[ $PYTHON_INSTALL_METHOD == "apt-get" ]]; then
        # >> install vim if needed
        if ! command -v "vim" &> /dev/null; then
            read -p "vim is not installed. Attempt to install vim via apt-get? [y/n] " install_vim
            if [[ "$install_vim" == "y" ]]; then
                printf "Installing vim via apt-get...\n"
                (
                sudo apt-get -y update
                sudo apt-get -y install vim
                ) | while read -r LINE; do echo "[vim install] $LINE"; done
            fi
        fi
        # >> install git if needed
        if ! command -v "git" &> /dev/null; then
            read -p "git is not installed. Attempt to install git via apt-get? [y/n] " install_git
            if [[ "$install_git" == "y" ]]; then
                printf "Installing git via apt-get...\n"
                (
                sudo apt-get -y update
                sudo apt-get -y install git
                ) | while read -r LINE; do echo "[git install] $LINE"; done
            fi
        fi
        # >> install rsync if needed
        if ! command -v "rsync" &> /dev/null; then
            read -p "rsync is not installed. Attempt to install rsync via apt-get? [y/n] " install_rsync
            if [[ "$install_rsync" == "y" ]]; then
                printf "Installing rsync via apt-get...\n"
                (
                sudo apt-get -y update
                sudo apt-get -y install rsync
                ) | while read -r LINE; do echo "[rsync install] $LINE"; done
            fi
        fi
        # >> install python if needed
        if ! command -v "${PYTHON_CMD}" &> /dev/null; then
            read -p "Python3 is not installed. Attempt to install ${PYTHON_CMD} via apt-get? [y/n] " install_python
            if [[ "$install_python" == "y" ]]; then
                printf "Installing ${PYTHON_CMD} via apt-get...\n"
                (
		sudo apt-get -y update
                sudo apt-get -y install ${PYTHON_CMD}
	        ) | while read -r LINE; do echo "[${PYTHON_CMD} install] $LINE"; done
            fi
        fi
	# >> install pip if needed
	if ! "${PYTHON_CMD}" -m pip --help &> /dev/null; then
	    read -p "Pip is not installed. Attempt to install pip via apt-get? [y/n] " install_pip
	    if [[ "$install_pip" == "y" ]]; then
                printf "Installing pip via apt-get...\n"
		(
		sudo apt-get -y update
		sudo apt-get -y install ${PYTHON_CMD}-pip
	        ) | while read -r LINE; do echo "[${PYTHON_CMD}-pip install] $LINE"; done 
	    fi
	fi
	# >> install venv if needed
        if [[ "$PYTHON_ENVIRONMENT_TYPE" == "pip" ]] && ( ( [[ "$SYSTEM_SUBTYPE" == "GNU/Linux" ]] && ! command -v "${PYTHON_CMD}-venv" &> /dev/null ) || ( ! "${PYTHON_CMD}" -m venv --help &> /dev/null) ) ; then
            read -p "Venv is not installed. Attempt to install venv via apt-get? [y/n] " install_venv
	    if [[ "$install_venv" == "y" ]]; then
                printf "Installing venv via apt-get...\n"
		(
		sudo apt-get -y update
		sudo apt-get -y install ${PYTHON_CMD}-venv
	        ) | while read -r LINE; do echo "[${PYTHON_CMD}-venv install] $LINE"; done
	    fi
        fi
        # >> install conda if needed
	if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]] && ! command -v "${CONDA_CMD}" &> /dev/null; then
            read -p "Conda is not installed. Attempt to install conda via homebrew? [y/n] " install_conda
            if [[ "$install_conda" == "y" ]]; then
                printf "Installing conda via homebrew...\n"
                $BREW_CMD install --cask miniconda 
            fi
        fi 
    elif [[ $PYTHON_INSTALL_METHOD == "module" ]]; then
        # >> check that conda is not requested
        if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]]; then
            printf "Conda environment requested, but module-based python loading does not support conda environments. Please modify setup.sh to specify a different environment type.\n"
            exit 1
        fi
        # >> test main package modules
        if ! module avail ${MAIN_PACKAGE_MODULES} &> /dev/null; then
            printf "Failed to load main package modules '%s'. Please check that these are correct for your system and modify setup.sh if needed.\n" "${MAIN_PACKAGE_MODULES}"
            exit 1
        fi
        ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }module purge; module load ${MAIN_PACKAGE_MODULES};"
    else
        printf "Unknown python installation method '%s'. Please modify setup.sh to specify how you want to install or load python.\n" "$PYTHON_INSTALL_METHOD"
        exit 1
    fi

    ### > make/load python environment
    if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]]; then
        # >> check if there is already a conda environment with the same name, and use it if so, otherwise create a new one
        if $CONDA_CMD env list | grep -qE "^\s*${PYTHON_ENVIRONMENT_NAME}\s"; then
            printf "Conda environment '%s' already exists. Activating it...\n" "$PYTHON_ENVIRONMENT_NAME"
        else
            printf "Creating conda environment named '%s' ...\n" "$PYTHON_ENVIRONMENT_NAME"
	    $CONDA_CMD create -n "$PYTHON_ENVIRONMENT_NAME" python -y | while read -r LINE; do echo "[conda environment creation] $LINE"; done
        fi
        # source conda's shell hook so that `conda activate` works (it's a shell function, not a binary command)
        source "$($CONDA_CMD info --base)/etc/profile.d/conda.sh"
        while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done # deactivate any existing conda envs first so activate puts the env at the front of PATH
        conda activate "$PYTHON_ENVIRONMENT_NAME" || { printf "Failed to activate conda environment '%s'.\n" "$PYTHON_ENVIRONMENT_NAME"; exit 1; }
        ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }"'source "'"$($CONDA_CMD info --base)/etc/profile.d/conda.sh"'";'" conda activate ${PYTHON_ENVIRONMENT_NAME};"
    elif [[ "$PYTHON_ENVIRONMENT_TYPE" == "pip" ]]; then
        # >> check if there is already a pip environment, and use it if so, otherwise create a new one
	if [[ -f ".${PYTHON_ENVIRONMENT_NAME}/bin/activate" ]]; then
            printf "Pip environment '%s' already exists. Activating it...\n" ".${PYTHON_ENVIRONMENT_NAME}"
        else
            printf "Creating pip environment at '%s' ...\n" ".${PYTHON_ENVIRONMENT_NAME}"
            ( yes | $PYTHON_CMD -m venv ".${PYTHON_ENVIRONMENT_NAME}" ) | while read -r LINE; do echo "[pip environment creation] $LINE"; done
        fi
        source ".${PYTHON_ENVIRONMENT_NAME}/bin/activate"
        ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }"'source "'"$REPO_LOCATION/.${PYTHON_ENVIRONMENT_NAME}/bin/activate"'";'
    else
        printf "Unknown environment type '%s'. Please modify setup.sh to specify a valid environment type.\n" "$PYTHON_ENVIRONMENT_TYPE"
        exit 1
    fi

    ### > install python module
    printf "Installing jaba as a module to python ${PYTHON_ENVIRONMENT_TYPE} environment ${PYTHON_ENVIRONMENT_NAME} ...\n"
    (
    yes | $PYTHON_CMD -m pip install --upgrade pip
    yes | $PYTHON_CMD -m pip install -e . # note that you should install this to an environment you like
    ) | while read -r LINE; do echo "[jabapy install] $LINE"; done

    ### > deactivate environment
    if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]]; then
        conda deactivate
        DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }conda deactivate;"
    elif [[ "$PYTHON_ENVIRONMENT_TYPE" == "pip" ]]; then
        deactivate
        DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }deactivate;"
    fi

    ### > setup jaba settings in bashrc 
    printf "\nSetting up jaba variables and aliases in %s ...\n" "$BASHRC_FILE"
    if [[ -f "$BASHRC_TEMP_FILE" ]]; then
        printf "Temp file %s already exists, likely left over from a previous failed installation. " "$BASHRC_TEMP_FILE"
	read -p "Would you like to remove it? [y/n]" remove_tmp_bashrc
	if [[ "$remove_tmp_bashrc" == "y" ]]; then
	    printf "Okay, removing temporary file.\n"
            rm "$BASHRC_TEMP_FILE"
	else
            printf "Please check and remove ${BASHRC_TEMP_FILE} manually before proceeding with installation/reinstallation.\n"
            exit 1
	fi
    fi
    
    rsync -ac "$BASHRC_FILE" "$BASHRC_TEMP_FILE" > /dev/null || { printf "Failed to create temporary copy of bashrc file.\n"; exit 1; }
    remove_block_between_markers "$BASHRC_TEMP_FILE" "$JABA_VARIABLES_STRING" "$JABA_VARIABLES_ENDSTRING" || exit 1
    {
        printf "${JABA_VARIABLES_STRING}\n"
        printf "export JABA_LOCATION=\"%s\"\n" "$REPO_LOCATION"
        #printf "export JABA_DATE_INSTALLED_UTC=\"%s\"\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        printf "export JABA_MARKER_START=\"%s\"\n" "$JABA_VARIABLES_STRING"
        printf "export JABA_MARKER_END=\"%s\"\n" "$JABA_VARIABLES_ENDSTRING"

        printf "export JABA_INFERRED_SYSTEM=\"%s\"\n" "$INFERRED_SYSTEM"
        printf "export JABA_HOSTNAME=\"%s\"\n" "$HOSTNAME"
        printf "export JABA_SYSTEM_TYPE=\"%s\"\n" "$SYSTEM_TYPE"
        printf "export JABA_SYSTEM_SUBTYPE=\"%s\"\n" "$SYSTEM_SUBTYPE"
        printf "export JABA_SCHEDULER_CMD=\"%s\"\n" "$SCHEDULER_CMD"
        printf "export JABA_SRUN_CMD=\"%s\"\n" "$SRUN_CMD"

        printf "export JABA_PYTHON_CMD=\"%s\"\n" "$PYTHON_CMD"
        printf "export JABA_CONDA_CMD=\"%s\"\n" "$CONDA_CMD"
        printf "export JABA_JUPYTER_CMD=\"%s\"\n" "$JUPYTER_CMD"
        printf "export JABA_PYTHON_ENVIRONMENT_NAME=\"%s\"\n" "$PYTHON_ENVIRONMENT_NAME"
        printf "export JABA_PYTHON_ENVIRONMENT_TYPE=\"%s\"\n" "$PYTHON_ENVIRONMENT_TYPE"

        printf "alias jaba-uninstall=\"(cd ${REPO_LOCATION}; bash ./uninstall.sh;)\"\n"
        printf "alias jaba-reinstall=\"(cd ${REPO_LOCATION}; bash ./install.sh;); source '$HOME/.bashrc';\"\n"
        printf "alias jaba-update=\"(cd ${REPO_LOCATION}; bash ./uninstall.sh; git pull origin; bash ./install.sh;); source '$HOME/.bashrc';\"\n"

        printf "\n#python environment\n"
	    printf "activate_jaba_python_environment () { %s }\n" "${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}"
	    printf "export -f activate_jaba_python_environment "'&> /dev/null'"\n"
        printf "deactivate_jaba_python_environment () { %s }\n" "${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}"
	    printf "export -f deactivate_jaba_python_environment "'&> /dev/null'"\n"
	    printf "alias jaba-activate=\"activate_jaba_python_environment;\"\n"
	    printf "alias jaba-deactivate=\"deactivate_jaba_python_environment;\"\n"
        printf "alias pyenv='$PYENVSH_FILE'\n"
        printf "alias py='pyenv'\n"
        if [[ ! "$SCHEDULER_CMD" == "" ]]; then
            printf "alias pyq='$SCHEDULER_CMD $PYENVSH_FILE'\n"
        fi
        printf "alias jupy='${REPO_LOCATION}/scripts/jupy.sh'\n"
        printf "alias qcs='${QCSSH_FILE}'\n"
        if [[ ! "$SCHEDULER_CMD" == "" ]]; then
            printf "alias qcsq='$SCHEDULER_CMD $QCSSH_FILE'\n"
        fi
        
        read -p "Also add jaba development aliases? [y/n] " add_jaba_dev_aliases
        if [[ "$add_jaba_dev_aliases" == "y" ]]; then
            printf "\n#jaba development aliases\n"
            printf "alias jaba-cd=\"cd ${REPO_LOCATION}\"\n"
            printf "alias jaba-cd-scripts\"cd ${REPO_LOCATION}/scripts\"\n"
	    printf "alias jaba-pwd\"echo ${REPO_LOCATION}\"\n"
            printf "alias jaba-edit-install=\"vim ${REPO_LOCATION}/install.sh;\"\n"
            printf "alias jaba-edit-py=\"vim ${PYENVSH_FILE};\"\n"
            printf "alias jaba-edit-pyq=jaba-edit-py\n"
            printf "alias jaba-edit-qcs=\"vim ${QCSSH_FILE}; vim ${REPO_LOCATION}/tools/quickchecksim.py;\"\n"
            printf "alias jaba-edit-qcsq=jaba-edit-qcs\n"
            printf "alias jaba-edit-todo=\"vim ${REPO_LOCATION}/TODO.txt;\"\n"
            printf "alias jaba-pull=\"(cd ${REPO_LOCATION}; git pull origin;)\"\n"
            printf "alias jaba-status=\"(cd ${REPO_LOCATION}; git status;)\"\n"
	        printf "alias jaba-diff=\"(cd ${REPO_LOCATION}; git diff;)\"\n"
	        printf "alias jaba-commit=\"(cd ${REPO_LOCATION}; git add .; git commit;)\"\n"
            printf "alias jaba-push=\"(cd ${REPO_LOCATION}; git push origin;)\"\n"
            printf "alias jaba-fetch=\"(cd ${REPO_LOCATION}; git fetch origin;)\"\n"
            printf "alias jaba-softupdate=\"(cd ${REPO_LOCATION}; git pull origin; bash ./install.sh;); source '$HOME/.bashrc';\"\n"
        fi

        read -p "Also add Jaeden's other (non-jaba) general aliases? [y/n] " add_general_aliases
        if [[ "$add_general_aliases" == "y" ]]; then
            printf "\n#non-jaba general aliases\n"
            printf "alias tailf='tail -f'\n"
            if [[ "$SCHEDULER_CMD" == "sbatch" ]]; then
                printf "alias sq='squeue -u $(whoami)'\n"
            fi
	    printf "alias ss='source ${BASHRC_FILE}'\n"
        fi

        printf "${JABA_VARIABLES_ENDSTRING}\n"
    } >> "${BASHRC_TEMP_FILE}"

    
    FILE_DIFFERENCE=$(diff -u "$BASHRC_FILE" "$BASHRC_TEMP_FILE")
    if [[ ! -z "$FILE_DIFFERENCE" ]]; then
        printf "\nProposed changes to %s:\n" "$BASHRC_FILE"
        #printf "%s\n" "${FILE_DIFFERENCE}"
        diff --color -u "$BASHRC_FILE" "$BASHRC_TEMP_FILE"
	read -p "Should I make the above changes to your .bashrc? [y/n] " confirm_bashrc
        if [[ "$confirm_bashrc" == "y" ]]; then
            mv -v "$BASHRC_TEMP_FILE" "$BASHRC_FILE" > /dev/null 
        else
            read -p "Okay, I will abort the bashrc changes and end the program. Should I keep a backup of the proposed changes for you to look at? [y/n] " keep_bashrc_changes
            if [[ "$keep_bashrc_changes" == "y" ]]; then
                printf "Keeping proposed changes at %s.\n" "$BASHRC_TEMP_FILE"
            else
                rm "$BASHRC_TEMP_FILE" # remove temp file
            fi
            exit 0
        fi
    else
        printf "No changes were made to the .bashrc file. Removing temporary file...\n"
        rm "$BASHRC_TEMP_FILE"
    fi
fi

printf "\n"
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
        (
            cd scripts/giz
            git checkout main
            bash setup.sh || exit 1
        )
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
