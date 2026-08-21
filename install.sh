#!/usr/bin/env bash
REPO_LOCATION="$(cd "$(dirname "$0")" && pwd)" # start in the repo directory (assuming setup.sh is in the root)
cd $REPO_LOCATION

info()   { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()   { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
prompt() { printf -v p "\033[1;36m[PROMPT]\033[0m ${1}: "; read -p "$p" "$2"; }
prompt_yn() { prompt "$1 [y/n]" YN; YN=$(echo "$YN" | tr -d ' ' | tr '[:upper:]' '[:lower:]'); }
custom_qualifer() { echo -e '\033[38;5;208m['"${1}"']\033[0m '"$2"; } 

info ".... JABA INSTALLATION ...."
##############################
# defaults
BASHRC_FILE="$HOME/.bashrc"  # only use bashrc for simplicity, even if on mac, but then redirect bash_profile/zshrc to source bashrc
BASHRC_TEMP_FILE="${BASHRC_FILE}.tmp"
PYENVSH_FILE="$REPO_LOCATION/scripts/pyenv.sh"
QCSSH_FILE="$REPO_LOCATION/scripts/qcs.sh"

VIMRC_FILE="$HOME/.vimrc"
VIMRC_TEMP_FILE="${VIMRC_FILE}.tmp"

PYTHON_CMD="python3"
BREW_CMD="brew"
CONDA_CMD="conda"  # do NOT name this CONDA_EXE; conda init overwrites that variable
JUPYTER_CMD="jupyter notebook" # or jupyter lab if you prefer (change here)
PYTHON_ENVIRONMENT_NAME="jaba_env"
PYTHON_ENVIRONMENT_TYPE="pip" 

JABA_ADDED_STRING=">>> Added by Jaba >>>"
JABA_ADDED_ENDSTRING="<<< Added by Jaba <<<"
JABA_VARIABLES_STRING="# ${JABA_ADDED_STRING}"
JABA_VARIABLES_ENDSTRING="# ${JABA_ADDED_ENDSTRING}"
JABA_VARIABLES_STRING_VIM="\" ${JABA_ADDED_STRING}"
JABA_VARIABLES_ENDSTRING_VIM="\" ${JABA_ADDED_ENDSTRING}"

INFERRED_SYSTEM="Unknown" # to be set later


source "${BASHRC_FILE}"
JABA_LOCATION=REPO_LOCATION # safeguard for code errors

### linux or mac?
SYSTEM_TYPE="$(uname -s)"
if [[ "$SYSTEM_TYPE" == "Darwin" ]]; then
    # MAC
    info "I think you are using a Mac."
    PYTHON_INSTALL_METHOD="homebrew"
    INFERRED_SYSTEM="General Mac"
elif [[ "$SYSTEM_TYPE" == "Linux" ]]; then
    # LINUX
    info "I think you are using Linux."
    PYTHON_INSTALL_METHOD="apt-get"
    INFERRED_SYSTEM="General Linux Local"
else
    error "I can't tell what system you are using... I get '$SYSTEM_TYPE' from 'uname -s'. Please check and modify jaba's setup.sh accordingly."
fi
SYSTEM_SUBTYPE="$(uname -o)"

## slurm or no scheduler?
SCHEDULER_CMD=""  # leave blank if no scheduler
SRUN_CMD=""
if sinfo --version &> /dev/null; then
    info "Slurm is available."
    SCHEDULER_CMD="sbatch"
    SRUN_CMD="srun"
    if module --version &> /dev/null; then
        info "Module system is available."
        PYTHON_INSTALL_METHOD="module"
    else
        error "There is no module system available, but slurm somehow is? This is unexpected. Please check and modify jaba's setup.sh accordingly."
    fi
else
    info "Slurm is not available. I will assume that you have don't have a task scheduler."
fi
if [[ "$SYSTEM_TYPE" == "Linux" && "$SCHEDULER_CMD" != "" ]]; then
    INFERRED_SYSTEM="General Linux Server"
fi

### infer host, override variables as needed
HOSTNAME="$(hostname -f)"
if [[ "$HOSTNAME" == *"frontera"* && "$SYSTEM_TYPE" == "Linux" && "$SCHEDULER_CMD" == "sbatch" ]]; then
    info "I think you are on Frontera. Resetting parameters accordingly."
    INFERRED_SYSTEM="Frontera"
    SRUN_CMD="ibrun"
    MAIN_PACKAGE_MODULE_LOAD_COMMANDS="module purge; module load intel/19.1.1 mvapich2-x/2.3 python3/3.7.0 phdf5/1.10.4;"
elif [[ "$HOSTNAME" == "Jaedens-MacBook-Pro.local" && "$SYSTEM_TYPE" == "Darwin" && "$SCHEDULER_CMD" == "" ]]; then
    info "I think you are on Jaeden's MacBook Pro. Resetting parameters accordingly."
    INFERRED_SYSTEM="Jaedens MacBook"
    PYTHON_ENVIRONMENT_TYPE="conda"
elif [[ "$HOSTNAME" == *"frontier"* && "$SYSTEM_TYPE" == "Linux" && "$SCHEDULER_CMD" == "sbatch" ]]; then
    info "I think you are on Frontier. Resetting parameters accordingly."
    INFERRED_SYSTEM="Frontier"
    MAIN_PACKAGE_MODULE_LOAD_COMMANDS="module reset; module swap PrgEnv-cray PrgEnv-gnu; module load cray-mpich cray-python cray-hdf5;"

# ADD A NEW CONDITION STATEMENT HERE IF YOU HAVE AN UNRECOGNIZED SYSTEM TYPE OR NEED PERSONAL DEFAULTS ...

else
    info "I don't recognize your host '$HOSTNAME'." 
    if [[ $SCHEDULER_CMD == "" ]]; then
        info "Since you don't appear to be on a cluster, I'll try setting up everything up using the defaults for your machine type."
    else
        warn "Yet you appear to be on a cluster... I'll try using some defaults, but I would highly suggest modifying jaba's setup.sh to add the relevant modules."
        MAIN_PACKAGE_MODULE_LOAD_COMMANDS="module reset; module load python hdf5;"
    fi
fi

### check if bashrc exists and if not, create it
if [[ ! -e "$BASHRC_FILE" ]]; then
    info "Creating ${BASHRC_FILE} (it did not exist)."
    touch "$BASHRC_FILE" || { error "Failed to create ${BASHRC_FILE}"; }
fi

### if local, add .zshrc and .bash_profile to source .bashrc if it doesn't already (or zshrc if using zsh)
if [[ "$SCHEDULER_CMD" == "" ]]; then
    ZSHRC_FILE="$HOME/.zshrc"
    if [[ ! -e "$ZSHRC_FILE" ]]; then
	if [[ "${SYSTEM_TYPE}" == "Darwin" ]]; then 
            prompt_yn "Do want to create a .zshrc file that sources .bashrc? (do this if you regularly use zsh, recommended for mac)"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Creating ${ZSHRC_FILE} (it did not exist)."
                echo "source $BASHRC_FILE" > "$ZSHRC_FILE" || { error "Failed to create ${ZSHRC_FILE}"; }
            fi
	else
            info "skipping .zshrc redirect to .bashrc since you are not on a mac"
	fi
    else
        source "$ZSHRC_FILE"
        if ! grep -q "source $BASHRC_FILE" "$ZSHRC_FILE"; then
            info "Adding source bashrc command to ${ZSHRC_FILE}."
            echo "source $BASHRC_FILE" >> "$ZSHRC_FILE" || { error "Failed to update ${ZSHRC_FILE}"; }
        else
	    info "your .zshrc already redirects to .bashrc"
	fi
    fi

    BASH_PROFILE_FILE="$HOME/.bash_profile"
    if [[ ! -e "$BASH_PROFILE_FILE" ]]; then
        prompt_yn "Do want to create a .bash_profile file that sources .bashrc? (do this if you regularly use bash)"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            info "Creating ${BASH_PROFILE_FILE} (it did not exist)."
            echo "source $BASHRC_FILE" > "$BASH_PROFILE_FILE" || { error "Failed to create ${BASH_PROFILE_FILE}"; }
        fi
    else
        source "$BASH_PROFILE_FILE"
        if ! grep -q "source $BASHRC_FILE" "$BASH_PROFILE_FILE"; then
            info "Adding source bashrc command to ${BASH_PROFILE_FILE}."
            echo "source $BASHRC_FILE" >> "$BASH_PROFILE_FILE" || { error "Failed to update ${BASH_PROFILE_FILE}"; }
        else
	    info "your .bash_profile already redirects to .bashrc"
	fi
    fi

    SH_PROFILE_FILE="$HOME/.profile"
    if [[ ! -e "$SH_PROFILE_FILE" ]]; then
	if [[ "${SYSTEM_TYPE}" != "Darwin" ]]; then
            prompt_yn "Do want to create a .profile file that sources .bashrc? (do this if you regularly use sh)"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Creating ${SH_PROFILE_FILE} (it did not exist)."
                echo "source $BASHRC_FILE" > "$SH_PROFILE_FILE" || { error "Failed to create ${SH_PROFILE_FILE}"; }
	    fi
	else
            info "skipping .profile redirect to .bashrc since you are on a mac"
	fi
    else
        source "$SH_PROFILE_FILE"
        if ! grep -q "source $BASHRC_FILE" "$SH_PROFILE_FILE"; then
            info "Adding source bashrc command to ${SH_PROFILE_FILE}."
            echo "source $BASHRC_FILE" >> "$SH_PROFILE_FILE" || { error "Failed to update ${SH_PROFILE_FILE}"; }
        else
	    info "your .profile already redirects to .bashrc"
	fi
    fi
fi

##############################
## main setup script

remove_block_between_markers() {
    local target_file="$1"
    local start_marker="$2"
    local end_marker="$3"

    if [[ ! -f "${target_file}" ]]; then
        info "No file found at ${target_file}; nothing to reset."
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
        info "Found start marker '${start_marker}' but not end marker '${end_marker}' in ${target_file}; refusing to modify."
        rm -f "$tmp_file"
        return 1
    elif [[ $awk_status -eq 3 ]]; then
        #info "Did not find start marker \"${start_marker}\" in \"${target_file}\"; nothing to remove."
        rm -f "$tmp_file"
        return 0
    elif [[ $awk_status -ne 0 ]]; then
        info "Failed to process ${target_file} for reset (awk status ${awk_status})."
        rm -f "$tmp_file"
        return 1
    fi

    if ! mv "$tmp_file" "$target_file"; then
        info "Failed to copy over temporary file ${tmp_file} to target file ${target_file}."
        rm -f "$tmp_file"
        return 1
    fi

    rm -f "$tmp_file"
    return 0
}

### check if jaba block already exists in bashrc, and if so, ask user if they want to reset it, or keep it but skip the main setup
DO_MAIN_SETUP="N"
if grep -Fxq "$JABA_VARIABLES_STRING" "$BASHRC_FILE"; then
    info "Jaba variables already in ${BASHRC_FILE}."
    prompt_yn "Reset jaba?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        DO_MAIN_SETUP="Y"
    else
        info "Skipping main setup."
    fi
else
    info "Jaba variables not already in ${BASHRC_FILE}."
    DO_MAIN_SETUP="Y"
fi

if [[ $DO_MAIN_SETUP == "Y" ]]; then
    ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS=""
    DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS=""

    ## > install/load in python and other required packages
    if [[ $PYTHON_INSTALL_METHOD == "homebrew" ]]; then
        # >> install homebrew if needed
        if ! command -v "${BREW_CMD}" &> /dev/null; then
            prompt_yn "Homebrew is not installed. Attempt to install homebrew?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                error "Installing homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
                    error "Homebrew installation failed."
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
                    error "Homebrew installed, but '${BREW_CMD}' is still not on PATH. Restart your shell or add brew to PATH, then re-run setup."
                fi
            else
                error "Homebrew is required for this installation method. Please install homebrew or modify setup.sh to specify a different installation method."
            fi
        fi

        # >> install git if needed
        if ! command -v "git" &> /dev/null; then
            prompt_yn "git is not installed. Attempt to install git via homebrew?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing git via homebrew..."
                $BREW_CMD install git
            fi
        fi
        # >> install rsync if needed (not strictly required)
        if ! command -v "rsync" &> /dev/null; then
            prompt_yn "rsync is not installed. Attempt to install rsync via homebrew?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing rsync via homebrew..."
                $BREW_CMD install rsync
            fi
        fi
        # >> install tmux if needed (not strictly required)
        if ! command -v "tmux" &> /dev/null; then
            prompt_yn "tmux is not installed. Attempt to install tmux via homebrew?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing tmux via homebrew..."
                $BREW_CMD install tmux
            fi
        fi
        # >> install python if needed
        if ! command -v "${PYTHON_CMD}" &> /dev/null; then
            prompt_yn "Python3 is not installed. Attempt to install ${PYTHON_CMD} via homebrew?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing ${PYTHON_CMD} via homebrew..."
                $BREW_CMD install ${PYTHON_CMD}
            fi
        fi
        # >> install conda if needed
        if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]] && ! command -v "${CONDA_CMD}" &> /dev/null; then
            prompt_yn "Conda is not installed. Attempt to install conda via homebrew?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing conda via homebrew..."
                $BREW_CMD install --cask miniconda
            fi
        fi  
    elif [[ $PYTHON_INSTALL_METHOD == "apt-get" ]]; then
        # >> install vim if needed
        if ! command -v "vim" &> /dev/null; then
            prompt_yn "vim is not installed. Attempt to install vim via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing vim via apt-get..."
                (
                sudo apt-get -y update
                sudo apt-get -y install vim
                ) | while read -r LINE; do custom_qualifer "vim install" "$LINE"; done
            fi
        fi
        # >> install git if needed
        if ! command -v "git" &> /dev/null; then
            prompt_yn "git is not installed. Attempt to install git via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing git via apt-get..."
                (
                sudo apt-get -y update
                sudo apt-get -y install git
                ) | while read -r LINE; do custom_qualifer "git install" "$LINE"; done
            fi
        fi
        # >> install rsync if needed (not strictly required)
        if ! command -v "rsync" &> /dev/null; then
            prompt_yn "rsync is not installed. Attempt to install rsync via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing rsync via apt-get..."
                (
                sudo apt-get -y update
                sudo apt-get -y install rsync
                ) | while read -r LINE; do custom_qualifer "rsync install" "$LINE"; done
            fi
        fi
        # >> install tmux if needed (not strictly required)
        if ! command -v "tmux" &> /dev/null; then
            prompt_yn "tmux is not installed. Attempt to install tmux via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing tmux via apt-get..."
                (
                sudo apt-get -y update
                sudo apt-get -y install tmux
                ) | while read -r LINE; do custom_qualifer "tmux install" "$LINE"; done
            fi
        fi
        # >> install python if needed
        if ! command -v "${PYTHON_CMD}" &> /dev/null; then
            prompt_yn "Python3 is not installed. Attempt to install ${PYTHON_CMD} via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing ${PYTHON_CMD} via apt-get..."
                (
		        sudo apt-get -y update
                sudo apt-get -y install ${PYTHON_CMD}
	            ) | while read -r LINE; do custom_qualifer "${PYTHON_CMD} install" "$LINE"; done
            fi
        fi
        # >> install pip if needed
        if ! "${PYTHON_CMD}" -m pip --help &> /dev/null; then
            prompt_yn "Pip is not installed. Attempt to install pip via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing pip via apt-get..."
                (
                sudo apt-get -y update
                sudo apt-get -y install ${PYTHON_CMD}-pip
                ) | while read -r LINE; do custom_qualifer "${PYTHON_CMD}-pip install" "$LINE"; done 
            fi
        fi
        # >> install venv if needed
        if [[ "$PYTHON_ENVIRONMENT_TYPE" == "pip" ]] && ( ( [[ "$SYSTEM_SUBTYPE" == "GNU/Linux" ]] && ! command -v "${PYTHON_CMD}-venv" &> /dev/null ) || ( ! "${PYTHON_CMD}" -m venv --help &> /dev/null) ) ; then
            prompt_yn "Venv is not installed. Attempt to install venv via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                    info "Installing venv via apt-get..."
            (
            sudo apt-get -y update
            sudo apt-get -y install ${PYTHON_CMD}-venv
            ) | while read -r LINE; do custom_qualifer "${PYTHON_CMD}-venv install" "$LINE"; done
            fi
        fi
        # >> install conda if needed
	    if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]] && ! command -v "${CONDA_CMD}" &> /dev/null; then
            prompt_yn "Conda is not installed. Attempt to install conda via apt-get?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Installing conda via apt-get..."
                (
                sudo apt-get -y update
                sudo apt-get -y install miniconda
                ) | while read -r LINE; do custom_qualifer "conda install" "$LINE"; done
            fi
        fi 
    elif [[ $PYTHON_INSTALL_METHOD == "module" ]]; then
        # >> check that conda is not requested
        if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]]; then
            error "Conda environment requested, but module-based python loading does not support conda environments. Please modify setup.sh to specify a different environment type."
        fi
        # >> test main package modules
        ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }${}"
    else
        error "Unknown python installation method \"${PYTHON_INSTALL_METHOD}\". Please modify setup.sh to specify how you want to install or load python."
    fi

    ### > setup git/github if needed
    SETUP_GIT=0
    if [ -z "$(git config user.name)" ]; then
        prompt "Git user.name is not set on this system. Please enter your git user.name" GIT_USER_NAME
        git config --global user.name "$GIT_USER_NAME"
        SETUP_GIT=1
    fi
    if [ -z "$(git config user.email)" ]; then
        prompt "Git user.email is not set on this system. Please enter your git user.email (note this should match e.g., one of your GitHub emails)" GIT_USER_EMAIL
        git config --global user.email "$GIT_USER_EMAIL"
        SETUP_GIT=1
    fi
    if [[ $SETUP_GIT -eq 1 ]]; then
        if [ "$(git config init.defaultBranch)" != "main" ]; then
            prompt_yn "This might be an older version of git since the default branch is not set to main. Do you want to set it to main? (recommended)"
            git config --global init.defaultBranch main
        fi

        prompt_yn "Do you want to set up Github SSH keys for this system? (recommended for pushing to GitHub)"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            info "Okay, I'll launch a generic script to set up SSH keys. Many of the prompts may be irrelevant for your use case."
            printf "%s\n" "--------------------------------"
            /bin/bash "$REPO_LOCATION/scripts/setup_ssh.sh" </dev/tty || { error "Failed to set up Github SSH keys."; }
            printf "%s\n" "--------------------------------"
            info "Done setting up Github SSH keys."
        fi
    fi

    ### > make/load python environment
    if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]]; then
        # >> check if there is already a conda environment with the same name, and use it if so, otherwise create a new one
        if $CONDA_CMD env list | grep -qE "^\s*${PYTHON_ENVIRONMENT_NAME}\s"; then
            info "Conda environment \"${PYTHON_ENVIRONMENT_NAME}\" already exists. Activating it..."
        else
            info "Creating conda environment named \"${PYTHON_ENVIRONMENT_NAME}\" ..."
	    $CONDA_CMD create -n "$PYTHON_ENVIRONMENT_NAME" python -y | while read -r LINE; do echo "[conda environment creation] $LINE"; done
        fi
        # source conda's shell hook so that `conda activate` works (it's a shell function, not a binary command)
        source "$($CONDA_CMD info --base)/etc/profile.d/conda.sh"
        while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done # deactivate any existing conda envs first so activate puts the env at the front of PATH
        conda activate "$PYTHON_ENVIRONMENT_NAME" || { error "Failed to activate conda environment \"${PYTHON_ENVIRONMENT_NAME}\"."; }
        ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }"'source "'"$($CONDA_CMD info --base)/etc/profile.d/conda.sh"'";'" conda activate ${PYTHON_ENVIRONMENT_NAME};"
    elif [[ "$PYTHON_ENVIRONMENT_TYPE" == "pip" ]]; then
        # >> check if there is already a pip environment, and use it if so, otherwise create a new one
	if [[ -f ".${PYTHON_ENVIRONMENT_NAME}/bin/activate" ]]; then
            info "Pip environment \"${PYTHON_ENVIRONMENT_NAME}\" already exists. Activating it..."
        else
            info "Creating pip environment at \"${REPO_LOCATION}/${PYTHON_ENVIRONMENT_NAME}\" ..."
            ( yes | $PYTHON_CMD -m venv ".${PYTHON_ENVIRONMENT_NAME}" ) | while read -r LINE; do custom_qualifer "pip environment creation" "$LINE"; done
        fi
        source ".${PYTHON_ENVIRONMENT_NAME}/bin/activate"
        ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${ACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }"'source "'"$REPO_LOCATION/.${PYTHON_ENVIRONMENT_NAME}/bin/activate"'";'
    else
        error "Unknown environment type \"${PYTHON_ENVIRONMENT_TYPE}\". Please modify setup.sh to specify a valid environment type."
    fi

    ### > install python module
    info "Installing jaba as a module to python ${PYTHON_ENVIRONMENT_TYPE} environment ${PYTHON_ENVIRONMENT_NAME} ..."
    (
    yes | $PYTHON_CMD -m pip install --upgrade pip
    yes | $PYTHON_CMD -m pip install -e . # note that you should install this to an environment you like
    ) | while read -r LINE; do custom_qualifer "jabapy install" "$LINE"; done

    ### > deactivate environment
    if [[ "$PYTHON_ENVIRONMENT_TYPE" == "conda" ]]; then
        conda deactivate
        DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }conda deactivate;"
    elif [[ "$PYTHON_ENVIRONMENT_TYPE" == "pip" ]]; then
        deactivate
        DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS="${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS}${DEACTIVATE_PYTHON_ENVIRONMENT_COMMANDS:+ }deactivate;"
    fi

    ### > setup jaba settings in bashrc 
    info "Setting up jaba variables and aliases in $BASHRC_FILE ..."
    if [[ -f "$BASHRC_TEMP_FILE" ]]; then
        info "Temp file ${BASHRC_TEMP_FILE} already exists, likely left over from a previous failed installation."
        prompt_yn "Would you like to remove it?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            info "Okay, removing temporary file."
                rm -f "${BASHRC_TEMP_FILE}"
        else
                error "Please check and remove ${BASHRC_TEMP_FILE} manually before proceeding with installation/reinstallation."
        fi
    fi
    
    cp "$BASHRC_FILE" "$BASHRC_TEMP_FILE" > /dev/null || { error "Failed to create temporary copy of bashrc file."; }
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
        printf "alias jaba-reinstall=\"(cd ${REPO_LOCATION}; bash ./install.sh;); source '$BASHRC_FILE';\"\n"
        printf "alias jaba-update=\"(cd ${REPO_LOCATION}; bash ./uninstall.sh; git pull origin; bash ./install.sh;); source '$BASHRC_FILE';\"\n"

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

        prompt_yn "Also add extra script aliases?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            printf "\n#jaba extra script aliases\n"
            printf "alias sxh='${REPO_LOCATION}/scripts/sxh.sh'\n"
            printf "alias setup-ssh='${REPO_LOCATION}/scripts/setup_ssh.sh'\n"
        fi

        prompt_yn "Also add jaba development aliases?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            printf "\n#jaba development aliases\n"
            printf "alias jaba-cd=\"cd ${REPO_LOCATION}\"\n"
            printf "alias jaba-scripts-cd=\"cd ${REPO_LOCATION}/scripts\"\n"
	        printf "alias jaba-pwd=\"echo ${REPO_LOCATION}\"\n"
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
            printf "alias jaba-softupdate=\"(cd ${REPO_LOCATION}; git pull origin; bash ./install.sh;); source '$BASHRC_FILE';\"\n"
        fi

        NON_JABA_SETTINGS_INSTALL=0
        prompt_yn "Also add Jaeden's other (non-jaba) general settings?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            NON_JABA_SETTINGS_INSTALL=1
            printf "\n#non-jaba general aliases\n"
            printf "alias ss='source ${BASHRC_FILE}'\n"
            printf "alias tailf='tail -f'\n"
            if [[ "$SCHEDULER_CMD" == "sbatch" ]]; then
                printf "alias sq='squeue -u $(whoami)'\n"
            fi
            #colors
            printf "unset LSCOLORS\n"
            printf "CLICOLOR=1\n"
            printf "export LSCOLORS=exfxcxdxcxegedabagaced\n"
            printf "alias ls='ls -G --color=auto'\n"
            #for tmux stuff
            printf "export TERM=xterm-256color\n"
        fi

        printf "${JABA_VARIABLES_ENDSTRING}\n"
    } >> "${BASHRC_TEMP_FILE}"

    
    FILE_DIFFERENCE=$(diff -u "$BASHRC_FILE" "$BASHRC_TEMP_FILE")
    if [[ ! -z "$FILE_DIFFERENCE" ]]; then
        info "Proposed changes to ${BASHRC_FILE}:"
        diff --color -u "$BASHRC_FILE" "$BASHRC_TEMP_FILE"
	    prompt_yn "Should I make the above changes to your .bashrc?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            mv -v "$BASHRC_TEMP_FILE" "$BASHRC_FILE" > /dev/null 
        else
            prompt_yn "Okay, I will abort the bashrc changes and end the program. Should I keep a backup of the proposed changes for you to look at?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Keeping proposed changes at ${BASHRC_TEMP_FILE}."
            else
                rm -f "${BASHRC_TEMP_FILE}" # remove temp file
            fi
            exit 0
        fi
    else
        info "No changes were made to the .bashrc file."
        rm -f "${BASHRC_TEMP_FILE}"
    fi

    # Also install my ~/.vimrc settings
    if [ "$NON_JABA_SETTINGS_INSTALL" -eq 1 ]; then
        prompt_yn "Should I also install Jaeden's vimrc settings?"
        info "Setting up vim settings in ${VIMRC_FILE}..."
        if [[ -f "$VIMRC_TEMP_FILE" ]]; then
            info "Temp file ${VIMRC_TEMP_FILE} already exists, likely left over from a previous failed installation."
            prompt_yn "Would you like to remove it?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Okay, removing temporary file."
                    rm -f "${VIMRC_TEMP_FILE}"
            else
                    error "Please check and remove ${VIMRC_TEMP_FILE} manually before proceeding with installation/reinstallation."
            fi
        fi
        if [[ ! -e "$VIMRC_FILE" ]]; then
            info "Creating ${VIMRC_FILE} (it did not exist)."
            touch "$VIMRC_FILE" || { error "Failed to create ${VIMRC_FILE}"; }
        fi
        cp "$VIMRC_FILE" "$VIMRC_TEMP_FILE" > /dev/null || { error "Failed to create temporary copy of vimrc file."; }
        remove_block_between_markers "$VIMRC_TEMP_FILE" "$JABA_VARIABLES_STRING_VIM" "$JABA_VARIABLES_ENDSTRING_VIM" || exit 1
        {
            printf "${JABA_VARIABLES_STRING_VIM}\n"
            printf "syntax on\n"
            printf "colorscheme retrobox\n"
            printf "set t_Co=256\n"
            printf "set mouse=a\n"
            printf "set ttymouse=sgr\n"
            printf "${JABA_VARIABLES_ENDSTRING_VIM}\n"
        } >> "${VIMRC_TEMP_FILE}"
        FILE_DIFFERENCE=$(diff -u "$VIMRC_FILE" "$VIMRC_TEMP_FILE")
        if [[ ! -z "$FILE_DIFFERENCE" ]]; then
            info "Proposed changes to ${VIMRC_FILE}:"
            diff --color -u "$VIMRC_FILE" "$VIMRC_TEMP_FILE"
            prompt_yn "Should I make the above changes to your .vimrc?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                mv -v "$VIMRC_TEMP_FILE" "$VIMRC_FILE" > /dev/null 
            else
                prompt_yn "Okay, I will abort the vimrc changes and end the program. Should I keep a backup of the proposed changes for you to look at?"
                if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                    info "Keeping proposed changes at ${VIMRC_TEMP_FILE}."
                else
                    rm -f "${VIMRC_TEMP_FILE}" # remove temp file
                fi
                exit 0
            fi
        else
            info "No changes were made to the .vimrc file."
            rm -f "${VIMRC_TEMP_FILE}"
        fi
    fi
fi

##############################
### setup git submodules
prompt_yn "Set up submodules?"
if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
    info "Setting up git submodules..."
    git submodule update --init --recursive
    git config --global push.recurseSubmodules on-demand
    git config --global submodule.recurse true

    ### > setup GIZ
    prompt_yn "Set up GIZ?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        info "Setting up GIZ submodule..."
        (
            cd scripts/giz
            git checkout main
            bash setup.sh || exit 1
        )
    else
        info "Skipping GIZ setup."
    fi

    ### > setup SKI
    prompt_yn "Set up SKI?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        info "Setting up SKI submodule..."
        (
            cd scripts/ski
            git checkout main
            bash setup.sh || exit 1
        )
    else
        info "Skipping SKI setup."
    fi

    ### > setup CLD
    prompt_yn "Set up CLD?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        info "Setting up CLD submodule..."
        (
            cd scripts/cld
            git checkout main
            bash setup.sh || exit 1
        )
    else
        info "Skipping CLD setup."
    fi

else
    info "Skipping git submodule setup."
fi
##############################

info "Setup complete."
exit 0
