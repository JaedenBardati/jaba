#!/usr/bin/env bash
# This script will help you setup ssh keys for a new cluster.
# Jaeden Bardati 2026 (jbardati@caltech.edu)

YOUR_SSH_DIR="${HOME}/.ssh/"

info()   { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()   { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
prompt() { printf -v p "\033[1;36m[PROMPT]\033[0m ${1}: "; read -p "$p" "$2"; }
prompt_yn() { prompt "$1 [y/n]" YN; YN=$(echo "$YN" | tr -d ' ' | tr '[:upper:]' '[:lower:]'); } 

info "This script will guide you through the SSH setup for a new cluster." # TODO or github

# go to your ssh directory
if [ ! -d $YOUR_SSH_DIR ]; then
    warn "Your ssh directory does not seem to exist at ${YOUR_SSH_DIR}, where it should."
    prompt_yn "Would you like me to make it?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        mkdir -p $YOUR_SSH_DIR
    else
        error "You need an SSH directory to continue this program."
    fi
fi
cd $YOUR_SSH_DIR
info "Using ${YOUR_SSH_DIR} as working directory."

# start with looking any SSH keys you have already
PRIVATE_KEY=""
PUBLIC_KEY=""
if ls *.pub > /dev/null 2>&1; then
    info "Found existing SSH keys:"
    ls *.pub
    prompt_yn "Would you like to use one of these SSH keys?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        while true; do
            for f in *.pub; do DEFAULT_PUBLIC_KEY="$f"; break; done
            prompt "Please enter a public key above that you would like to use (default is ${DEFAULT_PUBLIC_KEY})" PUBLIC_KEY
            [[ "$PUBLIC_KEY" == "" || "$PUBLIC_KEY" == " " ]] && PUBLIC_KEY="${DEFAULT_PUBLIC_KEY}"
            [ ! -f "$PUBLIC_KEY" ] && { warn "The key you entered (${PUBLIC_KEY}) does not exist."; continue; }
            PRIVATE_KEY="${PUBLIC_KEY%.pub}"
            [ -f "$PRIVATE_KEY" ] && { break; } || { warn "The public key (${PUBLIC_KEY}) you entered exists but does not seem to have the associated private key (${PRIVATE_KEY})."; }
        done
    fi
else
    info "Could not find any keys in your SSH directory."
fi

# make key if needed
if [[ "$PUBLIC_KEY" == "" || "$PRIVATE_KEY" == "" ]]; then
    info "Since we need an SSH key to use, I'll attempt to make one for you."
    DONE=0
    while [ "$DONE" -eq 0 ]; do
        DEFAULT_KEY_TYPE="ed25519"
        prompt "Please enter a key type that you would like to make (default is ${DEFAULT_KEY_TYPE})" KEY_TYPE ;
        [[ "$KEY_TYPE" == "" || "$KEY_TYPE" == " " ]] && KEY_TYPE="${DEFAULT_KEY_TYPE}"
        
        DEFAULT_KEY_NAME="id_${KEY_TYPE}"
        prompt "Please enter a key name that you would like to make (default is ${DEFAULT_KEY_NAME})" KEY_NAME ;
        [[ "$KEY_NAME" == "" || "$KEY_NAME" == " " ]] && KEY_NAME="${DEFAULT_KEY_NAME}"

        DEFAULT_KEY_COMMENT="$(whoami)@$(hostname)"
        prompt "Please enter a key comment that you would like to make (default is ${DEFAULT_KEY_COMMENT})" KEY_COMMENT ;
        [[ "$KEY_COMMENT" == "" || "$KEY_COMMENT" == " " ]] && KEY_COMMENT="${DEFAULT_KEY_COMMENT}"
        
        ssh-keygen -t "$KEY_TYPE" -f "./$KEY_NAME" -C "$KEY_COMMENT" && DONE=1 || warn "Error with keygen, try again."
    done
    PRIVATE_KEY=${KEY_NAME}
    PUBLIC_KEY=${KEY_NAME}.pub

    # check keys exist
    if [[ ! -f "${YOUR_SSH_DIR}/${PRIVATE_KEY}" || ! -f "${YOUR_SSH_DIR}/${PUBLIC_KEY}" ]]; then
        error "Something went wrong with the keygen and one or both of these keys were not created (private key location: ${YOUR_SSH_DIR}/${PRIVATE_KEY}, public key location: ${YOUR_SSH_DIR}/${PUBLIC_KEY})."
    fi
fi

# add server if needed
prompt_yn "Would you like to setup a new server host?"
if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
    USERNAME=""
    HOSTNAME=""
    HOSTNAME_SHORTCUT=""
    DONE=0
    while [ "$DONE" -eq 0 ]; do
        prompt "Please enter your username for the server (e.g., jbardati)" USERNAME ;
        [[ "$USERNAME" == "" || "$USERNAME" == " " ]] && { warn "You need an actual input for this one, try again."; continue; }

        prompt "Please enter the host name server address you would like to setup (e.g., login.hpc.caltech.edu)" HOSTNAME ;
        [[ "$HOSTNAME" == "" || "$HOSTNAME" == " " ]] && { warn "You need an actual input for this one, try again."; continue; }

        prompt "Please enter a shortcut name for the server that you can call later with 'ssh NAME' (e.g., caltech-hpc)" HOSTNAME_SHORTCUT ;
        [[ "$HOSTNAME_SHORTCUT" == "" || "$HOSTNAME_SHORTCUT" == " " ]] && { info "Defaulting host shortcut name to ${HOSTNAME}"; HOSTNAME_SHORTCUT=${HOSTNAME}; }

        if [ -f "$YOUR_SSH_DIR/config" ] && grep -qw -e "HostName ${HOSTNAME}" "$YOUR_SSH_DIR/config"; then
            warn "The host name you specified already exists in your file."
            prompt_yn "Would you like to use this pre-existing config (you must be sure the other parameters are the same as those you entered)?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Okay, I'll use this config as entered."
                DONE=1
                break;
            fi
            prompt_yn "Would you like to manually remove this config now (you must know what you are doing)?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Opening the file in 3 seconds. Please delete the full relevant block (with HostName ${HOSTNAME})."
                sleep 3
                vim $YOUR_SSH_DIR/config
                grep -qw -e "HostName ${HOSTNAME}" "$YOUR_SSH_DIR/config" && error "You did not seem to have deleted the relevant hostname..." # not a comprehensive check, be careful here
            else
                info "Since you don't want to use this pre-existing config or manually remove it, you will have to enter a different hostname."
                continue;
            fi
        fi

        if [ -f "$YOUR_SSH_DIR/config" ] && grep -qw -e "Host ${HOSTNAME_SHORTCUT}" "$YOUR_SSH_DIR/config"; then
            warn "The host shortcut name you specified already exists in your file. Please try a different shortcut name."
            continue;
        fi

        CONFIG_STR=$(cat <<EOF

Host ${HOSTNAME_SHORTCUT}
    HostName ${HOSTNAME}
    User ${USERNAME}
EOF
        )
        info "Here's what I'm planning on adding to your config file: $CONFIG_STR"
        prompt_yn "Should I add this to your config file?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            info "Great, appending it to the end of the config file."
            printf "%s\n" "$CONFIG_STR" >> "$YOUR_SSH_DIR/config"
            DONE=1
        else
            warn "Then you'll have to re-enter the info I'm afraid."
            continue;
        fi

        prompt_yn "Do you want to make any more edits to the config file?"
        [[ "$YN" == "y" || "$YN" == "yes" ]] && vim "$YOUR_SSH_DIR/config" ;

    done

    # set it up on the server side
    prompt_yn "Do you want to setup ssh keys on the server side?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        info "Now I'll set up your ssh key on the server side."
        if command -v ssh-copy-id >/dev/null 2>&1; then
            ssh-copy-id -i "${YOUR_SSH_DIR}/${PUBLIC_KEY}" "${HOSTNAME_SHORTCUT}" || error "Something went wrong..."
        else
            error "I currently don't have the capacity to set up the server side without ssh-copy-id installed. Please install it (e.g. homebrew on mac or )."
        fi
    fi

    # test the connection now
    prompt_yn "Would you like to test the connection now?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        ssh "${HOSTNAME_SHORTCUT}" || error "Something went wrong..."
    fi
fi

# add github key
#prompt_yn "Would you like to setup github keys?" # TODO

# ending phrase
info "That's all. Exiting now..."

