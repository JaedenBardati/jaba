#!/usr/bin/env bash
# This script will help you setup SSH (e.g., SSH keys, server nicknames).
# Jaeden Bardati 2026 (jbardati@caltech.edu)

YOUR_SSH_DIR="$HOME/.ssh"
PUBLIC_KEY=""
PRIVATE_KEY=""

info()   { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()   { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
prompt() { printf -v p '\033[1;36m[PROMPT]\033[0m %s: ' "$1"; read -r -p "$p" "$2"; }
prompt_yn() { prompt "$1 [y/n]" YN; YN=$(echo "$YN" | tr -d ' ' | tr '[:upper:]' '[:lower:]'); } 

get_keys() {
    if [[ -n "$PUBLIC_KEY" && -n "$PRIVATE_KEY" ]]; then
        return 0
    fi
    cd "$YOUR_SSH_DIR" # double check that we're in the right directory (this function assumes it)
    local _PRIVATE_KEY=""
    local _PUBLIC_KEY=""

    # look for any SSH keys you have already have
    if ls *.pub > /dev/null 2>&1; then
        info "Found existing SSH keys:"
        ls *.pub
        prompt_yn "Would you like to use one of these SSH keys?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            local _DEFAULT_PUBLIC_KEY
            for f in *.pub; do _DEFAULT_PUBLIC_KEY="$f"; break; done
            while true; do
                _PRIVATE_KEY=""
                _PUBLIC_KEY=""
                prompt "Please enter a public key above that you would like to use (default is ${_DEFAULT_PUBLIC_KEY})" _PUBLIC_KEY
                [[ -z "$(printf '%s' "$_PUBLIC_KEY" | tr -d '[[:space:]]')" ]] && _PUBLIC_KEY="${_DEFAULT_PUBLIC_KEY}"
                _PUBLIC_KEY="${_PUBLIC_KEY%.pub}.pub" # ensure it ends with .pub
                [ ! -f "$_PUBLIC_KEY" ] && { warn "The public key you entered (${_PUBLIC_KEY%.pub}) does not exist."; continue; }
                _PRIVATE_KEY="${_PUBLIC_KEY%.pub}"
                [ -f "$_PRIVATE_KEY" ] && { break; } || { warn "The public key (${_PUBLIC_KEY}) you entered exists but does not seem to have the associated private key (${_PRIVATE_KEY})."; }
            done
        fi
    else
        info "Could not find any keys in your SSH directory."
    fi

    # make SSH keys if needed
    if [[ "$_PUBLIC_KEY" == "" || "$_PRIVATE_KEY" == "" ]]; then
        info "Since we need an SSH key to use, I'll attempt to make one for you."
        local _DONE=0
        local _KEY_TYPE=""
        local _KEY_NAME=""
        local _KEY_COMMENT=""
        while [ "$_DONE" -eq 0 ]; do
            local _DEFAULT_KEY_TYPE="ed25519"
            prompt "Please enter a key type that you would like to make (default is ${_DEFAULT_KEY_TYPE})" _KEY_TYPE ;
            [[ -z "$(printf '%s' "$_KEY_TYPE" | tr -d '[[:space:]]')" ]] && _KEY_TYPE="${_DEFAULT_KEY_TYPE}"
            
            local _DEFAULT_KEY_NAME="id_${_KEY_TYPE}"
            prompt "Please enter a key name that you would like to make (default is ${_DEFAULT_KEY_NAME})" _KEY_NAME ;
            [[ -z "$(printf '%s' "$_KEY_NAME" | tr -d '[[:space:]]')" ]] && _KEY_NAME="${_DEFAULT_KEY_NAME}"

            local _DEFAULT_KEY_COMMENT="$(whoami)@$(hostname -f)"
            prompt "Please enter a key comment that you would like to make (default is ${_DEFAULT_KEY_COMMENT})" _KEY_COMMENT ;
            [[ -z "$(printf '%s' "$_KEY_COMMENT" | tr -d '[[:space:]]')" ]] && _KEY_COMMENT="${_DEFAULT_KEY_COMMENT}"
            
            ssh-keygen -t "$_KEY_TYPE" -f "${_KEY_NAME}" -C "$_KEY_COMMENT" && _DONE=1 || warn "Error with keygen, try again."
        done
        _PRIVATE_KEY="${_KEY_NAME}"
        _PUBLIC_KEY="${_KEY_NAME}.pub"
        chmod 600 "$_PRIVATE_KEY"
        chmod 644 "$_PUBLIC_KEY"
    fi

    # save full path in global variables and double check keys actually exist 
    PRIVATE_KEY="${YOUR_SSH_DIR}/${_PRIVATE_KEY}"
    PUBLIC_KEY="${YOUR_SSH_DIR}/${_PUBLIC_KEY}"
    if [[ ! -f "${PRIVATE_KEY}" || ! -f "${PUBLIC_KEY}" ]]; then
        error "Something went wrong with the keygen and one or both of these keys were not created (private key location: ${YOUR_SSH_DIR}/${_PRIVATE_KEY}, public key location: ${YOUR_SSH_DIR}/${_PUBLIC_KEY})."
    fi
}

info "This script will guide you through the SSH setup."
command -v ssh >/dev/null || error "You must install ssh first."; # check if ssh is installed

# make sure your ssh directory exists (and go there)
if [ ! -d "$YOUR_SSH_DIR" ]; then
    warn "Your ssh directory does not seem to exist at ${YOUR_SSH_DIR}, where it should."
    prompt_yn "Would you like me to make it?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        mkdir -p "$YOUR_SSH_DIR"
        chmod 700 "$YOUR_SSH_DIR"
    else
        error "You need an SSH directory to continue this program. Edit the variable YOUR_SSH_DIR in this script to point this program somewhere else."  
    fi
fi
cd "$YOUR_SSH_DIR"
info "Using ${YOUR_SSH_DIR} as working directory."

# make sure config file exists and make backup of config file if desired
if [ ! -f "${YOUR_SSH_DIR}/config" ]; then # if no config file exists, make one
    if [ ! -f "${YOUR_SSH_DIR}/config.bak" ]; then 
        info "Your ssh config file does not seem to exist at ${YOUR_SSH_DIR}/config, so I'll make a blank one for you."
        touch "${YOUR_SSH_DIR}/config"
    else
        prompt_yn "It looks like you have a backup of your SSH config file at ${YOUR_SSH_DIR}/config.bak, but no config file. Would you like to restore the backup?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            cp "${YOUR_SSH_DIR}/config.bak" "${YOUR_SSH_DIR}/config"
            info "I restored your SSH config file from ${YOUR_SSH_DIR}/config.bak"
        else
            warn "Okay, I'll just make a blank config file."
            touch "${YOUR_SSH_DIR}/config"
        fi
    fi
elif [ ! -f "${YOUR_SSH_DIR}/config.bak" ] && grep -q '[^[:space:]]' "${YOUR_SSH_DIR}/config" ; then  # if no backup exists (and config file does and is not empty), make one
    prompt_yn "You don't have a backup of your SSH config file. Would you like me make one?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        cp "${YOUR_SSH_DIR}/config" "${YOUR_SSH_DIR}/config.bak"
        info "I made a backup of your SSH config file at ${YOUR_SSH_DIR}/config.bak"
    fi
fi
if [ -f "${YOUR_SSH_DIR}/config" ] && [ -f "${YOUR_SSH_DIR}/config.bak" ] && ! cmp -s "${YOUR_SSH_DIR}/config" "${YOUR_SSH_DIR}/config.bak" ; then # if both config file and backup exist and differ
    info "You have a backup SSH config file at ${YOUR_SSH_DIR}/config.bak that is different from your actual config file. Here's what changed from the backup:"
    diff -u "${YOUR_SSH_DIR}/config.bak" "${YOUR_SSH_DIR}/config"
    prompt_yn "Would you like to restore the backup?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        cp "${YOUR_SSH_DIR}/config.bak" "${YOUR_SSH_DIR}/config"
        info "I restored your SSH config file from ${YOUR_SSH_DIR}/config.bak"
    else
        prompt_yn "Would you like to overwrite your existing backup with your current SSH config file instead?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            cp "${YOUR_SSH_DIR}/config" "${YOUR_SSH_DIR}/config.bak"
            info "I made a backup of your SSH config file at ${YOUR_SSH_DIR}/config.bak"
        fi
    fi
fi

# add new server if desired
DONE_BIG_LOOP=1
while [ "$DONE_BIG_LOOP" -eq 1 ]; do
    GREP_OUTPUT="$(grep "^Host " "${YOUR_SSH_DIR}/config" | cut -d' ' -f2)"
    if [ -n "${GREP_OUTPUT}" ]; then
        info "You currently have the following servers set up in your SSH config file:"
        echo "$GREP_OUTPUT"
    else
        info "You currently do not have any servers set up in your SSH config file."
    fi

    prompt_yn "Would you like to set up a new server host?" #(or delete an existing one)
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        HOST_NICKNAME=""
        HOST_NAME=""
        USER_NAME=""
        while true; do
            while true; do
                prompt "Please enter a nickname for the server that you can call later with \"ssh name\" (e.g., frontera)" HOST_NICKNAME ;
                [[ -z "$(printf '%s' "$HOST_NICKNAME" | tr -d '[[:space:]]')" ]] || { break; } && { warn "You need an actual input."; }
            done
            if [ -f "${YOUR_SSH_DIR}/config" ] && grep -Fqw -e "Host ${HOST_NICKNAME}" "${YOUR_SSH_DIR}/config"; then
                warn "The host nickname you specified already exists in your file. Here's the relevant block in your config file:" 
                awk -v host="${HOST_NICKNAME}" '$1=="Host" && $2==host {p=1; print; next} $1=="Host" {p=0} p' "${YOUR_SSH_DIR}/config"

                prompt_yn "Would you like to continue using this pre-existing config instead?"
                if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                    USER_NAME=$(ssh -G "${HOST_NICKNAME}" | awk '$1 == "user" {print $2}')
                    HOST_NAME=$(ssh -G "${HOST_NICKNAME}" | awk '$1 == "hostname" {print $2}')
                    info "Okay, I'll use the existing config with username ${USER_NAME} and hostname ${HOST_NAME}."
                    break;
                else
                    prompt_yn "Would you like to remove this whole config now (and possibly replace it with one you specify after)?"
                    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                        info "Removing the existing config and adding the new one."
                        tmp_file=$(mktemp)
                        cp "${YOUR_SSH_DIR}/config" "${tmp_file}"
                        awk -v host="$HOST_NICKNAME" '
    $0 == "Host " host { skip=1; next }
    skip && /^Host / { skip=0 }
    !skip { print }
' "${tmp_file}" > "${YOUR_SSH_DIR}/config"
                        grep -Fqw -e "Host ${HOST_NICKNAME}" "${YOUR_SSH_DIR}/config" && (mv "${tmp_file}" "${YOUR_SSH_DIR}/config"; error "I did not seem to have deleted the relevant host nickname (but I restored it)...")
                        
                        info "I made the following changes to your config file:"
                        diff -u "${tmp_file}" "${YOUR_SSH_DIR}/config"
                        
                        prompt_yn "Should I keep these changes?" YN;
                        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                            rm "${tmp_file}"
                        else
                            mv "${tmp_file}" "${YOUR_SSH_DIR}/config"
                            warn "I restored your config file to the previous version. You'll have to re-enter the info I'm afraid (or leave via Ctrl+C)."
                            continue;
                        fi
                        
                        prompt_yn "Do you want to replace it with a new config now?" YN;
                        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                            info "Replacing the config with a new one..."
                        else
                            warn "Okay, I will not replace the config (it will remain deleted)."
                            break;
                        fi
                    else
                        info "Since you don't want to use this pre-existing config or manually remove it, you will have to enter a different nickname."
                        continue;
                    fi
                fi
            fi

            # set up the new config
            while true; do
                prompt "Please enter your username for the new server config (e.g., jbardati)" USER_NAME ;
                [[ -z "$(printf '%s' "$USER_NAME" | tr -d '[[:space:]]')" ]] || { break; } && { warn "You need an actual input."; }
            done
            while true; do
                prompt "Please enter the host name server address (e.g., frontera.tacc.utexas.edu)" HOST_NAME ;
                [[ -z "$(printf '%s' "$HOST_NAME" | tr -d '[[:space:]]')" ]] || { break; } && { warn "You need an actual input."; }
            done

            CONFIG_STR=$(cat <<EOF
Host ${HOST_NICKNAME}
    HostName ${HOST_NAME}
    User ${USER_NAME}
EOF
            )

            prompt_yn "Would you like to setup the ssh keys on the server now?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                SETUP_KEYS=1
                get_keys

                prompt_yn "Would you like to add this key to the ssh-agent for persistent use in this terminal session? (recommended)"
                if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                    info "Adding the key to the ssh-agent now."
                    if eval "$(ssh-agent -s)" >/dev/null; then
                        if [[ "$(uname)" == "Darwin" ]]; then
                            ssh-add --apple-use-keychain ${YOUR_SSH_DIR}/${_PRIVATE_KEY} || warn "Could not add the key to the ssh-agent. Continuing anyway, but you should do this later."
                        else
                            ssh-add ${YOUR_SSH_DIR}/${_PRIVATE_KEY} || warn "Could not add the key to the ssh-agent. Continuing anyway, but you should do this later."
                        fi
                    else
                        warn "Could not start the ssh-agent. Continuing anyway, but you should do this later."
                    fi
                fi

                prompt_yn "Do you want to add the key to the ssh-agent permanently? (useful but can be a security risk unless you set the private key explicitly below)";
                if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                    CONFIG_STR="${CONFIG_STR}
    AddKeysToAgent yes"
                fi

                if [[ "$(uname)" == "Darwin" ]]; then
                    prompt_yn "Do you want to use the keychain to store the key? (recommended for macOS)"
                    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                        CONFIG_STR="${CONFIG_STR}
    UseKeychain yes"
                    fi
                fi

                prompt_yn "Do you want to point the config to the private key directly? (recommended)"
                if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                    CONFIG_STR="${CONFIG_STR}
    IdentityFile ${PRIVATE_KEY}"

                    prompt_yn "Do you want to explicitly only allow the private key you just specified to be used in the config? (recommended)"
                    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                        CONFIG_STR="${CONFIG_STR}
    IdentitiesOnly yes"
                    fi
                fi

            else
                SETUP_KEYS=0
            fi

            prompt_yn "Do you want to explicitly disallow ssh multiplexing for this host? (not recommended, but is necessary for some servers)"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                CONFIG_STR="${CONFIG_STR}
    ControlMaster no"
            fi

            CONFIG_STR="${CONFIG_STR}
"

            info "Here's what I'm planning on adding to your config file: 
$CONFIG_STR"
            prompt_yn "Should I add this to your config file?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Great, appending it to the front of the config file."
                tmp_file=$(mktemp)
                printf '%s\n' "$CONFIG_STR" > "$tmp_file"
                cat "${YOUR_SSH_DIR}/config" >> "$tmp_file"
                mv "$tmp_file" "${YOUR_SSH_DIR}/config"
                break;
            else
                warn "Then you'll have to re-enter the info I'm afraid (or leave via Ctrl+C)."
                continue;
            fi
        done

        # set it up on the server side
        if [[ "$SETUP_KEYS" == "1" ]]; then
            info "Now I'll set up your key on the server side."
            if command -v ssh-copy-id >/dev/null 2>&1; then
                ssh-copy-id -i "${PUBLIC_KEY}" "${HOST_NICKNAME}" || error "It seems like ssh-copy-id failed. You will have to manually set up your public key on the server side. You can do this by copying the contents of ${PUBLIC_KEY} into the file ~/.ssh/authorized_keys on the server."
            else
                error "I currently don't have the capacity to set up the server side without ssh-copy-id installed. Please install it (e.g. homebrew on mac or )."
            fi
        fi

        # test the connection now
        prompt_yn "Would you like to test the connection to ${HOST_NICKNAME} now?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            ssh "${HOST_NICKNAME}" || error "Something went wrong..."
        fi
    else
        DONE_BIG_LOOP=0
    fi
done

# Add global multiplexing default if desired
DEFAULT_CONTROLMASTER=$(ssh -G "something-very-specific-that-i-really-really-hope-you-dont-somehow-have-a-ssh-config-setup-for" | awk '/controlmaster/ {print $2}')
if [ "${DEFAULT_CONTROLMASTER}" != "auto" ] && [ "${DEFAULT_CONTROLMASTER}" != "yes" ]; then
    info "It looks like your default ssh config does not allow multiplexing (multiplexing is generally recommended if allowed)."
    prompt_yn "Would you like to change your default ssh config to allow multiplexing?"
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        CONFIG_STR=$(cat <<EOF
Host *
    ControlMaster auto
    ControlPath ${YOUR_SSH_DIR}/sockets/%C
    ControlPersist 30m
    ServerAliveInterval 60
EOF
        )
        CONFIG_STR="${CONFIG_STR}
"
        if grep -E -q '^Host ?\*[ ]*$' "${YOUR_SSH_DIR}/config" ; then
            info "Since you already have a Host \* entry, you will have to manually merge the multiplexing options to it:"
            echo "$CONFIG_STR"
            prompt "Copy-paste the above into your config file and then press enter to vim into your config file (edit with i, save/exit with esc + :wq)..." DUMMY
            vim "${YOUR_SSH_DIR}/config" # TODO do this automatically somehow
        else
            info "I'll make the necessary directories and add the following to the end of your config file:"
            echo "$CONFIG_STR"
            mkdir -p "${YOUR_SSH_DIR}/sockets" # matching convention for ControlPath default above 
            chmod 700 "${YOUR_SSH_DIR}/sockets"
            printf '%s\n' "$CONFIG_STR" >> "${YOUR_SSH_DIR}/config"
        fi
    fi
fi

# Blacklist certain domains from using multiplexing default if desired
DEFAULT_CONTROLMASTER=$(ssh -G "something-very-specific-that-i-really-really-hope-you-dont-somehow-have-a-ssh-config-setup-for" | awk '/controlmaster/ {print $2}')
if [ "${DEFAULT_CONTROLMASTER}" == "auto" ] || [ "${DEFAULT_CONTROLMASTER}" == "yes" ]; then
    prompt_yn "Your SSH config allows multiplexing. Do you want to explicitly turn off multiplexing for certain domains? (this is generally not recommended, but is necessary for some servers)"; # use standard wildcard (e.g. *) rules
    if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
        prompt "Please enter the domain you would like to disallow multiplexing for (e.g., *.caltech.edu)" DOMAIN ;
        CONFIG_STR=$(cat <<EOF
Host ${DOMAIN}
    ControlMaster no
EOF
        )
        CONFIG_STR="${CONFIG_STR}
"
        info "I'll add the following to your config file before the first wildcard entry:"
        echo "$CONFIG_STR"
        
        prompt_yn "Do you want me to add this to your config file?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            info "Great, I'll place it before the first wildcard entry."
            tmp_file=$(mktemp)
            awk '
        !inserted && /\*/ {
            # Read every line from fd 3 and print it
            while ((getline line < "/dev/fd/3") > 0) {
                print line
            }
            inserted = 1
        }
        { print }
    ' "${YOUR_SSH_DIR}/config" 3<<< "$CONFIG_STR" > "$tmp_file" && mv "$tmp_file" "${YOUR_SSH_DIR}/config"
        else
            warn "Then you'll have to do this manually later or re-run the script."
        fi
    fi
fi

# Set up github keys if desired
prompt_yn "Would you like to setup github keys?"
if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
    get_keys

    TRY_GH=1
    while true; do
        info "Now I'll set up your key on github."
        if [[ "${TRY_GH}" == "1" ]] && command -v gh >/dev/null 2>&1; then
            info "Using gh to set up your key on github."
            gh auth login || { warn "It seems like gh auth failed. You will have to manually set up your public key."; TRY_GH=0; break; }
            gh ssh-key add "${PUBLIC_KEY}" --title "$(whoami)@$(hostname -f)" || { warn "It seems like gh ssh-key add failed. You will have to manually set up your public key."; TRY_GH=0; break; }
        else
            info "Since gh is not installed, you will have to set up your key manually with a web browser."
            info "Here are the contents of your public key:"
            cat "${PUBLIC_KEY}"
            if [[ "$(uname)" == "Darwin" ]] && [[ -n "$(command -v pbcopy)" ]]; then
                pbcopy < "${PUBLIC_KEY}"
                info "I have copied your public key to your clipboard for you."
            else
                info "I could not copy your public key to your clipboard automatically. Please copy it manually."
            fi
            info "Please paste it into your github account settings under SSH keys (https://github.com/settings/keys)."
            prompt "Press enter when you have completed this..." DUMMY
        fi

        prompt_yn "Do you want to test your github SSH connection now?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            ssh -T git@github.com 2>&1 | grep -q "successfully authenticated" && break || warn "Something went wrong with the github SSH connection test. You may not have copied the public key correctly. Let's try again."
        fi
    done
    info "Successfully set up your github SSH keys."
fi


# look at what was changed and delete backup if desired
if [ -f "${YOUR_SSH_DIR}/config" ] && [ -f "${YOUR_SSH_DIR}/config.bak" ]; then # need a backup in the first place
    if ! cmp -s "${YOUR_SSH_DIR}/config" "${YOUR_SSH_DIR}/config.bak"; then # if the config file and backup differ
        info "Here's a diff of your config file and the backup I made:"
        diff -u "${YOUR_SSH_DIR}/config.bak" "${YOUR_SSH_DIR}/config"

        prompt_yn "Would you like to restore your config file to the backup (and erase all recent changes made)?"
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            info "Restoring your config file to the backup I made."
            mv "${YOUR_SSH_DIR}/config.bak" "${YOUR_SSH_DIR}/config"
        else
            prompt_yn "Would you like to keep the (old) backup of your config file?"
            if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
                info "Okay, I will keep the backup at ${YOUR_SSH_DIR}/config.bak"
            else
                info "Okay, I will delete the backup at ${YOUR_SSH_DIR}/config.bak"
                rm "${YOUR_SSH_DIR}/config.bak"
            fi
        fi
    else 
        info "Your config file is identical to the backup I made, so I will delete it."
        rm "${YOUR_SSH_DIR}/config.bak"
    fi
fi

prompt_yn "Do you want to make any final edits to the SSH config file?";
[[ "$YN" == "y" || "$YN" == "yes" ]] && vim "${YOUR_SSH_DIR}/config" ;

# ending phrase
info "That's all. Exiting now..."
exit 0
