#!/usr/bin/env bash
# This script will get a file from a remote server and sync a local copy with the remote copy to edit locally.
# It is highly recommended to set up ssh agent keys before running this script, so that you don't have to enter your password/authentication every time.
# Jaeden Bardati 2026 (jbardati@caltech.edu)

eval $(ssh-agent -s) > /dev/null  # activate ssh agent

info()   { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()   { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
prompt() { printf -v p '\033[1;36m[PROMPT]\033[0m %s: ' "$1"; read -r -p "$p" "$2"; }
prompt_yn() { prompt "$1 [y/n]" YN; YN=$(echo "$YN" | tr -d ' ' | tr '[:upper:]' '[:lower:]'); } 

print_usage() {
    echo "Usage: $0 <remote_path> [local_file]"
    echo "Example: $0 my-ssh-alias:/remote/path/"
    exit 1
}

REMOTE="${1}"
LOCAL="${2}"

INTERVAL=1
OPEN="open" # works for mac, change to e.g. vim on linux, leaving empty will not open the file automatically
OVERWRITE_LOCAL_WITH_REMOTE=1

# Validate input arguments
if [ -z "$REMOTE" ]; then
    print_usage
fi
if [ "${REMOTE#*:}" = "$REMOTE" ]; then
    error "Remote path must be specified in the format user@host:/path/to/file"
fi
remote_host="${REMOTE%:*}"
remote_path="${REMOTE#*:}"
if ! ssh "$remote_host" "test -e '$remote_path'" >/dev/null 2>&1; then
    error "Remote path '$remote_path' on host '$remote_host' does not exist or is not accessible."
fi

if [ -z "$LOCAL" ]; then
    LOCAL="${REMOTE##*:}"
    LOCAL="./$(basename "$LOCAL")"
    info "No local source provided, defaulting to '$LOCAL'."
    OVERWRITE_LOCAL_WITH_REMOTE=1
fi
if [ -f "$LOCAL" ]; then
    prompt_yn "Local file '$LOCAL' already exists. Do you want to overwrite it with the remote file?"
    if [ "$YN" != "y" ] && [ "$YN" != "yes" ]; then
        OVERWRITE_LOCAL_WITH_REMOTE=0
    fi
fi
if [ -d "$LOCAL" ]; then
    prompt_yn "Local directory '$LOCAL' already exists. Do you want to overwrite it with the remote directory?"
    if [ "$YN" != "y" ] && [ "$YN" != "yes" ]; then
        OVERWRITE_LOCAL_WITH_REMOTE=0
    fi
fi
if [ "${LOCAL#*:}" != "$LOCAL" ]; then
    warn "Local path entered was actually a remote path."
    print_usage
fi
if [ "$OVERWRITE_LOCAL_WITH_REMOTE" -eq 1 ]; then
    info "Copying remote file to local as temporary file..."
    rsync -avz "$REMOTE" "$LOCAL" >/dev/null || error "Failed to copy remote file to local."
fi

# prepare temporary files
LAST_RUN_FILE="/tmp/edit_sync_last_run.$$"
touch "$LAST_RUN_FILE"

if [ "$OVERWRITE_LOCAL_WITH_REMOTE" -eq 1 ]; then
    trap 'rm -f "$LAST_RUN_FILE"; rm -rf "$LOCAL"; exit 0' INT TERM EXIT
else
    trap 'rm -f "$LAST_RUN_FILE"; exit 0' INT TERM EXIT
fi

# Open file and start the sync loop
info "Watching '$LOCAL' for changes... Press [CTRL+C] to stop."
if [ -n "$OPEN" ] && [ -f "$LOCAL" ]; then
    eval "${OPEN} \"$LOCAL\"" # open the local file with the specified command
fi
while true; do
    CHANGE_DETECTED=0
    if [ -d "$LOCAL" ]; then
        # Directory: check if any file inside is newer than the reference file
        if [ -n "$(find "$LOCAL" -type f -newer "$LAST_RUN_FILE" -print -quit)" ]; then
            CHANGE_DETECTED=1
        fi
    else
        # Single file: check if this specific file is newer than the reference file
        if [ -n "$(find "$LOCAL" -newer "$LAST_RUN_FILE" -print -quit)" ]; then
            CHANGE_DETECTED=1
        fi
    fi

    # Do the sync if a local change occurred
    if [ "$CHANGE_DETECTED" -eq 1 ]; then
        info " Syncing: $LOCAL -> $REMOTE"
        rsync -avz "$LOCAL" "$REMOTE"
        touch "$LAST_RUN_FILE"
    fi
    
    sleep "$INTERVAL"
done