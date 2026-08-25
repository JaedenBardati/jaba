#!/usr/bin/env bash
# This script handles setting up a tmux session for a remote server, split into local, ssh, and stfp panes. 
# Must have tmux already installed (via e.g. brew install tmux).
# Usage : sxh [server_host]  (e.g., sxh jbardat@frontera.tacc.utexas.edu, or sxh frontera, if setup in ~/.ssh/config)
# Jaeden Bardati 2026 (jbardati@caltech.edu)

SERVER="$1"
TMUX_SESSION_NAME="sxh"
TMUX_WINDOW_NAME="${SERVER}"
LOCAL_HOSTNAME="$(hostname -f)"

info()   { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()   { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
prompt() { printf -v p "\033[1;36m[PROMPT]\033[0m ${1}: "; read -p "$p" "$2"; }
prompt_yn() { prompt "$1 [y/n]" YN; YN=$(echo "$YN" | tr -d ' ' | tr '[:upper:]' '[:lower:]'); } 

# basic checks
info "This script will setup a ssh/sftp connection to a server host in a tmux window.";
[ -z "$SERVER" ] && error "Missing server argument. Usage: sxh [SERVER_HOST]";  # check if server argument is provided
command -v tmux >/dev/null || error "You must install tmux first.";             # check if tmux is installed
command -v ssh >/dev/null || error "You must install ssh first.";               # check if ssh is installed

# set up session
if tmux ls >/dev/null 2>&1; then
    info "Here's a list of your current tmux sessions:"
    tmux ls
fi

if ! tmux has-session -t ${TMUX_SESSION_NAME} 2>/dev/null; then
    info "You don't have a \"${TMUX_SESSION_NAME}\" session open yet, so I'll make one for you..."
    tmux new-session -d -s "${TMUX_SESSION_NAME}" -n "${TMUX_WINDOW_NAME}"
    
    # tmux session options
    tmux set -g default-terminal "tmux-256color"
    tmux set-option -t "${TMUX_SESSION_NAME}" mouse on
    tmux bind-key -T copy-mode MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "pbcopy"
    tmux bind-key -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "pbcopy"
    export TERM=screen-256color

    # alternate copy-mode 
    #tmux bind-key -T copy-mode MouseDragEnd1Pane send-keys -X copy-pipe-no-clear "pbcopy"
    #tmux bind-key -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-no-clear "pbcopy"
    #tmux bind-key -T copy-mode MouseDown1Pane send-keys -X cancel \; select-pane
    #tmux bind-key -T copy-mode-vi MouseDown1Pane send-keys -X cancel \; select-pane
    #tmux bind-key -T copy-mode Any send-keys -X cancel \; select-pane \; send-keys -M
    #tmux bind-key -T copy-mode-vi Any send-keys -X cancel \; select-pane \; send-keys -M
    #tmux bind-key -T copy-mode Escape send-keys -X cancel
    #tmux bind-key -T copy-mode-vi Escape send-keys -X cancel

else
    # set up window
    if tmux list-windows -t ${TMUX_SESSION_NAME} >/dev/null 2>&1; then 
        info "Here's a list of your windows for the existing \"${TMUX_SESSION_NAME}\" tmux session:"
        tmux list-windows -t ${TMUX_SESSION_NAME}
    fi

    if ! tmux select-window -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}" &>/dev/null; then
        info "Could not find a \"${TMUX_WINDOW_NAME}\" window in that session, so I'll make one for you..."
        tmux new-window -t "${TMUX_SESSION_NAME}" -n "${TMUX_WINDOW_NAME}"
    fi
fi

# set up panes
SETUP_SSH_SFTP=1
CURRENT_NPANES="$(tmux list-panes -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}" | wc -l)"
if [ ${CURRENT_NPANES} -le 1 ]; then
    info "Setting up the desired pane structure..."

    # make splits
    tmux split-window -h -l 62% -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}"
    tmux select-pane -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.0"
    tmux split-window -v -l 38% -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.0"

    # name them
    tmux set-option -t "${TMUX_SESSION_NAME}" pane-border-status top
    tmux select-pane -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.1" -T "Local Terminal (${LOCAL_HOSTNAME})"
    tmux select-pane -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.0" -T "SFTP Terminal (${SERVER})"
    tmux select-pane -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.2" -T "SSH Terminal (${SERVER})"
elif [ ${CURRENT_NPANES} -ne 3 ]; then
    warn "I don't recognize the pane structure (there's ${CURRENT_NPANES} panes), so I'll avoid running any commands to set up ssh/sftp."
    SETUP_SSH_SFTP=0
fi

# set up ssh/sftp
if [ ${SETUP_SSH_SFTP} -eq 1 ]; then
    # setup ssh
    PANE_PID=$(tmux display-message -p -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.2" '#{pane_pid}')
    if ! pgrep -P "$PANE_PID" -f '^(ssh|sftp)' > /dev/null; then
        info "It doesn't look like you're running SSH in this terminal, so I'll get that going..."
        tmux send-keys -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.2" "ssh ${SERVER}" Enter
    fi

    CONTROL_MASTER=$(ssh -G "${SERVER}" | awk '/controlmaster/ {print $2}') # is multiplexing allowed?
    SOCKET_PATH=$(ssh -G "${SERVER}" | awk '/controlpath/ {print $2}')
    SOCKET_PATH="${SOCKET_PATH/#\~/$HOME}"

    # setup sftp
    PANE_PID=$(tmux display-message -p -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.0" '#{pane_pid}')
    if ! pgrep -P "$PANE_PID" -f '^(ssh|sftp)' > /dev/null; then
        info "It doesn't look like you're running SFTP in this terminal, so I'll get that going..."
        if [[ "$CONTROL_MASTER" == "auto" || "$CONTROL_MASTER" == "yes" ]] && [ -n "$SOCKET_PATH" ]; then
            # if multiplexing is allowed, wait for the socket to be created before starting sftp
            tmux send-keys -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.0" \
                "clear; echo 'Waiting for multiplex master socket to initialize...'; clear; while [ ! -S '$SOCKET_PATH' ]; do sleep 1; done; sftp ${SERVER}" Enter
        else
            # if multiplexing is not allowed, just start sftp when ssh does
            tmux send-keys -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}.0" \
                "clear; echo 'Waiting for you to clear MFA in the SSH pane...'; clear; while tmux list-panes -t '${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}' -F '#{pane_index} #{pane_current_command}' | grep -E '^2 ssh' >/dev/null; do sleep 1; done; sftp ${SERVER}" Enter
        fi
    fi
fi

# attach
info "Attaching to session ..."
tmux attach-session -t "${TMUX_SESSION_NAME}" \; select-window -t "${TMUX_WINDOW_NAME}" \; select-pane -t "2"
# prompt_yn "Would you like to attach to the tmux session?" YN;
# if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
#     info "Attaching to session ..."
#     tmux attach-session -t "${TMUX_SESSION_NAME}" \; select-window -t "${TMUX_WINDOW_NAME}"
# fi

# remove session or window if desired
prompt_yn "Would you like to remove this window?" YN
if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
    tmux respawn-pane -k -t :.
    tmux kill-window -t "${TMUX_SESSION_NAME}:${TMUX_WINDOW_NAME}"

    if tmux ls >/dev/null 2>&1; then
        prompt_yn "Would you like to remove to the whole session (and all windows)?" YN;
        if [[ "$YN" == "y" || "$YN" == "yes" ]]; then
            tmux list-panes -a -s -F '#{pane_id}' | xargs -I {} sh -c '
    tmux send-keys -t "$1" C-c Enter "exit" Enter
    ' _ {} && tmux kill-session -t "${TMUX_SESSION_NAME}"
        fi
    fi
fi

if tmux ls >/dev/null 2>&1; then
    info "Here's a final list of your current tmux sessions:"
    tmux ls
fi

info "All done, exiting now..."
exit 0
