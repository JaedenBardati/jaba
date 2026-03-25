#!/usr/bin/env bash

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

remove_block_between_markers() {
	local target_file="$1"
	local start_marker="$2"
	local end_marker="$3"

	if [[ -z "$target_file" || ! -f "$target_file" ]]; then
		printf "[remove_block_between_markers] target file missing or invalid: %s\n" "$target_file" >&2
		return 0
	fi

	local tmp_file
	tmp_file="$(mktemp)"
	if [[ -z "$tmp_file" || ! -f "$tmp_file" ]]; then
		printf "[remove_block_between_markers] failed to create temporary file for %s\n" "$target_file" >&2
		return 1
	fi

	awk -v start="$start_marker" -v end="$end_marker" '
		BEGIN { inblock=0 }
		$0==start { inblock=1; next }
		$0==end { if (inblock) { inblock=0; next } }
		!inblock { print }
	' "$target_file" > "$tmp_file" || {
		printf "[remove_block_between_markers] awk failed for file=%s start_marker=%s end_marker=%s\n" "$target_file" "$start_marker" "$end_marker" >&2
		rm -f "$tmp_file"
		return 1
	}

	mv "$tmp_file" "$target_file" || {
		printf "[remove_block_between_markers] failed to replace %s with temp output %s\n" "$target_file" "$tmp_file" >&2
		rm -f "$tmp_file"
		return 1
	}

	return 0
}

remove_exact_line_if_present() {
	local target_file="$1"
	local line_to_remove="$2"

	if [[ -z "$target_file" || ! -f "$target_file" ]]; then
		return 0
	fi

	local tmp_file
	tmp_file="$(mktemp)"

	awk -v line="$line_to_remove" '$0 != line { print }' "$target_file" > "$tmp_file" || {
		rm -f "$tmp_file"
		return 1
	}

	mv "$tmp_file" "$target_file" || {
		rm -f "$tmp_file"
		return 1
	}

	return 0
}

printf "This script will uninstall jaba by removing jaba's shell configuration and python environment. However, this will NOT remove any installations of python, brew or other system packages possibly installed by jaba. You will have to remove those manually if you want to. It will also NOT remove the jaba repository itself. Again, do this yourself if you want to (rm -r <jaba_directory>). The primary intent of this script is actually to clean jaba before reinstalling a newer version.\n"
read -p "Are you sure that you want to remove jaba? [y/n] " confirm_uninstall
if [[ "$confirm_uninstall" != "y" ]]; then
    printf "Uninstall cancelled.\n"
    exit 0
fi

### remove jaba shell block from bashrc
if remove_block_between_markers "$HOME/.bashrc" "$JABA_MARKER_START" "$JABA_MARKER_END"; then
	printf "Removed jaba shell block from %s (if present).\n" "$HOME/.bashrc"
else
	printf "Failed to update %s.\n" "$HOME/.bashrc"
	exit 1
fi


### remove python environment created/used by install
if [[ "$JABA_PYTHON_ENVIRONMENT_TYPE" == "conda" ]]; then
	if command -v "$JABA_CONDA_CMD" > /dev/null 2>&1; then
		if "$JABA_CONDA_CMD" env list | awk 'NR > 2 {print $1}' | sed 's/\*//g' | grep -Fxq "$JABA_PYTHON_ENVIRONMENT_NAME"; then
			# shellcheck disable=SC1091
			source "$($JABA_CONDA_CMD info --base)/etc/profile.d/conda.sh" || true
			while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
				$JABA_CONDA_CMD deactivate || break
			done
			"$JABA_CONDA_CMD" env remove -n "$JABA_PYTHON_ENVIRONMENT_NAME" -y || {
				printf "Failed to remove conda environment %s.\n" "$JABA_PYTHON_ENVIRONMENT_NAME"
				exit 1
			}
			printf "Removed conda environment %s.\n" "$JABA_PYTHON_ENVIRONMENT_NAME"
		else
			printf "Conda environment %s not found; skipping.\n" "$JABA_PYTHON_ENVIRONMENT_NAME"
		fi
	else
		printf "Conda command '%s' not found; skipping conda environment removal.\n" "$JABA_CONDA_CMD"
	fi
elif [[ "$JABA_PYTHON_ENVIRONMENT_TYPE" == "pip" ]]; then
	if [[ -d "${JABA_LOCATION}/.${JABA_PYTHON_ENVIRONMENT_NAME}" ]]; then
		rm -rf "${JABA_LOCATION}/.${JABA_PYTHON_ENVIRONMENT_NAME}" || {
			printf "Failed to remove pip environment at %s.\n" "$JABA_LOCATION/.${JABA_PYTHON_ENVIRONMENT_NAME}"
			exit 1
		}
		printf "Removed pip environment at %s.\n" "${JABA_LOCATION}/.${JABA_PYTHON_ENVIRONMENT_NAME}"
	else
		printf "Pip environment path not found (%s); skipping.\n" "${JABA_LOCATION}/.${JABA_PYTHON_ENVIRONMENT_NAME}"
	fi
else
	printf "Unknown environment type '%s'; skipping environment removal. Please update the uninstall script accordingly.\n" "$JABA_PYTHON_ENVIRONMENT_TYPE"
fi


printf "Uninstall complete.\n"
exit 0
