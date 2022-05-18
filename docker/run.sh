#!/bin/bash

echo "Activating environment..."
source /root/.bash_aliases


# Preserve environment variables (slurm and some nvidia variables are set at runtime)
env | grep '^SLURM_\|^NVIDIA_' >> /etc/environment

# Disable python buffer for commands that are executed here as user "user". This ensures
# that the buffer is not disabled entirely but that commands executed in other contexts
# use the buffer, e.g., when the user logs in via ssh and executes a command.
echo "PYTHONUNBUFFERED=1" >> /etc/environment

# Switch to codebase (defaults to home directory)
if [ -z "$CODEBASE" ] || ! cd "$CODEBASE"; then
  cd /home/user/code/nnunet_pathology
fi
printf "Working directory: %s\n" "$(pwd)"

# Check if extra arguments were given and execute it as a command.
if [ -z "$2" ]; then
  # Print the command for logging.
  printf "No extra arguments given, running jupyter and sshd\n\n"

  # Start the SSH daemon and a Jupyter notebook.
  /usr/sbin/sshd
  sudo --user=user --set-home /bin/bash -c '/usr/local/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='
else
  # Print the command for logging.
  printf "Executing command: %s\n\n" "$*"

  # Execute the passed command.
  sudo --user=user --set-home "${@}"
fi
