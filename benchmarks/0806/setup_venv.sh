#!/bin/bash

# Name of the virtual environment
VENV_NAME="myenv"

# Python version to use (default: python3)
PYTHON_VERSION="python3"

# Directory to create the virtual environment (default: current directory)
VENV_DIR="."

# Activate the virtual environment after creation (default: true)
ACTIVATE_AFTER_SETUP=true

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--name) VENV_NAME="$2"; shift ;;
        -p|--python) PYTHON_VERSION="$2"; shift ;;
        -d|--directory) VENV_DIR="$2"; shift ;;
        -a|--activate) ACTIVATE_AFTER_SETUP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Full path to the virtual environment
VENV_PATH="$VENV_DIR/$VENV_NAME"

# Create the virtual environment
echo "Creating virtual environment '$VENV_NAME' using $PYTHON_VERSION at $VENV_DIR..."
$PYTHON_VERSION -m venv "$VENV_PATH"

# Check if the virtual environment was created successfully
if [ $? -eq 0 ]; then
    echo "Virtual environment created successfully at $VENV_PATH."
else
    echo "Failed to create virtual environment."
    exit 1
fi

# Activate the virtual environment if requested
if [ "$ACTIVATE_AFTER_SETUP" = true ] ; then
    echo "Activating the virtual environment..."
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated. You are now in '$VENV_NAME'."
else
    echo "Virtual environment setup complete. To activate it, run:"
    echo "source $VENV_PATH/bin/activate"
fi
