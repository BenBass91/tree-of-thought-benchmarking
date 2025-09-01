#!/bin/bash
# Script to run Python with the correct environment

# Use pyenv python if available, otherwise fall back to system python
if [ -f ~/.pyenv/versions/3.13.3/bin/python ]; then
    PYTHON=~/.pyenv/versions/3.13.3/bin/python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "Using Python: $PYTHON"
$PYTHON "$@"
