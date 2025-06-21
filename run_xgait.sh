#!/bin/bash
# Run XGait with proper environment settings for Apple Silicon

# Enable MPS fallback for torchvision operations not supported on MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the main application with any provided arguments
python3 main.py "$@"
