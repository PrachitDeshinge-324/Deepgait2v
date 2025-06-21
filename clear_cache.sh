#!/bin/bash

# Clear cache files in the current directory (e.g., *.cache, *.tmp, __pycache__)
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f \( -name "*.cache" -o -name "*.tmp" \) -delete

echo "Cache cleared."