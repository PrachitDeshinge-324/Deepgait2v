#!/bin/bash

cd /Users/prachit/self/Working/only_gait && rm -f test_*.py PARSING_INTEGRATION_SUMMARY.md
cd /Users/prachit/self/Working/only_gait && find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
cd /Users/prachit/self/Working/only_gait && find . -name "*.pyc" -delete 2>/dev/null || true
cd /Users/prachit/self/Working/only_gait && find . -name ".DS_Store" -delete 2>/dev/null || true