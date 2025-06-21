#!/bin/bash

# Optimized XGait run script with performance improvements
echo "🚀 Starting optimized XGait inference..."

# Set PyTorch optimization environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Disable unnecessary warnings
export PYTHONWARNINGS="ignore:.*:UserWarning:.*"

# Torch compile compatibility
export TORCH_COMPILE_SUPPRESS_ERRORS=1
export PYTORCH_DISABLE_AUTOGRAD_SUBGRAPH_INLINING=1
export TORCH_DYNAMO_SUPPRESS_ERRORS=1


# Run with optimizations
echo "📊 Device info:"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"

echo "🎯 Starting video processing with optimizations..."
python main.py "$@"
