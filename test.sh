#!/bin/sh

# Sequential test.
echo "Sequential code:"
time python3 sequential_julia_set.py -0.4 0.6
echo
mv Julia.tiff Sequential.tiff

# Multiprocessing test.
echo "Multiprocessing code:"
time python3 multiprocessing_julia_set.py -0.4 0.6
echo
mv Julia.tiff Multiprocessing.tiff

# GPU test.
echo "GPU code:"
export PYOPENCL_CTX=0 # Happens to be the integrated Intel GPU with beignet on my machine.
time python3 gpu_julia_set.py -0.4 0.6
echo
mv Julia.tiff GPU.tiff

# Check for differences.
diff Sequential.tiff Multiprocessing.tiff
diff Sequential.tiff GPU.tiff
