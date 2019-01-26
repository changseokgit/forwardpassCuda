rm -rf run
clear
nvcc -o run VGG_CUDA.cu
./run
