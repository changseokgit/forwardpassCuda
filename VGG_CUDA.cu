#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
using namespace std;

struct feature {
    int channel;
    int width;
    int height;
    size_t size;
    float* h_elements;
    float* d_elements;
};

struct kernel {
    int number;
    int channel;
    int width;
    int height;
    size_t size;
    float* h_elements;
    float* d_elements;
};

__global__ void Conv2dKernel(feature A, kernel B, feature C){
    float Cvalue = 0;

    int c = threadIdx.x;
    
    int n = blockIdx.x;
    int w = blockIdx.y;
    int h = blockIdx.z;

    extern __shared__ float channels[];

    channels[c] = 0;
    for (int i = 0; i < B.width; i++){
        for (int j = 0; j < B.height; j++){
            int idxA = c * A.height * A.width + (w + i) * A.height + h + j;
            int idxB = n * B.channel * B.width * B.height + c * B.width * B.height + i * B.width + j;
            channels[c] += A.d_elements[idxA] * B.d_elements[idxB];
        }
    }
    __syncthreads();

    for (int i = 0; i < B.channel; i++)
        Cvalue += channels[i];

    __syncthreads();

    C.d_elements[n * gridDim.y * gridDim.z + w * gridDim.z + h] = Cvalue;

}

__global__ void AddBiasKernel(feature A, kernel B){
    int c = blockIdx.x;
    int w = blockIdx.y;
    int h = blockIdx.z;
    
    int idxA = c * A.width*A.height + w * A.height + h;

    A.d_elements[idxA] += B.d_elements[c];
}

__global__ void paddingKernel(float* A, feature B, int padding){
    // A is original array, B is padded array
    int c = blockIdx.x;
    int w = blockIdx.y;
    int h = blockIdx.z;
    
    int idxA = c * gridDim.y*gridDim.z + w * gridDim.z + h;
    int idxB = c * B.width*B.height + (w + padding) * B.height + (h + padding);
    B.d_elements[idxB] = A[idxA];
}

__global__ void ReLUKernel(feature A){
    int c = blockIdx.x;
    int w = blockIdx.y;
    int h = blockIdx.z;
    
    int idxA = c * A.width*A.height + w * A.height + h;

    if (A.d_elements[idxA] < 0)
        A.d_elements[idxA] = 0;
}

__global__ void MaxpoolingKernel(feature A, feature B, int filter){
    // A is input B is output
    int c = blockIdx.x;
    int w = blockIdx.y;
    int h = blockIdx.z;

    float max_value = A.d_elements[c * A.width*A.height + w*filter * A.height + h*filter];

    for (int i = 0; i < filter; i++){
        for (int j = 0; j < filter; j++){
            int idxA = c * A.height * A.width + (w*filter + i) * A.height + h*filter + j;
            if (max_value < A.d_elements[idxA])
                max_value = A.d_elements[idxA];
        }
    }

    int idxB = c * gridDim.y * gridDim.z + w * gridDim.z + h;
    B.d_elements[idxB] = max_value;


}

void init_feature(feature& A, int channel, int width, int height, float init_value = 0);
void init_kernel(kernel& A, int number, int channel, int width, int height, float init_value = 0);
void init_feature(feature& A, int channel, int width, int height, string filename);
void init_kernel(kernel& A, int number, int channel, int width, int height, string filename);

void fprint(feature data);
void kprint(kernel data);

void make_pad(feature& A, int pad);
void ReLU(feature& A);
void convolution(feature& A, kernel B, int stride);
void addBias(feature& A, kernel B);
void maxPooling(feature& A, int filter);
void view(feature& A);

int main()
{
    //feature A initialization
    feature A;
    init_feature(A, 3, 8, 8, 1);
    cudaMalloc(&A.d_elements, A.size);
    cudaMemcpy(A.d_elements, A.h_elements, A.size, cudaMemcpyHostToDevice);
    fprint(A);

    //kernel B initialization
    kernel B;
    init_kernel(B, 3, 3, 3, 3, 1);
    cudaMalloc(&B.d_elements, B.size);
    cudaMemcpy(B.d_elements, B.h_elements, B.size, cudaMemcpyHostToDevice);
    kprint(B);

    //kernel C initialization
    kernel C;
    init_kernel(C, 3, 1, 1, 1, 1);
    cudaMalloc(&C.d_elements, C.size);
    cudaMemcpy(C.d_elements, C.h_elements, C.size, cudaMemcpyHostToDevice);
    kprint(C);

    //calculation
    make_pad(A, 1);
    convolution(A, B, 1);
    addBias(A,C);
    ReLU(A);
    maxPooling(A, 2);
    view(A);



    //receive data from device
    delete A.h_elements;
    A.size = A.channel * A.width * A.height * sizeof(float);    
    A.h_elements = new float [A.size];
    cudaMemcpy(A.h_elements, A.d_elements, A.size, cudaMemcpyDeviceToHost);
    
    //print data
    fprint(A);
    
    //release all cuda memories
    cudaFree(A.d_elements);
    cudaFree(B.d_elements);
    cudaFree(C.d_elements);
    return 0;
}

void init_feature(feature& A, int channel, int width, int height, float init_value){
    A.channel = channel;
    A.width = width;
    A.height = height;
    size_t size = channel * width * height;
    A.h_elements = new float [size];
    if (init_value != 0)
        for (int i = 0; i < size; i++)
            A.h_elements[i] = init_value;
    A.size = size*sizeof(float);
}

void init_kernel(kernel& A, int number, int channel, int width, int height, float init_value){
    A.number = number;
    A.channel = channel;
    A.width = width;
    A.height = height;
    size_t size = number * channel * width * height;
    A.h_elements = new float [size];
    if (init_value != 0)
        for (int i = 0; i < size; i++)
            A.h_elements[i] = init_value;
    A.size = size*sizeof(float);
}

void init_feature(feature& A, int channel, int width, int height, string filename){
    A.channel = channel;
    A.width = width;
    A.height = height;
    size_t size = channel * width * height;
    A.h_elements = new float [size];

    // if (init_value != 0)
    //     for (int i = 0; i < size; i++)
    //         A.h_elements[i] = init_value;

    A.size = size*sizeof(float);
}

void init_kernel(kernel& A, int number, int channel, int width, int height, string filename){
    A.number = number;
    A.channel = channel;
    A.width = width;
    A.height = height;
    size_t size = number * channel * width * height;
    A.h_elements = new float [size];

    // if (init_value != 0)
    //     for (int i = 0; i < size; i++)
    //         A.h_elements[i] = init_value;

    A.size = size*sizeof(float);
}

void fprint(feature data){
    cout << "feature type \n[ " << data.channel << ", " << data.width << ", " << data.height << " ]" << endl;
    for (int i = 0; i < data.channel; i++){
        for (int j = 0; j < data.width; j++){
            for (int k = 0; k < data.height; k++){
                cout << setw(3) << data.h_elements[i * data.width*data.height + j * data.height + k];
            }
            cout << endl;
        }
        cout << endl;
    }
}
    
void kprint(kernel data){
    cout << "kernel type \n[ " << data.number << ", " << data.channel << ", " << data.width << ", " << data.height << " ]" << endl;
    for (int i = 0; i < data.number; i++){
        for (int j = 0; j < data.channel; j++){
            for (int k = 0; k < data.width; k++){
                for (int l = 0; l < data.height; l++){
                    cout << setw(3) << data.h_elements[i * data.channel*data.width*data.height + j * data.width*data.height + k * data.height + l];
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
}

void make_pad(feature& A, int pad){
    //invoke kernel padding
    float* temp;
    cudaMalloc(&temp, A.size);
    cudaMemcpy(temp, A.d_elements, A.size, cudaMemcpyDeviceToDevice);
    dim3 dimGrid(A.channel, A.width, A.height);
    A.height += 2*pad; A.width += 2*pad;
    A.size = A.channel * A.width * A.height * sizeof(float);
    cudaFree(A.d_elements);
    cudaMalloc(&A.d_elements, A.size);
    cudaMemset(A.d_elements, 0, A.size);
    paddingKernel<<<dimGrid, 1>>> (temp, A, pad);
    cudaFree(temp);
}

void ReLU(feature& A){
    //invoke kernel relu
    dim3 dimGrid(A.channel, A.width, A.height);
    ReLUKernel<<<dimGrid, 1>>>(A);
}

void convolution(feature& A, kernel B, int stride){
    feature temp;
    init_feature(temp, B.number, A.width - B.width + 1, A.height - B.height + 1, 0);
    cudaMalloc(&temp.d_elements, temp.size);

    dim3 dimGrid(temp.channel, temp.width, temp.height);
    Conv2dKernel<<<dimGrid, A.channel, A.channel>>>(A, B, temp);
    
    cudaFree(A.d_elements);
    cudaMalloc(&A.d_elements, temp.size);
    cudaMemcpy(A.d_elements, temp.d_elements, temp.size, cudaMemcpyDeviceToDevice);
    delete temp.h_elements;
    cudaFree(temp.d_elements);

    A.channel = temp.channel;
    A.width = temp.width;
    A.height = temp.height;
}

void addBias(feature& A, kernel B){
    dim3 dimGrid(A.channel, A.width, A.height);
    AddBiasKernel<<<dimGrid, 1>>>(A, B);
}

void maxPooling(feature& A, int filter){
    feature temp;
    init_feature(temp, A.channel, A.width / filter, A.height / filter);
    cudaMalloc(&temp.d_elements, temp.size);

    dim3 dimGrid(temp.channel, temp.width, temp.height);
    MaxpoolingKernel<<<dimGrid, 1>>>(A, temp, filter);

    cudaFree(A.d_elements);
    cudaMalloc(&A.d_elements, temp.size);
    cudaMemcpy(A.d_elements, temp.d_elements, temp.size, cudaMemcpyDeviceToDevice);
    delete temp.h_elements;
    cudaFree(temp.d_elements);

    A.channel = temp.channel;
    A.width = temp.width;
    A.height = temp.height;
}

void view(feature& A){
    A.channel = A.channel*A.width*A.height;
    A.width = 1;
    A.height = 1;
}
