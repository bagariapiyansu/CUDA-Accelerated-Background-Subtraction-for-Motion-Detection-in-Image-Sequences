#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ================= CUDA GAUSSIAN BLUR =================
__global__ void gaussian_blur(unsigned char* input, unsigned char* output, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width-1 && y < height-1) {

        int kernel[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };

        int sum = 0;
        int weight = 16;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int pixel = input[(y + dy) * width + (x + dx)];
                sum += pixel * kernel[dy+1][dx+1];
            }
        }

        output[y * width + x] = sum / weight;
    }
}

// ================= MAIN =================
int main() {

    int width, height, channels;

    // Load JPG → force grayscale
    unsigned char* h_img = stbi_load("dataset/input/in000001.jpg", &width, &height, &channels, 1);

    if (!h_img) {
    printf("Error loading image: %s\n", stbi_failure_reason());
    return 1;
}

    int size = width * height;

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_img, size, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((width+15)/16, (height+15)/16);

    gaussian_blur<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    unsigned char* h_out = (unsigned char*)malloc(size);

    cudaMemcpy(h_out, d_output, size, cudaMemcpyDeviceToHost);

    // Save output JPG
    stbi_write_jpg("output.jpg", width, height, 1, h_out, 100);

    printf("Blur applied. Saved as output.jpg\n");

    // Cleanup
    stbi_image_free(h_img);
    free(h_out);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}