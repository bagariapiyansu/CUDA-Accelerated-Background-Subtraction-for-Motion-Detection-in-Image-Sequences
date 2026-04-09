#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ================= READ JPG =================
unsigned char* read_jpg(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* data = stbi_load(filename, width, height, &channels, 1);

    if (!data) {
        printf("Error loading image: %s\n", stbi_failure_reason());
        exit(1);
    }

    return data;
}

// ================= AVERAGE FRAMES =================
unsigned char* average_frames(char filenames[][100], int count, int* width, int* height) {

    unsigned int* sum = NULL;

    for (int i = 0; i < count; i++) {
        int w, h;
        unsigned char* img = read_jpg(filenames[i], &w, &h);

        if (i == 0) {
            *width = w;
            *height = h;
            sum = (unsigned int*)calloc(w * h, sizeof(unsigned int));
        }

        for (int j = 0; j < w * h; j++) {
            sum[j] += img[j];
        }

        stbi_image_free(img);
    }

    unsigned char* avg = (unsigned char*)malloc((*width) * (*height));

    for (int j = 0; j < (*width) * (*height); j++) {
        avg[j] = sum[j] / count;
    }

    free(sum);
    return avg;
}

// ================= CUDA KERNELS =================
__global__ void subtract(unsigned char* prev, unsigned char* curr, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int diff = curr[i] - prev[i];
        if (diff < 0) diff = -diff;
        out[i] = diff;
    }
}

__global__ void threshold(unsigned char* in, unsigned char* out, int n, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (in[i] > T) ? 255 : 0;
    }
}

__global__ void erosion(unsigned char* in, unsigned char* out, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (x == 0 || y == 0 || x == width-1 || y == height-1) {
        out[y * width + x] = in[y * width + x];
        return;
    }

    int min = 255;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int val = in[(y+dy)*width + (x+dx)];
            if (val < min) min = val;
        }
    }

    out[y * width + x] = min;
}

// ================= MAIN =================
int main() {

    int width, height;
    int window = 5;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 1; i <= 50; i += window * 2) {

        char group1[5][100];
        char group2[5][100];
        char out_name[100];

        for (int k = 0; k < window; k++) {
            sprintf(group1[k], "dataset/input/in%06d.jpg", i + k);
            sprintf(group2[k], "dataset/input/in%06d.jpg", i + window + k);
        }

        sprintf(out_name, "dataset/output/out%06d.jpg", i);

        // -------- AVERAGE FRAMES --------
        unsigned char* avg1 = average_frames(group1, window, &width, &height);
        unsigned char* avg2 = average_frames(group2, window, &width, &height);

        int size = width * height;

        unsigned char *d_prev, *d_curr, *d_diff, *d_thresh, *d_morph;

        cudaMalloc(&d_prev, size);
        cudaMalloc(&d_curr, size);
        cudaMalloc(&d_diff, size);
        cudaMalloc(&d_thresh, size);
        cudaMalloc(&d_morph, size);

        cudaMemcpy(d_prev, avg1, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_curr, avg2, size, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        dim3 t2(16,16);
        dim3 b2((width+15)/16, (height+15)/16);

        // -------- START TIMER --------
        cudaEventRecord(start);

        subtract<<<blocks, threads>>>(d_prev, d_curr, d_diff, size);
        threshold<<<blocks, threads>>>(d_diff, d_thresh, size, 5);
        erosion<<<b2, t2>>>(d_thresh, d_morph, width, height);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // -------- COPY BACK --------
        unsigned char* h_out = (unsigned char*)malloc(size);
        cudaMemcpy(h_out, d_morph, size, cudaMemcpyDeviceToHost);

        // -------- SAVE --------
        stbi_write_jpg(out_name, width, height, 1, h_out, 100);

        printf("Processed window starting at frame %d | Time: %.3f ms\n", i, ms);

        // -------- MOTION DETECTION --------
        int motion_pixels = 0;

        for (int j = 0; j < size; j++) {
            if (h_out[j] == 255) motion_pixels++;
        }

        float percent = (motion_pixels * 100.0f) / size;

        if (percent > 0.1f) {
            printf("Motion detected (%.3f%%)\n", percent);
        }

        // -------- CLEANUP --------
        free(avg1);
        free(avg2);
        free(h_out);

        cudaFree(d_prev);
        cudaFree(d_curr);
        cudaFree(d_diff);
        cudaFree(d_thresh);
        cudaFree(d_morph);
    }

    return 0;
}