#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ================= READ JPG =================
unsigned char* read_jpg(const char *filename, int *width, int *height) {
    int channels;
    unsigned char *data = stbi_load(filename, width, height, &channels, 1);

    if (!data) {
        printf("Warning: Failed to load %s (%s)\n", filename, stbi_failure_reason());
        return NULL;
    }

    return data;
}

// ================= CUDA AVERAGING =================
__global__ void average_kernel(unsigned char* images, unsigned char* avg,
                               int num_images, int size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        int sum = 0;

        for (int k = 0; k < num_images; k++) {
            sum += images[k * size + i];
        }

        avg[i] = sum / num_images;
    }
}

// ================= GPU-BASED AVERAGE WINDOW =================
unsigned char* average_window(int start, int window, int* width, int* height) {

    unsigned char* h_images = NULL;
    int valid_count = 0;
    int fail_streak = 0;

    for (int k = 0; k < window; k++) {

        char filename[100];
        sprintf(filename, "dataset/input/in%06d.jpg", start + k);

        int w, h;
        unsigned char* img = read_jpg(filename, &w, &h);

        if (img == NULL) {
            fail_streak++;

            if (fail_streak >= 2) {
                printf("Error: 2 consecutive missing files. Stopping window.\n");
                break;
            }

            continue;
        }

        fail_streak = 0;

        if (valid_count == 0) {
            *width = w;
            *height = h;
            h_images = (unsigned char*)malloc(window * w * h);
        }

        memcpy(h_images + valid_count * (*width) * (*height),
               img, (*width) * (*height));

        valid_count++;

        stbi_image_free(img);
    }

    if (valid_count == 0) {
        printf("Error: No valid images in window.\n");
        return NULL;
    }

    int size = (*width) * (*height);

    unsigned char *d_images, *d_avg;

    cudaMalloc(&d_images, valid_count * size);
    cudaMalloc(&d_avg, size);

    cudaMemcpy(d_images, h_images,
               valid_count * size,
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    average_kernel<<<blocks, threads>>>(d_images, d_avg, valid_count, size);

    unsigned char* avg = (unsigned char*)malloc(size);

    cudaMemcpy(avg, d_avg, size, cudaMemcpyDeviceToHost);

    cudaFree(d_images);
    cudaFree(d_avg);
    free(h_images);

    return avg;
}

// ================= SUBTRACTION =================
__global__ void subtract(unsigned char *prev, unsigned char *curr, unsigned char *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int diff = curr[i] - prev[i];
        if (diff < 0) diff = -diff;
        out[i] = diff;
    }
}

// ================= THRESHOLD =================
__global__ void threshold(unsigned char *in, unsigned char *out, int n, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (in[i] > T) ? 255 : 0;
    }
}

// ================= MORPHOLOGY =================
__global__ void erosion(unsigned char *in, unsigned char *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        out[y * width + x] = in[y * width + x];
        return;
    }

    int min = 255;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int val = in[(y + dy) * width + (x + dx)];
            if (val < min) min = val;
        }
    }
    out[y * width + x] = min;
}

// ================= MAIN =================
int main() {

    int num = 0;
    float avg = 0;
    int width, height;

    const int window = 7;
    const int lower_bound = 2;
    int upper_bound = 140;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = lower_bound; i <= upper_bound; i += window) {

        char out_name[100];
        sprintf(out_name, "dataset/output/out%06d.jpg", i);

        unsigned char *h_prev = average_window(i - 1, window, &width, &height);
        unsigned char *h_curr = average_window(i, window, &width, &height);

        if (h_prev == NULL || h_curr == NULL) {
            printf("Skipping frame %d due to insufficient data\n", i);
            continue;
        }

        int size = width * height;

        unsigned char *d_prev, *d_curr, *d_diff, *d_thresh, *d_morph;

        cudaMalloc(&d_prev, size);
        cudaMalloc(&d_curr, size);
        cudaMalloc(&d_diff, size);
        cudaMalloc(&d_thresh, size);
        cudaMalloc(&d_morph, size);

        cudaMemcpy(d_prev, h_prev, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_curr, h_curr, size, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        dim3 t2(16, 16);
        dim3 b2((width + 15) / 16, (height + 15) / 16);

        cudaEventRecord(start);

        subtract<<<blocks, threads>>>(d_prev, d_curr, d_diff, size);
        threshold<<<blocks, threads>>>(d_diff, d_thresh, size, 5);
        erosion<<<b2, t2>>>(d_thresh, d_morph, width, height);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        unsigned char *h_diff = (unsigned char*)malloc(size);
        cudaMemcpy(h_diff, d_diff, size, cudaMemcpyDeviceToHost);

        stbi_write_jpg(out_name, width, height, 1, h_diff, 100);

        printf("Processed window starting at frame %d | Time: %.3f ms | Saved: %s\n",
               i, ms, out_name);

        unsigned char *h_thresh = (unsigned char*)malloc(size);
        cudaMemcpy(h_thresh, d_thresh, size, cudaMemcpyDeviceToHost);

        int motion_pixels = 0;

        for (int j = 0; j < size; j++) {
            if (h_thresh[j] == 255) motion_pixels++;
        }

        float percent = (motion_pixels * 100.0f) / size;
        num++;

        if (percent > 0.1f) {
            avg += percent;
            printf("Motion detected (%.3f%% pixels)\n", percent);
        }

        stbi_image_free(h_prev);
        stbi_image_free(h_curr);

        free(h_diff);
        free(h_thresh);

        cudaFree(d_prev);
        cudaFree(d_curr);
        cudaFree(d_diff);
        cudaFree(d_thresh);
        cudaFree(d_morph);
    }

    printf("Average motion percentage till the last frame considered: %.3f%% pixels\n", avg / num);

    return 0;
}