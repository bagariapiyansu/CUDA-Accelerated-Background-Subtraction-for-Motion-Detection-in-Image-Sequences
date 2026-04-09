#include <stdio.h>
#include <cuda.h>

unsigned char* read_pgm(const char*, int*, int*);
void write_pgm(const char*, unsigned char*, int, int);

extern __global__ void frame_diff(unsigned char*, unsigned char*, unsigned char*, int);

int main() {
    int width, height;

    unsigned char* h_f1 = read_pgm("images/frame1.pgm", &width, &height);
    unsigned char* h_f2 = read_pgm("images/frame2.pgm", &width, &height);

    int size = width * height;

    unsigned char *d_f1, *d_f2, *d_out;

    cudaMalloc(&d_f1, size);
    cudaMalloc(&d_f2, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_f1, h_f1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f2, h_f2, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    frame_diff<<<blocks, threads>>>(d_f1, d_f2, d_out, size);

    unsigned char* h_out = (unsigned char*)malloc(size);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // simple threshold
    for (int i = 0; i < size; i++) {
        if (h_out[i] > 30) h_out[i] = 255;
        else h_out[i] = 0;
    }

    write_pgm("output/result.pgm", h_out, width, height);

    printf("Done\n");

    return 0;
}