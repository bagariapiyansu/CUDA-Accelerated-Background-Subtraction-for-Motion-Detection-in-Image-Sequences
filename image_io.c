#include <stdio.h>
#include <stdlib.h>

unsigned char *read_pgm(const char *filename, int *width, int *height)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        printf("Error opening file\n");
        return NULL;
    }

    char format[3];
    fscanf(f, "%s", format);
    fscanf(f, "%d %d", width, height);

    int maxval;
    fscanf(f, "%d", &maxval);
    fgetc(f); // skip newline

    int size = (*width) * (*height);
    unsigned char *data = (unsigned char *)malloc(size);

    fread(data, sizeof(unsigned char), size, f);
    fclose(f);

    return data;
}

void write_pgm(const char *filename, unsigned char *data, int width, int height)
{
    FILE *f = fopen(filename, "wb");

    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width * height, f);

    fclose(f);
}