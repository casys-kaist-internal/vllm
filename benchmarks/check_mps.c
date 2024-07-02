#include <unistd.h>
#include <stdio.h>
#include <stdlib.h> // Include for atoi

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <unique_id>\n", argv[0]);
        return 1;
    }

    int unique_id = atoi(argv[1]); // Convert input argument to integer
    if (unique_id == 0 && argv[1][0] != '0')
    {
        fprintf(stderr, "Invalid unique ID. Please provide a valid number.\n");
        return 1;
    }

    // Construct the filename using the unique ID
    char filename[256]; // Allocate a buffer to hold the file path
    snprintf(filename, sizeof(filename), "/tmp/nvidia-mps-%d/control", unique_id);

    // Check if the file exists
    int result = access(filename, F_OK); // F_OK to check the existence of the file
    if (result == 0)
    {
        printf("%s MPS daemon is running!!\n", filename);
    }
    else
    {
        printf("%s MPS daemon doesn't exist!\n", filename);
    }
    return 0;
}
