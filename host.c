#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define SUMS_SIZE 10
#define NUM_WORK_ITEMS 10

int main()
{
    srand(time(NULL));
    int ARRAY_SIZE = 1000 + rand() % 9001; // 1000 to 10000
    int coordinatingProcessID  = rand() % 10; // 0 to 9
    int totalSum = 0;

    printf("Array Size: %d\nCo-ordinating Process ID : %d\n", ARRAY_SIZE, coordinatingProcessID);
    cl_int err; // variable to store error codes returned by OpenCL functions
    // set up OpenCL platform and device
    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, NULL);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

    // create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    //------------------------------------------------------------------------------------------------------------------------
    // creating input and output buffers
    int *array = (int *)malloc(sizeof(int) * ARRAY_SIZE);
    int sums[SUMS_SIZE]={0};
    int* displacements = (int*)malloc(sizeof(int) * SUMS_SIZE);
    
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        array[i] = 1;
    }
    int temp = 0;
    for (int i = 0; i < SUMS_SIZE; i++)
    {
        int random = rand()% (ARRAY_SIZE - temp);
        displacements[i] = temp;
        temp += random;
    }
    displacements[SUMS_SIZE - 1] = temp;

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, array, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * SUMS_SIZE, NULL, &err);
    cl_mem coordinatingProcessIdBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &coordinatingProcessID, &err);
    cl_mem displacementsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SUMS_SIZE, displacements, &err);
    cl_mem arraySizeBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &ARRAY_SIZE, &err);
    //------------------------------------------------------------------------------------------------------------------------
    // creating program and kernel
    // reading the kernel code from the file
    FILE *fp = fopen("myKernel.cl", "r");
    if (!fp)
    {
        printf("Failed to open file mykernel.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t kernelSize = ftell(fp);
    char *kernelSource = (char *)malloc(kernelSize + 1);
    rewind(fp);
    fread(kernelSource, sizeof(char), kernelSize, fp);
    fclose(fp);
    kernelSource[kernelSize] = '\0';

    // create program and kernel
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "myKernel", &err);

    // setting kernel arguments
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &coordinatingProcessIdBuffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &displacementsBuffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &arraySizeBuffer);

    size_t globalSize[1] = {NUM_WORK_ITEMS};
    size_t localSize[1] = {1};
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);

    // read output buffer back to host and print sums
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(int) * SUMS_SIZE, sums, 0, NULL, NULL);
    for (int i = 0; i < SUMS_SIZE; i++)
    {
        totalSum += sums[i];
        printf("sums[%d] = %d\n", i, sums[i]);
    }
    printf("\nTotal Sum: %d\n\n", totalSum);


    //print the displacements
    err = clEnqueueReadBuffer(queue, displacementsBuffer, CL_TRUE, 0, sizeof(int) * SUMS_SIZE, displacements, 0, NULL, NULL);
    for (int i = 0; i < SUMS_SIZE; i++)
    {
        printf("displacements[%d] = %d\n", i, displacements[i]);
    }
    // cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
